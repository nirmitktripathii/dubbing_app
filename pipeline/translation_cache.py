"""
translation_cache.py — Stage 3 efficiency layer (v2.5.1)

Two independent, best-effort persistence helpers that make the Gemini
translation stage cheaper and friendlier to free-tier API rate limits:

1. TranslationCache — a persistent pool of candidate translations keyed by
   (language, source_text). Because candidate SELECTION is local and free
   (real phonemes via espeak-ng + IndicSBERT semantics), a cached candidate
   pool lets a re-run of the same video select its final lines with ZERO API
   calls. Repeated phrases across a single video are also served from cache
   after the first generation. The pool ACCUMULATES across runs, so quality
   only improves.

2. Daily usage tracking — a per-day, per-model request counter persisted to
   disk. It powers an optional hard cap on requests-per-day (RPD) so a long
   job degrades with a clear error instead of hammering the API into 429s.

Design rule: NEVER abort the pipeline. Every disk operation is wrapped; on any
IO/JSON failure the layer disables itself VISIBLY (prints once) and behaves as
an empty cache / no-op counter. This mirrors the rest of v2.5: degrade openly,
never silently.

Locations (all overridable by env):
  DUBBING_CACHE_DIR           base dir for cache files   (default: ./.dubbing_cache)
  DUBBING_TRANSLATION_CACHE   full path to the candidate cache json
  (usage file lives at <cache_dir>/gemini_usage.json)
"""

import os
import json
import time
import hashlib
import threading
from datetime import date
from typing import List, Optional, Dict

# Bound the on-disk pool so a huge corpus can't grow the file without limit.
_MAX_CANDIDATES_PER_KEY = 12

_LOCK = threading.RLock()


def _cache_dir() -> str:
    base = os.environ.get("DUBBING_CACHE_DIR") or os.path.join(os.getcwd(), ".dubbing_cache")
    os.makedirs(base, exist_ok=True)
    return base


def _translation_cache_path() -> str:
    override = os.environ.get("DUBBING_TRANSLATION_CACHE")
    if override:
        d = os.path.dirname(override)
        if d:
            os.makedirs(d, exist_ok=True)
        return override
    return os.path.join(_cache_dir(), "translations.json")


def _usage_path() -> str:
    return os.path.join(_cache_dir(), "gemini_usage.json")


def _atomic_write(path: str, payload: dict) -> None:
    """Write JSON atomically (tmp file + os.replace) so a crash mid-write can
    never corrupt the cache."""
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    os.replace(tmp, path)


def _key(lang: str, text: str) -> str:
    raw = f"{(lang or '').strip().lower()}\x00{(text or '').strip()}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


# ── Candidate cache ────────────────────────────────────────────────────────

class TranslationCache:
    """Persistent pool of candidate translations, keyed by (language, source)."""

    def __init__(self, path: Optional[str] = None, enabled: bool = True):
        self.path = path or _translation_cache_path()
        self.enabled = enabled
        self._data: Dict[str, dict] = {}
        self._dirty = False
        self._loaded = False
        self._broken = False

    def _load(self) -> None:
        if self._loaded or not self.enabled:
            return
        self._loaded = True
        try:
            if os.path.exists(self.path):
                with open(self.path, "r", encoding="utf-8") as f:
                    self._data = json.load(f) or {}
        except Exception as e:  # noqa: BLE001 — cache must never crash the run
            self._broken = True
            self.enabled = False
            print(f"[TranslationCache] disabled — could not read {self.path}: {e}")

    def get(self, lang: str, text: str) -> List[str]:
        if not self.enabled:
            return []
        with _LOCK:
            self._load()
            entry = self._data.get(_key(lang, text))
            if not entry:
                return []
            return list(entry.get("candidates", []))

    def add(self, lang: str, text: str, candidates: List[str]) -> None:
        if not self.enabled or not text or not candidates:
            return
        clean = [c.strip() for c in candidates if isinstance(c, str) and c.strip()]
        if not clean:
            return
        with _LOCK:
            self._load()
            k = _key(lang, text)
            entry = self._data.get(k) or {"lang": lang, "text": text, "candidates": []}
            pool = entry["candidates"]
            for c in clean:
                if c not in pool:
                    pool.append(c)
            # Keep the most recent N (newer refinements are usually better).
            entry["candidates"] = pool[-_MAX_CANDIDATES_PER_KEY:]
            self._data[k] = entry
            self._dirty = True

    def save(self) -> None:
        if not self.enabled or not self._dirty or self._broken:
            return
        with _LOCK:
            try:
                _atomic_write(self.path, self._data)
                self._dirty = False
            except Exception as e:  # noqa: BLE001
                self.enabled = False
                print(f"[TranslationCache] disabled — could not write {self.path}: {e}")

    def stats(self) -> dict:
        with _LOCK:
            self._load()
            return {"enabled": self.enabled, "keys": len(self._data), "path": self.path}


# ── Daily usage tracking (RPD guard) ───────────────────────────────────────

_usage_cache: Optional[dict] = None


def _load_usage() -> dict:
    global _usage_cache
    if _usage_cache is not None:
        return _usage_cache
    data = {}
    try:
        p = _usage_path()
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f) or {}
    except Exception:  # noqa: BLE001 — usage tracking is best-effort
        data = {}
    _usage_cache = data
    return data


def _today() -> str:
    return date.today().isoformat()


def count_today(model: str) -> int:
    with _LOCK:
        data = _load_usage()
        return int(data.get(_today(), {}).get(model, 0))


def record_request(model: str) -> int:
    """Increment today's counter for `model` and persist. Returns the new count.
    Best-effort: a persistence failure is swallowed (counter still advances in
    memory for this process)."""
    with _LOCK:
        data = _load_usage()
        day = _today()
        bucket = data.setdefault(day, {})
        bucket[model] = int(bucket.get(model, 0)) + 1
        # Prune old days so the file stays tiny.
        if len(data) > 14:
            for old in sorted(data.keys())[:-14]:
                data.pop(old, None)
        try:
            _atomic_write(_usage_path(), data)
        except Exception:  # noqa: BLE001
            pass
        return bucket[model]


def rpd_limit() -> Optional[int]:
    """Per-model requests-per-day cap from DUBBING_GEMINI_RPD, or None (unset)."""
    raw = os.environ.get("DUBBING_GEMINI_RPD")
    if not raw:
        return None
    try:
        v = int(raw)
        return v if v > 0 else None
    except ValueError:
        return None
