"""Native reference voices for Basic-mode TTS — one per target language.

WHY THIS EXISTS
IndicF5 is a zero-shot cloner: it copies the reference clip's voice AND its accent. Basic mode
used one Hindi clip (sumedhu/hindi-emotion-voice-references, HIN_M_HAPPY_00057) as the
reference for all eleven languages, so a Tamil dub was "a Hindi speaker reading Tamil" — the
same cross-lingual mismatch that produced the onset babble in the English-reference runs.

That HF repo is a repack of AI4Bharat Rasa (ai4bharat/Rasa, CC BY 4.0), which has native
speakers for every language we dub. This module picks ONE clip per language from Rasa by
pre-registered rules, stores it with its transcript and provenance, and resolves which
reference a run uses.

WHAT IS CHECKED (the property, not the machinery — CLAUDE.md rule 5)
  * the row is labelled as the target language, and its transcript is in that language's own
    script (≥ MIN_SCRIPT_FRACTION of letters);
  * the clip is in the length band IndicF5 handles well (after trimming edge silence);
  * it is audible (RMS floor) and not clipped;
  * the file's real duration matches the dataset's stated duration (the audio belongs to the row);
  * the speaking rate implied by transcript/duration is plausible (the text belongs to the audio).
A clip that cannot be checked is rejected, not accepted.

RESOLUTION LADDER (resolve_basic_reference)
  1. a curated native clip for the language (manifest + sha256 verified)      source="native"
  2. Hindi → the pinned sumedhu clip, exactly as before                       source="pinned"
  3. any other language with no native clip → the Hindi clip, with a WARNING  source="fallback"
  4. HF unreachable → text-only reference (previous behaviour)                source="text-only"
The chosen source is always logged, so a fallback is visible in the run log.

Curation needs the Rasa terms accepted on HF by the account whose HF_TOKEN is used. Run it on
Modal (CPU):  modal run deploy/modal_app.py::curate_voice_refs
Module import is light (stdlib only); numpy/soundfile/pyarrow load inside the functions that
need them, so the API container can read coverage without the audio stack.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import time
import unicodedata
from typing import Callable, Optional

# ── Sources ───────────────────────────────────────────────────────────────────────────────
# The pinned Hindi clip (unchanged; transcription verified via Whisper small on 2026-07-05).
DEFAULT_HINDI_REF_REPO = "sumedhu/hindi-emotion-voice-references"
DEFAULT_HINDI_REF_FILE = "hindi_best_clips/male/happy/HIN_M_HAPPY_00057.wav"
DEFAULT_HINDI_REF_TEXT = "सुबह केरल चाय का एक गिलास मुझे तरो ताजा घर देता है"

RASA_REPO = "ai4bharat/Rasa"
RASA_LICENSE = "CC-BY-4.0"
RASA_ATTRIBUTION = "AI4Bharat, Rasa: expressive Indic TTS dataset (huggingface.co/datasets/ai4bharat/Rasa)"

# Language code -> Rasa config/directory name (identical to our display names).
CODE_TO_LANGUAGE = {
    "hi": "Hindi", "bn": "Bengali", "mr": "Marathi", "gu": "Gujarati", "pa": "Punjabi",
    "ta": "Tamil", "te": "Telugu", "kn": "Kannada", "ml": "Malayalam", "or": "Odia",
    "as": "Assamese",
}

# Unicode blocks of each language's own script. Bengali and Assamese share a block, as do
# Hindi and Marathi — the check is "right script", not "right language".
SCRIPT_RANGES = {
    "hi": (0x0900, 0x097F), "mr": (0x0900, 0x097F),
    "bn": (0x0980, 0x09FF), "as": (0x0980, 0x09FF),
    "pa": (0x0A00, 0x0A7F), "gu": (0x0A80, 0x0AFF), "or": (0x0B00, 0x0B7F),
    "ta": (0x0B80, 0x0BFF), "te": (0x0C00, 0x0C7F), "kn": (0x0C80, 0x0CFF),
    "ml": (0x0D00, 0x0D7F),
}

# ── Pre-registered selection rules (CLAUDE.md rule 9: written before any clip is seen) ─────
MIN_SECONDS = 3.5            # shorter refs give IndicF5 too little voice to copy
MAX_SECONDS = 9.0            # F5 truncates long refs, desyncing the transcript
TARGET_SECONDS = 6.0         # tie-break: closest to this wins
MIN_SCRIPT_FRACTION = 0.90   # letters in the target script / all letters
MIN_TEXT_LETTERS = 15        # a reference transcript needs enough text to align
MIN_RMS_DBFS = -35.0         # audible
MAX_CLIP_FRACTION = 0.001    # samples at |x| >= 0.999
DURATION_TOLERANCE_S = 0.5   # |file duration - stated duration|
LETTERS_PER_SECOND = (5.0, 25.0)   # transcript letters / trimmed seconds; outside = mismatch
EDGE_SILENCE_DB = -40.0      # below peak; trimmed from both ends, EDGE_PAD_S kept
EDGE_PAD_S = 0.10
PREFERRED_STYLES = ("neutral", "book", "conv", "news", "wiki", "happy")
DEFAULT_GENDER = os.environ.get("DUBBING_VOICE_GENDER", "male").strip().lower() or "male"
MAX_AUDIO_ROW_GROUPS = 6     # bound on audio pulled per language (~55 MB per Rasa row group)
READ_BLOCK = 256 * 1024      # fsspec read block for the remote parquet files

ENV_DIR = "DUBBING_VOICE_REF_DIR"
MANIFEST_NAME = "manifest.json"
SCHEMA_VERSION = 1


def criteria() -> dict:
    """The rules above as a dict — written into the manifest so a clip records what it passed."""
    return {
        "min_seconds": MIN_SECONDS, "max_seconds": MAX_SECONDS, "target_seconds": TARGET_SECONDS,
        "min_script_fraction": MIN_SCRIPT_FRACTION, "min_text_letters": MIN_TEXT_LETTERS,
        "min_rms_dbfs": MIN_RMS_DBFS, "max_clip_fraction": MAX_CLIP_FRACTION,
        "duration_tolerance_s": DURATION_TOLERANCE_S, "letters_per_second": list(LETTERS_PER_SECOND),
        "edge_silence_db": EDGE_SILENCE_DB, "preferred_styles": list(PREFERRED_STYLES),
    }


def ref_dir(override: Optional[str] = None) -> Optional[str]:
    return override or os.environ.get(ENV_DIR) or None


# ── Text checks ───────────────────────────────────────────────────────────────────────────
def _letters(text: str) -> list:
    # Letters AND combining marks: Indic vowel signs (matras) are category M, not L.
    return [c for c in text if unicodedata.category(c)[0] in ("L", "M")]


def script_fraction(text: str, lang_code: str) -> float:
    """Fraction of letters in ``text`` that lie in ``lang_code``'s script block (0.0 if none)."""
    lo, hi = SCRIPT_RANGES[lang_code]
    letters = _letters(text or "")
    if not letters:
        return 0.0
    return sum(1 for c in letters if lo <= ord(c) <= hi) / len(letters)


def _parse_seconds(value) -> Optional[float]:
    try:
        v = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return v if v > 0 else None


def metadata_ok(row: dict, lang_code: str) -> tuple:
    """Cheap pre-filter on dataset metadata only (no audio). Returns (ok, reason)."""
    stated_lang = str(row.get("language") or "").strip().lower()
    name = CODE_TO_LANGUAGE[lang_code].lower()
    if stated_lang and not (stated_lang == lang_code or stated_lang.startswith(name)):
        return False, f"language_mismatch={stated_lang}"
    text = (row.get("text") or "").strip()
    if len(_letters(text)) < MIN_TEXT_LETTERS:
        return False, "text_too_short"
    frac = script_fraction(text, lang_code)
    if frac < MIN_SCRIPT_FRACTION:
        return False, f"script_fraction={frac:.2f}"
    dur = _parse_seconds(row.get("duration"))
    if dur is None:
        return False, "duration_unparsable"   # cannot run the tolerance check -> not a pass
    # The band is re-checked on the trimmed audio; allow edge silence here.
    if dur < MIN_SECONDS or dur > MAX_SECONDS + 2.0:
        return False, f"duration={dur:.2f}s"
    return True, ""


def _style_rank(style: str) -> int:
    s = (style or "").lower()
    for i, pref in enumerate(PREFERRED_STYLES):
        if pref in s:
            return i
    return len(PREFERRED_STYLES)


def rank_key(row: dict, gender: str = DEFAULT_GENDER) -> tuple:
    """Deterministic ordering: wanted gender, calm style, length near target, then filename."""
    g = (row.get("gender") or "").strip().lower()
    dur = _parse_seconds(row.get("duration")) or 0.0
    return (0 if g[:1] == gender[:1] else 1,          # "male"/"Male"/"M" all match "male"
            _style_rank(row.get("style", "")),
            round(abs(dur - TARGET_SECONDS), 3),
            str(row.get("filename", "")))


# ── Audio checks ──────────────────────────────────────────────────────────────────────────
def _to_mono_float(samples):
    import numpy as np
    x = np.asarray(samples, dtype=np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1)
    return x


def trim_edge_silence(x, sr: int):
    """Trim leading/trailing audio quieter than EDGE_SILENCE_DB below peak (keeps EDGE_PAD_S)."""
    import numpy as np
    if x.size == 0:
        return x
    peak = float(np.max(np.abs(x)))
    if peak <= 0:
        return x[:0]
    thresh = peak * (10.0 ** (EDGE_SILENCE_DB / 20.0))
    hop = max(1, int(sr * 0.01))
    n = x.size // hop
    if n == 0:
        return x
    frames = np.abs(x[: n * hop]).reshape(n, hop).max(axis=1)
    loud = np.nonzero(frames >= thresh)[0]
    if loud.size == 0:
        return x[:0]
    pad = int(EDGE_PAD_S * sr)
    start = max(0, loud[0] * hop - pad)
    end = min(x.size, (loud[-1] + 1) * hop + pad)
    return x[start:end]


def check_clip(samples, sr: int, text: str, lang_code: str, stated_seconds) -> tuple:
    """Check one decoded clip against every rule. Returns (ok, reasons, stats, trimmed)."""
    import numpy as np
    x = _to_mono_float(samples)
    reasons = []
    raw_s = x.size / float(sr) if sr else 0.0
    stated = _parse_seconds(stated_seconds)
    if stated is None:
        reasons.append("stated_duration_unknown")
    elif abs(raw_s - stated) > DURATION_TOLERANCE_S:
        reasons.append(f"file_vs_stated={raw_s:.2f}/{stated:.2f}s")

    clip_frac = float(np.mean(np.abs(x) >= 0.999)) if x.size else 1.0
    if clip_frac > MAX_CLIP_FRACTION:
        reasons.append(f"clipped={clip_frac:.4f}")

    t = trim_edge_silence(x, sr)
    dur = t.size / float(sr) if sr else 0.0
    if not (MIN_SECONDS <= dur <= MAX_SECONDS):
        reasons.append(f"trimmed_duration={dur:.2f}s")
    rms = float(np.sqrt(np.mean(t.astype(np.float64) ** 2))) if t.size else 0.0
    rms_db = 20.0 * np.log10(rms) if rms > 0 else -200.0
    if rms_db < MIN_RMS_DBFS:
        reasons.append(f"rms={rms_db:.1f}dBFS")

    frac = script_fraction(text, lang_code)
    if frac < MIN_SCRIPT_FRACTION:
        reasons.append(f"script_fraction={frac:.2f}")
    lps = len(_letters(text)) / dur if dur > 0 else 0.0
    if not (LETTERS_PER_SECOND[0] <= lps <= LETTERS_PER_SECOND[1]):
        reasons.append(f"letters_per_second={lps:.1f}")

    stats = {"file_seconds": round(raw_s, 3), "stated_seconds": stated,
             "trimmed_seconds": round(dur, 3), "rms_dbfs": round(float(rms_db), 2),
             "clip_fraction": round(clip_frac, 6), "script_fraction": round(frac, 4),
             "letters_per_second": round(lps, 2), "sample_rate": int(sr)}
    return (not reasons), reasons, stats, t


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


# ── Curation (reads Rasa; CPU) ────────────────────────────────────────────────────────────
META_COLUMNS = ["filename", "text", "language", "gender", "style", "duration"]


def _rasa_shards(language: str, token: Optional[str]):
    """(fs, sorted shard paths) for one Rasa language directory."""
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem(token=token)
    paths = sorted(p for p in fs.glob(f"datasets/{RASA_REPO}/{language}/*.parquet"))
    return fs, paths


def select_from_parquet(open_file: Callable, shard_paths: list, lang_code: str,
                        gender: str = DEFAULT_GENDER, log: Callable = print,
                        max_audio_row_groups: int = MAX_AUDIO_ROW_GROUPS) -> dict:
    """Pick the best passing clip from the given parquet shards.

    Reads metadata columns for every row group first (cheap), ranks the rows that pass the
    metadata filter, then pulls the ``audio`` column only for the row groups holding the best
    candidates, in rank order, until one clip passes check_clip. ``open_file(path)`` returns a
    binary file object (HfFileSystem.open on Modal; builtin open in tests).

    Returns {"ok": True, "row": ..., "samples": trimmed, "sr": ..., "stats": ..., ...} or
    {"ok": False, "reason": ..., "rejections": {...}}.
    """
    import pyarrow.parquet as pq
    import soundfile as sf

    candidates = []                      # (rank_key, shard_idx, row_group, row_in_group, row)
    styles, genders, rejections = {}, {}, {}
    handles = {}

    def _open(si):
        if si not in handles:
            handles[si] = pq.ParquetFile(open_file(shard_paths[si]))
        return handles[si]

    def _collect(si):
        pf = _open(si)
        cols = [c for c in META_COLUMNS if c in pf.schema_arrow.names]
        for rg in range(pf.num_row_groups):
            for ri, row in enumerate(pf.read_row_group(rg, columns=cols).to_pylist()):
                styles[row.get("style") or "?"] = styles.get(row.get("style") or "?", 0) + 1
                genders[row.get("gender") or "?"] = genders.get(row.get("gender") or "?", 0) + 1
                ok, why = metadata_ok(row, lang_code)
                if not ok:
                    key = why.split("=")[0]
                    rejections[key] = rejections.get(key, 0) + 1
                    continue
                candidates.append((rank_key(row, gender), si, rg, ri, row))

    # Rasa shards are one speaker each (Tamil test-00000 is 457/457 Female), so probe each
    # shard's gender from the first row group's `gender` column alone — a few KB — and read the
    # full metadata only of a shard with the wanted voice. A shard of ~450 rows is plenty.
    probed_other = []
    for si in range(len(shard_paths)):
        pf = _open(si)
        if "gender" in pf.schema_arrow.names and pf.num_row_groups:
            g0 = pf.read_row_group(0, columns=["gender"]).column("gender").to_pylist()
            if not any((g or "").strip().lower()[:1] == gender[:1] for g in g0):
                probed_other.append(si)
                continue
        _collect(si)
        if any(c[0][0] == 0 for c in candidates):
            break
    if not candidates and probed_other:
        # No shard has the wanted gender (single-speaker language): take the other voice
        # rather than none — rank_key already orders it after any wanted-gender clip.
        log(f"[voice_refs] {lang_code}: no '{gender}' speaker found in {len(shard_paths)} "
            f"shard(s); using the other voice")
        _collect(probed_other[0])

    candidates.sort(key=lambda c: c[0])
    log(f"[voice_refs] {lang_code}: {len(candidates)} metadata candidates "
        f"(rejected: {rejections}); styles={dict(sorted(styles.items()))}")
    if not candidates:
        return {"ok": False, "reason": "no_metadata_candidates", "rejections": rejections,
                "style_histogram": styles, "gender_histogram": genders}

    audio_cache, tried = {}, 0
    for key, si, rg, ri, row in candidates:
        if (si, rg) not in audio_cache:
            if len(audio_cache) >= max_audio_row_groups:
                continue
            col = handles[si].read_row_group(rg, columns=["audio"]).column("audio").to_pylist()
            audio_cache[(si, rg)] = col
        audio = audio_cache[(si, rg)][ri] or {}
        blob = audio.get("bytes") if isinstance(audio, dict) else None
        tried += 1
        if not blob:
            rejections["no_audio_bytes"] = rejections.get("no_audio_bytes", 0) + 1
            continue
        try:
            samples, sr = sf.read(io.BytesIO(blob), dtype="float32", always_2d=False)
        except Exception as e:
            rejections["decode_error"] = rejections.get("decode_error", 0) + 1
            log(f"[voice_refs] {lang_code}: decode failed for {row.get('filename')}: {e}")
            continue
        ok, reasons, stats, trimmed = check_clip(samples, sr, row.get("text", ""), lang_code,
                                                 row.get("duration"))
        if ok:
            return {"ok": True, "row": row, "samples": trimmed, "sr": sr, "stats": stats,
                    "shard": shard_paths[si], "row_group": rg, "candidates_tried": tried,
                    "rejections": rejections, "style_histogram": styles,
                    "gender_histogram": genders}
        for r in reasons:
            k = r.split("=")[0]
            rejections[k] = rejections.get(k, 0) + 1
    return {"ok": False, "reason": "no_clip_passed_audio_checks", "rejections": rejections,
            "candidates_tried": tried, "style_histogram": styles, "gender_histogram": genders}


def curate(out_dir: str, languages: Optional[list] = None, token: Optional[str] = None,
           gender: str = DEFAULT_GENDER, log: Callable = print,
           selector: Optional[Callable] = None) -> dict:
    """Choose, check and store one native reference clip per language; write the manifest.

    Hindi is skipped by default — it keeps the pinned, already-validated clip. A language that
    fails is recorded in the manifest with its reason (and keeps the Hindi fallback), so the
    result is visible rather than silently absent. Existing entries for languages not being
    re-curated are kept.
    """
    import soundfile as sf

    languages = languages or [c for c in CODE_TO_LANGUAGE if c != "hi"]
    os.makedirs(out_dir, exist_ok=True)
    manifest = load_manifest(out_dir) or {}
    manifest.update({"schema": SCHEMA_VERSION, "source_repo": RASA_REPO,
                     "license": RASA_LICENSE, "attribution": RASA_ATTRIBUTION,
                     "criteria": criteria(), "gender_preference": gender})
    entries = manifest.setdefault("languages", {})

    for code in languages:
        language = CODE_TO_LANGUAGE[code]
        t0 = time.time()
        try:
            if selector is not None:
                res = selector(code)
            else:
                fs, shards = _rasa_shards(language, token)
                if not shards:
                    raise RuntimeError(f"no parquet shards under {RASA_REPO}/{language} "
                                       "(terms not accepted for this HF_TOKEN?)")
                # A small read block: fsspec's default (MBs) turns each ~10 KB column-chunk
                # read into a multi-MB download — measured 288 s for one shard's metadata.
                res = select_from_parquet(lambda p: fs.open(p, "rb", block_size=READ_BLOCK),
                                          shards, code, gender=gender, log=log)
        except Exception as e:
            res = {"ok": False, "reason": f"error: {type(e).__name__}: {e}"}
        if not res.get("ok"):
            entries[code] = {"status": "failed", "language": language,
                             "reason": res.get("reason"), "rejections": res.get("rejections"),
                             "curated_at": int(time.time())}
            log(f"[voice_refs] {code} ({language}): FAILED — {res.get('reason')} "
                f"{res.get('rejections') or ''}")
            continue
        row = res["row"]
        wav_name = f"{code}.wav"
        wav_path = os.path.join(out_dir, wav_name)
        tmp = wav_path + ".tmp"
        sf.write(tmp, res["samples"], res["sr"], subtype="PCM_16", format="WAV")
        os.replace(tmp, wav_path)
        entries[code] = {
            "status": "ok", "language": language, "file": wav_name,
            "sha256": _sha256(wav_path), "text": (row.get("text") or "").strip(),
            "gender": row.get("gender"), "style": row.get("style"),
            "source_filename": row.get("filename"), "source_shard": res.get("shard"),
            "source_row_group": res.get("row_group"), "stats": res["stats"],
            "candidates_tried": res.get("candidates_tried"),
            "style_histogram": res.get("style_histogram"),
            "curated_at": int(time.time()), "seconds_spent": round(time.time() - t0, 1),
        }
        log(f"[voice_refs] {code} ({language}): OK {row.get('filename')} "
            f"{res['stats']['trimmed_seconds']}s style={row.get('style')} "
            f"gender={row.get('gender')}")

    write_manifest(out_dir, manifest)
    return manifest


# ── Manifest ──────────────────────────────────────────────────────────────────────────────
def load_manifest(directory: Optional[str]) -> Optional[dict]:
    if not directory:
        return None
    path = os.path.join(directory, MANIFEST_NAME)
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def write_manifest(directory: str, manifest: dict) -> None:
    path = os.path.join(directory, MANIFEST_NAME)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def native_ref_id(lang_code: str, directory: Optional[str] = None) -> str:
    """Identity of the native reference a Basic-mode run WILL use ('' if none).

    Makes the exact decision resolve_basic_reference makes (same verification: sha256 +
    script), so the orchestrator's resume check and the supervisor's forced-silence entries
    carry the same signature the worker computes. Hashing one ~6 s clip is negligible.
    """
    if lang_code == "hi":
        return ""
    path, _text, entry, _problem = load_native_reference(lang_code, directory)
    return f"native:{entry['sha256'][:12]}" if path else ""


def load_native_reference(lang_code: str, directory: Optional[str] = None) -> tuple:
    """(path, text, entry, problem). ``problem`` is '' when the clip is usable.

    Verifies what matters at use time: the file is the one that was checked (sha256) and the
    transcript is still in the right script.
    """
    d = ref_dir(directory)
    if not d:
        return None, None, None, f"{ENV_DIR} not set"
    m = load_manifest(d)
    if m is None:
        return None, None, None, f"no {MANIFEST_NAME} in {d}"
    entry = (m.get("languages") or {}).get(lang_code)
    if not entry:
        return None, None, None, f"{lang_code} not curated"
    if entry.get("status") != "ok":
        return None, None, entry, f"{lang_code} curation {entry.get('status')}: {entry.get('reason')}"
    path = os.path.join(d, entry.get("file", ""))
    if not os.path.exists(path):
        return None, None, entry, f"missing file {path}"
    if _sha256(path) != entry.get("sha256"):
        return None, None, entry, f"sha256 mismatch for {path} (file changed since curation)"
    text = (entry.get("text") or "").strip()
    frac = script_fraction(text, lang_code)
    if frac < MIN_SCRIPT_FRACTION:
        return None, None, entry, f"transcript script_fraction={frac:.2f}"
    return path, text, entry, ""


# ── Resolution ────────────────────────────────────────────────────────────────────────────
def _download_pinned_hindi() -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=DEFAULT_HINDI_REF_REPO, filename=DEFAULT_HINDI_REF_FILE,
                           repo_type="dataset")


def resolve_basic_reference(lang_code: str, log_fn: Optional[Callable] = None,
                            directory: Optional[str] = None,
                            download_pinned: Callable = _download_pinned_hindi,
                            text_only_refs: Optional[dict] = None) -> dict:
    """Pick the Basic-mode reference for ``lang_code``. Never raises.

    Returns {"audio", "text", "source", "ref_id", "detail"}. ``ref_id`` goes into the segment
    signature: '' for the pinned/fallback Hindi clip (the voice every earlier run used, so old
    resume manifests stay valid) and 'native:<sha12>' for a native clip (so a voice change
    re-synthesizes instead of resuming stale audio).
    """
    def say(m):
        if log_fn is not None:
            try:
                log_fn(m)
            except Exception:
                pass

    language = CODE_TO_LANGUAGE.get(lang_code, lang_code)
    native_problem = "Hindi uses the pinned clip"
    if lang_code != "hi":
        path, text, entry, native_problem = load_native_reference(lang_code, directory)
        if path:
            say(f"Mode: Basic — native {language} reference voice "
                f"({entry.get('source_filename')}, {entry.get('gender')}, {entry.get('style')}, "
                f"{(entry.get('stats') or {}).get('trimmed_seconds')}s; {RASA_REPO}, {RASA_LICENSE})")
            return {"audio": path, "text": text, "source": "native",
                    "ref_id": f"native:{entry['sha256'][:12]}", "detail": entry.get("source_filename")}

    try:
        audio = download_pinned()
    except Exception as e:
        say(f"Warning: could not download the default reference voice: {e}")
        say("Falling back to text-only reference (may affect quality/stability).")
        text = (text_only_refs or {}).get(lang_code, "")
        return {"audio": None, "text": text, "source": "text-only", "ref_id": "",
                "detail": str(e)}

    if lang_code == "hi":
        say(f"Mode: Basic — pinned Hindi reference voice ({DEFAULT_HINDI_REF_FILE})")
        return {"audio": audio, "text": DEFAULT_HINDI_REF_TEXT, "source": "pinned",
                "ref_id": "", "detail": DEFAULT_HINDI_REF_FILE}

    say(f"WARNING: no native {language} reference voice ({native_problem}). Using the HINDI "
        f"reference clip cross-lingually — expect a Hindi accent and possible onset babble. "
        f"Fix: run `modal run deploy/modal_app.py::curate_voice_refs`.")
    return {"audio": audio, "text": DEFAULT_HINDI_REF_TEXT, "source": "fallback",
            "ref_id": "", "detail": native_problem}


def coverage(directory: Optional[str] = None) -> dict:
    """{lang_code: "native" | "pinned" | "fallback"} — the voice each language gets; for the UI
    and prewarm. (Reports what resolution would pick, not whether HF is reachable.)"""
    out = {}
    for code in CODE_TO_LANGUAGE:
        if code == "hi":
            out[code] = "pinned"
        else:
            out[code] = "native" if native_ref_id(code, directory) else "fallback"
    return out
