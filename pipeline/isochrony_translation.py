"""
isochrony_translation.py — Stage 3 of the Indic Dubbing Pipeline

Isochrony-Aware Translation: translates English segments into Indic languages
while constraining the output to fit the source audio duration budget AND
preserving the meaning of the source.

Core insight: "The real fix is upstream at translation."
Most dubbing pipelines generate a semantically correct translation, then try
to fix timing at the TTS stage. We fix it HERE — before audio is ever generated.

v2.5 — Semantic-gated, iterative isochrony
------------------------------------------
Timing alone is not enough: a translation can hit the phoneme budget perfectly
and still say the wrong thing. Selection is now a TWO-STAGE decision, exactly as
a human dubbing director would make it:

  Stage A (semantic gate): score every candidate's cross-lingual similarity to
    the English source with IndicSBERT (pipeline.semantic_similarity). Keep only
    the candidates that actually MEAN the same thing (similarity >= threshold).
  Stage B (isochrony pick): among the survivors, choose the one whose REAL
    phoneme count (pipeline.phoneme_counter, espeak-ng G2P) is closest to the
    duration-grounded budget.

And it is ITERATIVE. Instead of one generate-then-score pass, the model is asked
to generate Chain-of-Thought translations, we MEASURE them (semantics + real
phonemes), and we feed those measurements back into the next prompt — repeating
until every segment clears BOTH gates (similarity >= threshold AND phoneme gap
<= tolerance) or the combined objective reaches a global minimum (further
iterations stop improving it). The best candidate ever seen for each segment is
retained across iterations, so an extra round can never make a segment worse.

Graceful degradation: if IndicSBERT is unavailable (offline / gated download /
the huggingface_hub segfault seen on some machines) the semantic gate is skipped
VISIBLY — semantic_similarity.score_many() returns None — and selection falls
back to phoneme-fit only. If espeak-ng is unavailable the phoneme counter
degrades to a labelled heuristic. Neither failure aborts the run.

Language support: All 11 Indic languages supported by IndicF5.
"""

import json
import os
import time
import ssl
import re
import random
import threading
import concurrent.futures
from typing import Optional
import builtins

_orig_print = builtins.print

def print(*args, **kwargs):
    try:
        _orig_print(*args, **kwargs)
    except UnicodeEncodeError:
        new_args = [
            arg.encode('ascii', errors='replace').decode('ascii') if isinstance(arg, str) else arg
            for arg in args
        ]
        _orig_print(*new_args, **kwargs)

from google import genai
from google.genai import types

from pydantic import BaseModel, Field
from typing import List, Callable

from pipeline.phoneme_counter import (
    compute_target_budget,
    isochrony_score,
    count_indic_phonemes,
    phoneme_diff,
    active_ruler,
)
from pipeline import semantic_similarity
from pipeline import translation_cache

# Minimum isochrony score to accept without further refinement (legacy knob, kept
# for backward-compatible callers; the iterative loop below uses the richer
# phoneme-tolerance / semantic-threshold pair).
MINIMUM_ACCEPTABLE_SCORE = 0.75
# Number of candidate translations to generate per segment (MBR-style)
N_CANDIDATES = 3

# --- v2.5 iterative-loop knobs -------------------------------------------------
# Cross-lingual semantic similarity (IndicSBERT cosine, [0,1]) a candidate must
# reach to clear the meaning gate. 0.70 is a deliberately permissive gate: it
# rejects candidates that drift in meaning while still admitting the natural
# rephrasings that isochrony demands.
SEMANTIC_THRESHOLD = 0.70
# Relative phoneme gap |target - ideal| / ideal at or below which a candidate is
# "on budget". Matches the +/-15% acceptance band used by compute_target_budget.
PHONEME_TOLERANCE = 0.15
# Max Chain-of-Thought refinement rounds AFTER the initial batch (so at most
# MAX_ITERATIONS + 1 generations touch any segment).
MAX_ITERATIONS = 3

# Hard ceiling on spoken phoneme density (phonemes/second). Natural Indic narration
# ranges between 9.5 and 10.5 phonemes/sec. Beyond 11.5 phonemes/sec, IndicF5 begins
# dropping initial syllables or truncating words at segment boundaries.
MAX_PHONEME_DENSITY = 11.5

# Combined objective used for global-minima tracking when a candidate clears
# neither/one gate. Lower is better:  loss = w_sem*(1 - sim) + w_phon*rel_diff.
# Semantics is weighted higher — a mistranslation that fits the timing is worse
# than a faithful translation that is slightly off timing (TTS can absorb a
# small timing gap; it cannot fix wrong words).
SEMANTIC_WEIGHT = 0.6
PHONEME_WEIGHT = 0.4

# --- v2.6 whole-transcript audit (meaning + isochrony safety net) --------------
# The per-segment gate above scores each candidate against ITS OWN source span in
# isolation with an embedder. Two failure modes slip through it structurally:
#   1. Meaning inversion/negation — IndicSBERT cosine cannot tell "has power" from
#      "does NOT have power"; both score ~0.9 to the same source, so a flipped
#      line clears the gate.
#   2. Fabricated completion — Whisper cuts sentences mid-clause, each fragment is
#      translated alone, and under timing pressure the model invents a plausible
#      but wrong ending for a sentence that actually continues in the NEXT segment.
# Neither is visible one-segment-at-a-time. So AFTER selection we run a single
# whole-transcript LLM audit that re-reads the full English and full translation
# TOGETHER (an LLM reasons about meaning; an embedder cannot) and flags per-segment
# drift/inversion/omission/fabrication. Flagged segments are auto-healed —
# re-translated with the reviewer's specific reason + neighbour source context,
# inside the SAME phoneme budget so the meaning fix cannot break the timing — for
# up to AUDIT_MAX_FIX_ROUNDS rounds; anything still failing is kept as the best
# candidate and FLAGGED in the log and output (degrade, don't crash).
# Env: DUBBING_TRANSLATION_AUDIT=0 disables the whole pass.
AUDIT_MAX_FIX_ROUNDS = 2
# Windowing for long transcripts (avoids payload limits and attention degradation):
# Each window judges AUDIT_WINDOW_SIZE segments while showing AUDIT_WINDOW_OVERLAP
# adjacent segments before & after as read-only context to preserve discourse continuity.
AUDIT_WINDOW_SIZE = int(os.environ.get("DUBBING_AUDIT_WINDOW_SIZE", "15"))
AUDIT_WINDOW_OVERLAP = int(os.environ.get("DUBBING_AUDIT_WINDOW_OVERLAP", "3"))


# --- v2.5.1 rate-limit / hybrid-model config ----------------------------------
# gemini-3.1-flash-lite has a strict free-tier limit (both per-minute and
# per-day). To stay under it we split the work by PHASE:
#
#   • BULK phase (iteration 0) — the many first-draft candidates for every
#     segment — runs on lenient-limit Gemma models by default. This is the bulk
#     of all API calls.
#   • REFINE phase — the few Chain-of-Thought rounds on only the hard segments —
#     runs on gemini-3.1-flash-lite (quality where it matters, few calls).
#
# Both chains keep the full Gemini fallback ladder, so if a Gemma id is not
# available on a given key (or is renamed) the call self-heals to Gemini with a
# visible log line — it never hard-fails on a model-name guess.
#
# Every id is env-overridable so no code change is needed to retune:
#   DUBBING_GEMINI_BULK_MODEL     head of the bulk (Gemma) chain
#   DUBBING_GEMINI_REFINE_MODEL   head of the refine (Gemini) chain
#   DUBBING_GEMINI_MODEL          legacy alias for the refine head
#   DUBBING_GEMINI_RPM            client-side requests/minute pace (per model)
#   DUBBING_GEMINI_RPD            optional per-model requests/day hard cap
_GEMINI_FALLBACK = ["gemini-3.1-flash-lite", "gemini-3.5-flash-lite"]
_GEMMA_BULK_DEFAULT = ["gemma-4-31b-it", "gemma-4-26b-a4b-it"]


def _bulk_models() -> List[str]:
    """Model chain for the iteration-0 bulk batch: lenient Gemma first, then the
    Gemini ladder as a safety net."""
    head = os.environ.get("DUBBING_GEMINI_BULK_MODEL")
    chain = [head] if head else list(_GEMMA_BULK_DEFAULT)
    for m in _GEMINI_FALLBACK:
        if m not in chain:
            chain.append(m)
    return chain


def _refine_models() -> List[str]:
    """Model chain for refinement rounds: gemini-3.1-flash-lite first (quality),
    then the rest of the Gemini ladder."""
    head = os.environ.get("DUBBING_GEMINI_REFINE_MODEL") or os.environ.get("DUBBING_GEMINI_MODEL")
    if head:
        return [head] + [m for m in _GEMINI_FALLBACK if m != head]
    return list(_GEMINI_FALLBACK)


def _is_gemma(model: str) -> bool:
    """Gemma models on the Gemini API do NOT support structured output
    (`response_schema`); they must be driven text-mode + robust JSON parsing."""
    return "gemma" in (model or "").lower()


# ── Client-side throttle (per-model min interval derived from RPM) ───────────
_RATE_LOCK = threading.RLock()
_LAST_CALL: dict = {}  # model_name -> monotonic timestamp of last request


def _rpm_for(model: str) -> float:
    env = os.environ.get("DUBBING_GEMINI_RPM")
    if env:
        try:
            v = float(env)
            if v > 0:
                return v
        except ValueError:
            pass
    # Gemma free tiers are more generous per-minute than flash-lite.
    return 30.0 if _is_gemma(model) else 15.0


def _throttle(model: str) -> None:
    """Enforce a minimum spacing between requests to the same model so we never
    burst past the per-minute limit. Serialized under a lock (the app processes
    one dub at a time), which keeps pacing strict even if called concurrently."""
    interval = 60.0 / _rpm_for(model)
    with _RATE_LOCK:
        now = time.monotonic()
        wait = interval - (now - _LAST_CALL.get(model, 0.0))
        if wait > 0:
            time.sleep(wait)
        _LAST_CALL[model] = time.monotonic()


def _gemini_concurrency() -> int:
    """How many batched Gemini calls may be IN FLIGHT at once. `_throttle` still spaces the
    START of each request to a given model by 60/RPM, so concurrency never bursts past the
    per-minute cap — it only overlaps the response WAITS of independent batches, which is where
    a long clip's translation wall clock actually goes. 1 restores the old strictly-sequential
    behaviour; env-tunable via DUBBING_GEMINI_CONCURRENCY (default 4, clamped to [1, 16])."""
    try:
        v = int(os.environ.get("DUBBING_GEMINI_CONCURRENCY", "4"))
    except (TypeError, ValueError):
        return 4
    return max(1, min(v, 16))


def _run_batches_concurrent(chunks, run_one, concurrency, log_fn=None):
    """Fold ``run_one(chunk) -> (dict, served_list)`` over independent ``chunks`` with at most
    ``concurrency`` calls in flight, returning ``(merged_dict, served_list)``.

    Correctness is order-independent: chunks are DISJOINT segment slices, so each partial dict
    carries a distinct set of segment ids and ``merged.update`` can never overwrite one chunk's
    result with another's — the merged dict is identical no matter which chunk finishes first.
    ``served`` names are concatenated in chunk order so the "served by" line is deterministic.
    With ``concurrency <= 1`` or a single chunk this is byte-for-byte the old sequential loop.
    """
    chunks = list(chunks)
    if concurrency <= 1 or len(chunks) <= 1:
        merged, served = {}, []
        for ch in chunks:
            part, srv = run_one(ch)
            merged.update(part)
            served.extend(srv)
        return merged, served

    if log_fn:
        try:
            log_fn(f"  [IsochronyTranslation] Dispatching {len(chunks)} batches, "
                   f"up to {concurrency} in flight...")
        except Exception:
            pass
    results = [None] * len(chunks)     # index by submission order for deterministic aggregation
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
        fut_to_idx = {ex.submit(run_one, ch): i for i, ch in enumerate(chunks)}
        for fut in concurrent.futures.as_completed(fut_to_idx):
            results[fut_to_idx[fut]] = fut.result()   # run_one guards itself; never expected to raise
    merged, served = {}, []
    for part, srv in results:
        merged.update(part)
        served.extend(srv)
    return merged, served


_RETRY_DELAY_RE = re.compile(
    r"retry[_ ]?delay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)\s*s?", re.IGNORECASE
)


def _parse_retry_delay(err: str) -> Optional[float]:
    """Honour the server-suggested `retryDelay` in a 429 body when present
    (capped so a pathological value can't stall the run)."""
    m = _RETRY_DELAY_RE.search(err or "")
    if not m:
        return None
    try:
        return min(float(m.group(1)), 90.0)
    except ValueError:
        return None


# ── Robust JSON extraction (text-mode path for Gemma / fenced output) ───────

def _extract_json(text: str):
    """Parse JSON that may be wrapped in ```json fences or surrounded by prose.
    Gemma has no structured-output mode, so its replies arrive as text; this also
    hardens the Gemini path against the occasional stray fence."""
    if not text:
        return None
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t).strip()
    try:
        return json.loads(t)
    except Exception:
        pass
    # Fall back to bracket-matching the first balanced array or object.
    for open_ch, close_ch in (("[", "]"), ("{", "}")):
        start = t.find(open_ch)
        if start == -1:
            continue
        depth, in_str, esc = 0, False, False
        for idx in range(start, len(t)):
            ch = t[idx]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(t[start:idx + 1])
                    except Exception:
                        break
    return None


def _parse_batch(raw: str) -> dict:
    """Normalize a batch reply into {segment_id: [candidate, ...]}. Accepts both
    the schema shape ({"translations": [...]}) and the bare array a schema-less
    Gemma reply follows from the prompt."""
    data = _extract_json(raw)
    if isinstance(data, dict):
        items = data.get("translations") or data.get("segments") or data.get("results") or []
    elif isinstance(data, list):
        items = data
    else:
        items = []
    out = {}
    for item in items:
        if not isinstance(item, dict) or "segment_id" not in item:
            continue
        try:
            sid = int(item["segment_id"])
        except (TypeError, ValueError):
            continue
        cands = item.get("candidates")
        if isinstance(cands, str):
            cands = [cands]
        if isinstance(cands, list) and cands:
            out[sid] = [str(c) for c in cands if str(c).strip()]
    return out


def _parse_candidates(raw: str) -> list:
    """Normalize a single-segment reply into a candidate list. Accepts the schema
    shape ({"candidates": [...]}) and a bare array/string."""
    data = _extract_json(raw)
    if isinstance(data, dict):
        c = data.get("candidates")
    elif isinstance(data, list):
        c = data
    else:
        c = None
    if isinstance(c, str):
        c = [c]
    if isinstance(c, list):
        return [str(x) for x in c if str(x).strip()]
    return []


# Language name → IndicF5 language code mapping
LANGUAGE_CODES = {
    "hindi":     "hi",
    "bengali":   "bn",
    "marathi":   "mr",
    "gujarati":  "gu",
    "punjabi":   "pa",
    "tamil":     "ta",
    "telugu":    "te",
    "kannada":   "kn",
    "malayalam": "ml",
    "odia":      "or",
    "assamese":  "as",
}

# Maps UI-friendly display names to internal keys used above
DISPLAY_TO_INTERNAL = {
    "Hindi":     "hindi",
    "Bengali":   "bengali",
    "Marathi":   "marathi",
    "Gujarati":  "gujarati",
    "Punjabi":   "punjabi",
    "Tamil":     "tamil",
    "Telugu":    "telugu",
    "Kannada":   "kannada",
    "Malayalam": "malayalam",
    "Odia":      "odia",
    "Assamese":  "assamese",
}


# ── Pydantic models for structured output ──────────────────────────────────

class SegmentTranslation(BaseModel):
    segment_id: int = Field(description="The unique integer ID of the segment")
    candidates: List[str] = Field(description="A list containing exactly N translation candidates")

class BatchTranslationResponse(BaseModel):
    translations: List[SegmentTranslation] = Field(description="List of translated segments")

class SegmentCandidatesResponse(BaseModel):
    candidates: List[str] = Field(description="List of N translation candidates for this single segment")

class SegmentAudit(BaseModel):
    segment_id: int = Field(description="The integer ID of the audited segment")
    verdict: str = Field(
        description="One of: ok, drift, inversion, omission, fabrication, addition, incomplete"
    )
    needs_fix: bool = Field(
        description="True if the translation must be corrected for meaning fidelity"
    )
    reason: str = Field(
        description="One concise sentence naming the specific meaning problem, or 'faithful' if ok"
    )

class TranscriptAuditResponse(BaseModel):
    audits: List[SegmentAudit] = Field(description="One audit verdict per segment, in order")


# ── Client helpers ─────────────────────────────────────────────────────────

def _build_client(api_key: str) -> genai.Client:
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            client_args={"verify": ssl_context},
            async_client_args={"verify": ssl_context},
        ),
    )


# ── Client-side hard timeout for a single generate_content call ──────────────
# The google-genai SDK has shipped versions that pass timeout=None straight to httpx
# (googleapis/python-genai #911; pydantic-ai #4031), so HttpOptions(timeout=...) cannot be
# relied on. Without an enforced ceiling a slow/overloaded free-tier endpoint — e.g. the
# newly-released FREE gemma-4-31b-it — wedges the whole call with no error and Step 4 hangs
# forever (the exact symptom seen). We enforce the ceiling ourselves on a DAEMON thread so
# it holds regardless of SDK version and never blocks interpreter shutdown; on expiry the
# caller's retry/fallback logic proceeds immediately.
# Per-call hard ceilings. Gemma DENSE models are markedly slower to respond than the
# flash-lite Gemini models (150s+ observed on the free tier), so they get a generous
# ceiling — a legit slow Gemma reply must NOT be falsely killed. A flash-lite call that
# stays silent this long is instead a real wedge, so it keeps the tight ceiling.
# DUBBING_GEMINI_CALL_TIMEOUT overrides BOTH (explicit intent applies to every model).
_CALL_TIMEOUT_DEFAULT = 90.0    # Gemini flash-lite: fast; long silence == real wedge
_CALL_TIMEOUT_GEMMA = 240.0     # Gemma dense: legitimately slow to first response


def _call_timeout_seconds(model: str = "") -> float:
    raw = os.environ.get("DUBBING_GEMINI_CALL_TIMEOUT", "").strip()
    if raw:
        try:
            v = float(raw)
            if v > 0:
                return v
        except ValueError:
            pass
    return _CALL_TIMEOUT_GEMMA if _is_gemma(model) else _CALL_TIMEOUT_DEFAULT


# How often, while a call is in flight, to emit a "still waiting" liveness tick so a slow
# (but working) endpoint is never mistaken for the old silent hang. Env-tunable.
_CALL_HEARTBEAT_DEFAULT = 15.0


def _call_heartbeat_seconds() -> float:
    raw = os.environ.get("DUBBING_GEMINI_HEARTBEAT", "").strip()
    if raw:
        try:
            v = float(raw)
            if v > 0:
                return v
        except ValueError:
            pass
    return _CALL_HEARTBEAT_DEFAULT


def _run_with_timeout(fn, timeout_s: float, heartbeat_fn=None, heartbeat_interval: float = 15.0):
    """Run fn() on a daemon thread; raise TimeoutError if it outlasts timeout_s.

    While waiting, call heartbeat_fn(elapsed_s, timeout_s) every heartbeat_interval seconds
    (if given) so a slow-but-alive call emits visible liveness ticks instead of going dark —
    the whole point being that a working 40s call must not LOOK like the old infinite hang.

    The abandoned thread keeps running until the underlying request returns (a thread
    cannot be force-killed), but being a daemon it never blocks process exit, and the
    caller resumes its retry/fallback chain at once instead of blocking indefinitely.
    """
    box: dict = {}

    def _target():
        try:
            box["value"] = fn()
        except BaseException as e:  # propagate to the caller thread
            box["error"] = e

    t = threading.Thread(target=_target, name="genai-call", daemon=True)
    t.start()
    start = time.monotonic()
    while True:
        remaining = timeout_s - (time.monotonic() - start)
        if remaining <= 0:
            break
        # Wake at the next heartbeat tick (or at the deadline, whichever is first).
        t.join(min(heartbeat_interval, remaining) if heartbeat_fn else remaining)
        if not t.is_alive():
            break
        if heartbeat_fn:
            try:
                heartbeat_fn(time.monotonic() - start, timeout_s)
            except Exception:
                pass
    if t.is_alive():
        raise TimeoutError(
            f"generate_content exceeded {timeout_s:.0f}s client-side timeout "
            f"(SDK/network wedge; abandoning call)"
        )
    if "error" in box:
        raise box["error"]
    return box.get("value")


def _call_gemini(
    client,
    prompt: str,
    temperature: float = 0.4,
    response_schema=None,
    response_mime_type=None,
    log_fn: Optional[Callable[[str], None]] = None,
    models: Optional[List[str]] = None,
    served: Optional[List[str]] = None,
    validate_fn: Optional[Callable[[str], object]] = None,
    max_parse_retries: int = 2,
) -> str:
    """Call Gemini with logging, throttling, rate-limit-aware retries, and a
    model fallback chain.

    If `served` is provided, the name of the model that actually returned the
    response is appended to it — so the caller can report which model (Gemma vs
    Gemini) really served a phase, rather than which chain it *intended* to use.

    If `validate_fn` is provided, each raw reply is run through it before we
    accept it. `validate_fn(resp_text)` returns a truthy value when the reply is
    usable (e.g. `_parse_batch` returning a non-empty dict) and a falsy value (or
    raises) when it is not. On a parse failure we RE-ASK THE SAME MODEL up to
    `max_parse_retries` times with an explicit "return ONLY the JSON array"
    instruction at temperature 0, and only after those are exhausted do we fall
    through to the next model in the chain. This keeps a Gemma reformat glitch on
    the lenient Gemma quota instead of escalating it to the scarce Gemini one.

    `models` is the chain to walk (default: the refine/Gemini ladder). Behaviour
    that keeps us under the free-tier limits:

      • THROTTLE — before each call we space requests to the same model by
        60/RPM seconds (client-side), so we never burst past the per-minute cap.
      • 429 BACK-OFF ON THE SAME MODEL — a rate-limit error backs off (honouring
        the server's `retryDelay` when given, else exponential + jitter) and
        retries the SAME model, instead of burning down the fallback chain. Only
        after several rate retries do we advance to the next model. This fixes
        the old bug where one 429 skipped straight to a weaker model.
      • GEMMA HAS NO STRUCTURED OUTPUT — for a Gemma model we drop
        `response_schema`/`response_mime_type` (unsupported) and rely on
        text-mode JSON parsing at the call site.
      • MODEL-NOT-FOUND SELF-HEAL — a 404 / unsupported id advances to the next
        model immediately (so a Gemma id that doesn't exist on this key falls
        through to Gemini automatically).
      • DAILY CAP — each real request is recorded; if a model is over the
        optional DUBBING_GEMINI_RPD cap we skip it and advance; only when every
        model is capped do we raise.
    """
    chain = list(models) if models else _refine_models()
    rpd = translation_cache.rpd_limit()

    def _emit(m: str):
        if log_fn:
            log_fn(m)
        print(m)

    idx = 0
    rate_retries = 0
    transient_retries = 0      # 500/503 back-off budget for the CURRENT model
    parse_retries = 0          # reask budget for the CURRENT model; reset on advance
    total_tries = 0
    # Hard ceiling on real API calls. Sized to allow a couple of parse-reasks and
    # a few transient-server-error retries on the first model(s) without starving
    # the chain walk that follows.
    max_total_tries = 20
    max_rate_retries = 4
    max_transient_retries = 3
    _reformat_suffix = (
        "\n\nIMPORTANT: Return ONLY the JSON array requested above — no prose, no "
        "markdown fences, no explanation. Output must start with '[' and end with "
        "']' and be valid JSON."
    )

    while total_tries < max_total_tries and idx < len(chain):
        model_name = chain[idx]

        # Daily-cap guard: skip a model that is already over its RPD for today.
        if rpd is not None and translation_cache.count_today(model_name) >= rpd:
            _emit(f"  [Gemini API] '{model_name}' hit daily cap ({rpd}); advancing model.")
            idx += 1
            rate_retries = 0
            transient_retries = 0
            parse_retries = 0
            continue

        total_tries += 1
        # On a parse-reask, nudge the SAME model toward clean JSON at temp 0.
        reasking = parse_retries > 0
        call_prompt = prompt + _reformat_suffix if reasking else prompt
        call_temp = 0.0 if reasking else temperature
        _throttle(model_name)
        _emit(f"  [Gemini API] Calling '{model_name}' "
              f"(try {total_tries}/{max_total_tries}, prompt len: {len(call_prompt)}"
              f"{', JSON-reask' if reasking else ''})...")
        start_time = time.time()

        try:
            config_args = {"temperature": call_temp}
            # Gemma on the Gemini API rejects response_schema / json mime — send
            # a plain text-mode request and let the caller parse the JSON.
            if not _is_gemma(model_name):
                if response_schema is not None:
                    config_args["response_schema"] = response_schema
                if response_mime_type is not None:
                    config_args["response_mime_type"] = response_mime_type

            translation_cache.record_request(model_name)

            def _waiting(el, budget):
                _emit(f"  [Gemini API]   ...still waiting on '{model_name}' "
                      f"({el:.0f}s elapsed / {budget:.0f}s timeout) — alive, awaiting response.")

            response = _run_with_timeout(
                lambda: client.models.generate_content(
                    model=model_name,
                    contents=call_prompt,
                    config=types.GenerateContentConfig(**config_args),
                ),
                _call_timeout_seconds(model_name),
                heartbeat_fn=_waiting,
                heartbeat_interval=_call_heartbeat_seconds(),
            )
            elapsed = time.time() - start_time
            resp_text = response.text.strip() if response.text else ""
            _emit(f"  [Gemini API] ✓ '{model_name}' in {elapsed:.2f}s. "
                  f"Response len: {len(resp_text)} chars.")

            # Content validation (e.g. JSON parse). A reply that comes back but
            # doesn't parse is NOT a success — reask the same model, then advance.
            if validate_fn is not None:
                try:
                    valid = bool(validate_fn(resp_text))
                except Exception:
                    valid = False
                if not valid:
                    if parse_retries < max_parse_retries:
                        parse_retries += 1
                        _emit(f"  [Gemini API] '{model_name}' reply did not parse; "
                              f"re-asking same model for clean JSON "
                              f"(parse retry {parse_retries}/{max_parse_retries}).")
                        continue
                    _emit(f"  [Gemini API] '{model_name}' still unparseable after "
                          f"{max_parse_retries} reask(s); advancing to next model.")
                    idx += 1
                    parse_retries = 0
                    rate_retries = 0
                    transient_retries = 0
                    continue

            if served is not None:
                served.append(model_name)
            return resp_text
        except Exception as e:
            err = str(e)
            elapsed = time.time() - start_time
            _emit(f"  [Gemini API] ⚠️ '{model_name}' failed in {elapsed:.2f}s: {err[:140]}")

            is_model_error = ("not found" in err.lower()
                              or "not support" in err.lower()
                              or "404" in err)
            is_rate = any(code in err for code in ["429", "RESOURCE_EXHAUSTED"])
            # Transient SERVER-side failures on the (free, frequently overloaded)
            # Gemma endpoints. 500/INTERNAL is by far the most common in practice
            # and — exactly like 503 — almost always succeeds on a short retry, so
            # it MUST be retried, not raised. Leaving 500 out of this set was
            # collapsing every 15-segment batch into the slow per-segment fallback
            # (~40% of Gemma calls 500), burning idle GPU quota during Step 4.
            # A client-side hard timeout (our _run_with_timeout) or any httpx/SDK read
            # timeout is transient: back off and retry the same model, then advance —
            # never raise straight out and abort Step 4.
            is_timeout = ("timeout" in err.lower() or "timed out" in err.lower())
            is_transient = is_timeout or any(code in err for code in [
                "500", "INTERNAL", "502", "503", "504",
                "UNAVAILABLE", "overloaded", "DEADLINE_EXCEEDED",
            ])

            if is_model_error:
                # Wrong/renamed id → next model immediately (self-heal to Gemini).
                _emit(f"  [Gemini API] '{model_name}' unavailable; advancing to next model.")
                idx += 1
                rate_retries = 0
                transient_retries = 0
                parse_retries = 0
                continue

            if is_rate and rate_retries < max_rate_retries:
                # Back off and retry the SAME model — do not waste the fallback.
                rate_retries += 1
                server_delay = _parse_retry_delay(err)
                if server_delay is not None:
                    wait = server_delay
                else:
                    wait = min(2.0 ** rate_retries + random.uniform(0, 1.5), 90.0)
                _emit(f"  [Gemini API] rate limited; backing off {wait:.1f}s "
                      f"then retrying '{model_name}' (rate retry {rate_retries}/{max_rate_retries}).")
                time.sleep(wait)
                continue

            if is_transient and transient_retries < max_transient_retries:
                # Flaky 500/503 on the SAME model — short back-off then retry it,
                # rather than thrashing down the chain to a weaker/slower model.
                # These endpoints recover within a second or two, so this keeps the
                # batch on the fast primary and avoids the sequential collapse.
                transient_retries += 1
                wait = min(1.5 * (2.0 ** (transient_retries - 1)) + random.uniform(0, 1.0), 20.0)
                _emit(f"  [Gemini API] transient server error; backing off {wait:.1f}s "
                      f"then retrying '{model_name}' "
                      f"(transient retry {transient_retries}/{max_transient_retries}).")
                time.sleep(wait)
                continue

            if (is_rate or is_transient) and idx < len(chain) - 1:
                # Exhausted same-model retries → next model.
                _emit(f"  [Gemini API] advancing from '{model_name}' to next model.")
                idx += 1
                rate_retries = 0
                transient_retries = 0
                parse_retries = 0
                continue

            raise

    raise RuntimeError(
        f"[Gemini API] exhausted model chain {chain} without a successful response "
        f"(tries={total_tries}, likely daily/rate caps)."
    )



# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _build_batch_prompt(
    segments_with_budgets: list,
    target_language: str,
    n_candidates: int,
) -> str:
    """
    Build a Chain-of-Thought prompt that generates N_CANDIDATES translations
    per segment, each phoneme-count compliant AND meaning-preserving.
    """
    lang_cap = target_language.capitalize()
    examples_json = json.dumps(segments_with_budgets, ensure_ascii=False, indent=2)

    return f"""You are an expert {lang_cap} dubbing translator with deep knowledge of phonetics.
Your task is to translate English video segments into {lang_cap} for audio dubbing.

TWO HARD CONSTRAINTS — a good candidate must satisfy BOTH:

1. MEANING (semantic fidelity): the {lang_cap} translation must convey the SAME
   meaning as the English source. It will be scored by a cross-lingual semantic
   model; a fluent sentence that drifts in meaning will be REJECTED. Do not add,
   drop, or invent information.

2. TIMING (isochrony): each translated segment must be naturally speakable in the
   same duration as the original English. The "phoneme_budget" field gives the
   target real-phoneme count:
     - "ideal_target" is the bullseye.
     - "min_target" and "max_target" define the acceptable range.
   Staying in this range makes the dubbed audio fit the original video timing.

SEGMENT CONTEXT & GRAMMATICAL INTEGRITY (Studio Dubbing Guidelines):
Each segment contains "english_text" along with adjacent context:
  - "context_before": the source text of the PREVIOUS segment (for context only).
  - "context_after": the source text of the NEXT segment (for context only).

Rules:
1. GRAMMATICAL NATURALNESS: Translations must be grammatically complete, natural, and idiomatic in {lang_cap}. Follow natural {lang_cap} SOV (Subject-Object-Verb) structure. Avoid awkward literal word-by-word calques that leave dangling postpositions (e.g. 'में', 'का', 'के') or isolated prefixes.
2. NO CONTENT DUPLICATION: Translate ONLY the message conveyed by "english_text". Do NOT duplicate information already spoken in context_before, and do NOT steal content that belongs to context_after.
3. SEAMLESS DISCOURSE FLOW: If "english_text" is a clause of a complex sentence, formulate the {lang_cap} clause so it flows naturally in spoken delivery and connects smoothly with neighbouring context without abrupt syntactic breaks.

For EACH segment, generate exactly {n_candidates} candidate translations that span
the meaning/timing trade-off:
- Candidate 1: Most faithful translation (prioritise meaning).
- Candidate 2: Balanced translation (faithful AND inside [min_target, max_target]).
- Candidate 3: Tightest natural phrasing that still preserves the full meaning.

Chain-of-Thought Instructions (apply silently for each segment):
1. Read the English text and its phoneme budget.
2. Draft {n_candidates} translations in {lang_cap} that all preserve the meaning.
3. Estimate the spoken phoneme count of each translation.
4. Ensure at least one candidate falls within [min_target, max_target].
5. To shorten without losing meaning: use shorter synonyms, drop redundant
   particles, restructure — never omit a piece of the message.

Return ONLY a valid JSON array. Each element must be an object:
{{
  "segment_id": <int>,
  "candidates": ["candidate1 text", "candidate2 text", "candidate3 text"]
}}

Do NOT include markdown fences, explanations, or any text outside the JSON array.

Segments to translate:
{examples_json}"""


def _build_feedback_prompt(
    feedback_items: list,
    target_language: str,
    n_candidates: int,
) -> str:
    """Iterative refinement prompt: feed MEASURED semantics + real phoneme counts
    for each segment's current best translation back to the model, and ask for
    improved candidates that close the specific gap identified."""
    lang_cap = target_language.capitalize()
    items_json = json.dumps(feedback_items, ensure_ascii=False, indent=2)

    return f"""You are an expert {lang_cap} dubbing translator refining earlier drafts.

For each segment below you are given your current best {lang_cap} translation and
its MEASURED scores:
- "semantic_similarity": cross-lingual similarity to the English source, 0.0–1.0
  (1.0 = identical meaning). If this is below {SEMANTIC_THRESHOLD}, the translation
  has DRIFTED in meaning and must be corrected first.
- "phoneme_count_now" vs "ideal_phonemes" with "phoneme_status": how far the
  spoken length is from the timing budget. "too_long" means SHORTEN it;
  "too_short" means EXPAND it (add naturally, never pad with filler).
- "audit_issue" (may be absent): a specific meaning or fluency error found in the
  current translation — e.g. inverted meaning, dropped information, unnatural
  grammar, dangling postposition, or duplicated context. When present, this is
  the HIGHEST priority.
- "context_before" / "context_after" (may be absent/empty): the source text of the
  neighbouring segments, for context ONLY. Do NOT repeat meaning from context_before
  or steal meaning from context_after. Ensure {lang_cap} grammar is natural and
  grammatically sound.

For EACH segment, generate exactly {n_candidates} NEW improved {lang_cap} candidates that:
1. Fix the "audit_issue" first if one is given: rewrite so the translation says
   exactly what "english_text" says — no reversal, no invented or dropped meaning,
   no borrowed completion from the neighbouring segments.
2. Fix drift next: if semantic_similarity is low, rewrite so the translation means
   exactly what the English says.
3. Then fit timing: move the phoneme count toward "ideal_phonemes" and inside
   [min_target, max_target], in the direction given by "phoneme_status".
4. Stay natural and idiomatic — these lines will be spoken aloud.

Return ONLY a valid JSON array. Each element must be an object:
{{
  "segment_id": <int>,
  "candidates": ["candidate1 text", "candidate2 text", "candidate3 text"]
}}

Do NOT include markdown fences, explanations, or any text outside the JSON array.

Segments to refine:
{items_json}"""


# ---------------------------------------------------------------------------
# Candidate scoring / selection (semantic gate → phoneme pick)
# ---------------------------------------------------------------------------

def _combined_loss(sem: Optional[float], rel_diff: float) -> float:
    """Global-minima objective. Lower is better. Falls back to phoneme-only loss
    when semantics could not be measured (sem is None)."""
    if sem is None:
        return round(rel_diff, 6)
    return round(SEMANTIC_WEIGHT * (1.0 - sem) + PHONEME_WEIGHT * rel_diff, 6)


def _repetition_penalty(text: str) -> float:
    """Guard against the classic LLM degeneration (adjacent duplicated words)."""
    words = text.split()
    if len(words) <= 3:
        return 0.0
    dupes = sum(1 for i in range(len(words) - 1) if words[i] == words[i + 1])
    return 0.25 * dupes


def _evaluate_candidates(
    source_text: str,
    candidates: list,
    internal_lang: str,
    source_duration: Optional[float],
) -> list:
    """Measure a set of candidates: real phonemes + cross-lingual semantics.

    Returns a list of record dicts (one per unique, non-empty candidate). Semantic
    scores come from ONE batched IndicSBERT encode; if the model is unavailable,
    every record's 'sem' is None (phoneme-only selection downstream).
    """
    uniq = list(dict.fromkeys(c.strip() for c in candidates if c and c.strip()))
    if not uniq:
        return []

    sims = semantic_similarity.score_many(source_text, uniq)  # list[float] | None

    records = []
    for idx, cand in enumerate(uniq):
        diff = phoneme_diff(source_text, cand, internal_lang, source_duration)
        iso = isochrony_score(source_text, cand, internal_lang, source_duration)
        sem = sims[idx] if sims is not None else None
        rep = _repetition_penalty(cand)

        # Spoken phoneme density governor (pps): Prevents cramming excessive syllables
        # into a fixed duration window, which causes IndicF5 to drop initial words/plosives.
        pps = diff["target_phonemes"] / source_duration if source_duration and source_duration > 0 else 0.0
        rate_penalty = 0.0
        if pps > MAX_PHONEME_DENSITY:
            rate_penalty = (pps - MAX_PHONEME_DENSITY) * 0.5
        # bool(...) so a numpy.bool_ (pps is a numpy float) never reaches a JSON dump downstream.
        rate_ok = bool(pps <= MAX_PHONEME_DENSITY) if pps > 0 else True

        records.append({
            "text": cand,
            "sem": sem,
            "isochrony": iso,
            "rel_diff": diff["rel_diff"],
            "abs_diff": diff["abs_diff"],
            "target_phonemes": diff["target_phonemes"],
            "ideal_target": diff["ideal_target"],
            "direction": diff["direction"],
            "pps": round(pps, 2),
            "rate_ok": rate_ok,
            # Repetition and speech-rate penalties folded into the objective
            "loss": _combined_loss(sem, diff["rel_diff"]) + rep + rate_penalty,
        })
    return records


def _select_best(records: list, semantic_threshold: float, phoneme_tolerance: float):
    """Two-stage selection: semantic gate, then closest phoneme count.

    Returns (best_record, satisfied) where `satisfied` is True iff the chosen
    candidate clears BOTH gates and stays within natural speech rate (or clears
    the phoneme gate when semantics are unmeasured this run).
    """
    if not records:
        return None, False

    measured = [r for r in records if r["sem"] is not None]

    if measured:
        gated = [r for r in measured if r["sem"] >= semantic_threshold]
        if gated:
            # Stage B: among meaning-faithful candidates, prefer those within natural speech rate (rate_ok),
            # closest to the phoneme budget wins; combined loss breaks ties.
            rate_gated = [r for r in gated if r.get("rate_ok", True)]
            pool = rate_gated if rate_gated else gated
            pool.sort(key=lambda r: (r["rel_diff"], r["loss"]))
            best = pool[0]
        else:
            # Nothing cleared the meaning gate — keep the MOST faithful candidate
            # (highest similarity), phoneme closeness as tie-break. Better a
            # slightly-off-timing faithful line than an on-time mistranslation.
            measured.sort(key=lambda r: (-r["sem"], r["rel_diff"]))
            best = measured[0]
    else:
        # Semantics unavailable this run: phoneme-fit only (visible degradation).
        rate_records = [r for r in records if r.get("rate_ok", True)]
        pool = rate_records if rate_records else records
        pool.sort(key=lambda r: (r["rel_diff"], -r["isochrony"]))
        best = pool[0]

    sem_ok = best["sem"] is None or best["sem"] >= semantic_threshold
    phon_ok = best["rel_diff"] <= phoneme_tolerance
    rate_ok = best.get("rate_ok", True)
    # bool(...) so the returned `satisfied` (stored as each segment's gates_passed) is a
    # native Python bool, not a numpy.bool_ — the latter is not JSON-serializable and was
    # crashing the Step-6 job-spec write. Cleaning it here keeps every downstream JSON
    # boundary (TTS spec, resume manifest, translation cache) safe at the source.
    return best, bool(sem_ok and phon_ok and rate_ok)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Windowed meaning audit (post-selection safety net)
# ---------------------------------------------------------------------------

def _slice_audit_windows(
    segments: list,
    translated_segments: list,
    window_size: int = AUDIT_WINDOW_SIZE,
    overlap: int = AUDIT_WINDOW_OVERLAP,
    filter_ids: Optional[set] = None,
) -> list:
    """
    Partition segments into overlapping windows for the LLM meaning audit.

    Each window contains:
      - 'target_items': slice of segments to be audited & assigned verdicts
      - 'context_before': up to `overlap` prior segments (read-only context)
      - 'context_after': up to `overlap` subsequent segments (read-only context)
      - 'window_idx': 0-indexed window number
      - 'total_windows': total number of windows
      - 'target_ids': list of segment IDs targeted in this window

    If `filter_ids` is provided (e.g. during auto-heal), windows containing none
    of the target IDs are skipped, avoiding unnecessary LLM calls.
    """
    n = len(segments)
    if n == 0:
        return []

    windows_meta = []
    w_start = 0
    while w_start < n:
        w_end = min(w_start + window_size, n)
        windows_meta.append((w_start, w_end))
        w_start = w_end

    total_windows = len(windows_meta)
    result = []
    for w_idx, (w_start, w_end) in enumerate(windows_meta):
        target_ids = list(range(w_start, w_end))
        if filter_ids is not None and not (set(target_ids) & filter_ids):
            continue

        ctx_before_start = max(0, w_start - overlap)
        ctx_after_end = min(n, w_end + overlap)

        target_items = [
            {"segment_id": i, "english": segments[i]["text"], "translation": translated_segments[i]["text"]}
            for i in range(w_start, w_end)
        ]
        before_items = [
            {"segment_id": i, "english": segments[i]["text"], "translation": translated_segments[i]["text"]}
            for i in range(ctx_before_start, w_start)
        ]
        after_items = [
            {"segment_id": i, "english": segments[i]["text"], "translation": translated_segments[i]["text"]}
            for i in range(w_end, ctx_after_end)
        ]

        result.append({
            "window_idx": w_idx,
            "total_windows": total_windows,
            "target_items": target_items,
            "context_before": before_items,
            "context_after": after_items,
            "target_ids": target_ids,
        })
    return result


def _build_audit_prompt(
    target_items: list,
    context_before: list,
    context_after: list,
    target_language: str,
    window_idx: int = 0,
    total_windows: int = 1,
    check_completeness: bool = True,
) -> str:
    """Build an audit prompt for one window of segments, explicitly showing
    surrounding context before & after to prevent false-positive flags on
    sentence fragments split across cuts.

    When ``check_completeness`` is set, the reviewer ALSO flags a translation that
    is grammatically UNFINISHED (drops the closing verb/copula, trails off on a
    postposition) — but ONLY when THAT segment's own english is itself a complete
    sentence. That source-completeness gate keeps this disjoint from "fabrication":
    a fragment translating a fragment stays "ok" and is never force-completed."""
    lang_cap = target_language.capitalize()
    target_json = json.dumps(target_items, ensure_ascii=False, indent=2)

    context_sections = []
    if total_windows > 1:
        context_sections.append(
            f"AUDIT SCOPE: Window {window_idx + 1} of {total_windows}. "
            f"You are auditing segments {target_items[0]['segment_id']} to {target_items[-1]['segment_id']}.\n"
        )

    if context_before:
        before_json = json.dumps(context_before, ensure_ascii=False, indent=2)
        context_sections.append(
            "--- PRIOR CONTEXT (FOR BACKGROUND UNDERSTANDING ONLY - DO NOT AUDIT OR RETURN IN OUTPUT) ---\n"
            f"{before_json}\n"
        )

    context_sections.append(
        "--- TARGET SEGMENTS TO AUDIT (YOU MUST JUDGE EVERY SEGMENT IN THIS LIST) ---\n"
        f"{target_json}\n"
    )

    if context_after:
        after_json = json.dumps(context_after, ensure_ascii=False, indent=2)
        context_sections.append(
            "--- SUBSEQUENT CONTEXT (FOR BACKGROUND UNDERSTANDING ONLY - DO NOT AUDIT OR RETURN IN OUTPUT) ---\n"
            f"{after_json}\n"
        )

    body = "\n".join(context_sections)
    target_ids_str = f"from {target_items[0]['segment_id']} to {target_items[-1]['segment_id']}"

    if check_completeness:
        header_scope = "MEANING FIDELITY and GRAMMATICAL COMPLETENESS"
        incomplete_rule = (
            f'- "incomplete": the {lang_cap} translation is grammatically UNFINISHED — it drops the '
            f'finite verb or copula the sentence requires (e.g. it ends on a noun or postposition '
            f'where Hindi "है"/"हैं"/"था" is needed), or it trails off on a postposition '
            f'(में, का, के, को, से, पर). Use this verdict ONLY when THIS segment\'s OWN english is a '
            f'COMPLETE sentence (it ends with . ! or ?). Example: english "Welcome to Fun Science '
            f'Demos." rendered as "फन साइंस डेमोज़ में स्वागत।" is INCOMPLETE — the natural finished '
            f'form is "फन साइंस डेमोज़ में आपका स्वागत है।". If this segment\'s english is itself a '
            f'fragment that continues in the next segment, NEVER use "incomplete" (that would be '
            f'"fabrication" in reverse).\n'
        )
        verdict_enum = "ok|inversion|fabrication|addition|omission|drift|incomplete"
        fluency_note = (
            'Do NOT flag for style, word choice, naturalness, or length/timing. A terse but faithful, '
            'grammatically complete line is "ok". A fragment that stops mid-thought because its english '
            'ALSO stops mid-thought is "ok" — but a translation left grammatically unfinished while its '
            'OWN english is a complete sentence is "incomplete", not "ok".'
        )
    else:
        header_scope = "MEANING FIDELITY ONLY"
        incomplete_rule = ""
        verdict_enum = "ok|inversion|fabrication|addition|omission|drift"
        fluency_note = (
            'Do NOT flag for style, word choice, naturalness, fluency, or length/timing. A terse but '
            'faithful line is "ok". A fragment that stops mid-thought because its english also stops '
            'mid-thought is "ok".'
        )

    return f"""You are a senior bilingual {lang_cap} dubbing reviewer. Audit a finished translation for {header_scope}.

You are given a window of numbered segments. Read the PRIOR and SUBSEQUENT context segments to understand incomplete thoughts and sentences that span across segment boundaries. Then judge EACH TARGET segment's translation against ITS OWN english source.

HOW THIS TRANSCRIPT IS SEGMENTED:
The English was split by automatic speech recognition, which routinely CUTS SENTENCES IN THE MIDDLE. Many segments start and/or end mid-sentence. This is EXPECTED:
- A translation that faithfully renders a sentence FRAGMENT — even one that reads incomplete on its own — is CORRECT. Verdict "ok".
- Judge each translation ONLY against the words in ITS OWN english segment; use the surrounding context only to understand where the sentence began or where it continues.

Flag a segment (needs_fix = true) ONLY for a real error against its own english source:
- "inversion": reverses or negates the meaning (drops or adds a "not", says the opposite).
- "fabrication": invents an ending or information not present in THIS segment's english — most commonly by "completing" a sentence whose real continuation lives in the next segment.
- "addition": states meaning not in this segment's english (including content that belongs to a neighbouring segment).
- "omission": drops a meaningful part of what this segment's english actually says.
- "drift": says something materially different from the source, not covered above.
{incomplete_rule}Everything else is verdict "ok", needs_fix false.

{fluency_note}

Return ONLY valid JSON (no markdown fences, no commentary) of exactly this shape:
{{"audits": [{{"segment_id": <int>, "verdict": "{verdict_enum}", "needs_fix": <true|false>, "reason": "<one short sentence naming the problem, or 'faithful'>"}}]}}
Include EVERY segment_id in the TARGET SEGMENTS ({target_ids_str}) exactly once. Do NOT include verdicts for context segments.

{body}"""


def _parse_audit(raw: str, expected_ids: Optional[set] = None) -> dict:
    """Normalize an audit reply into {segment_id: {verdict, needs_fix, reason}}.
    Accepts the schema shape ({"audits": [...]}) and a bare array; reconciles a
    missing/oddly-typed needs_fix against the verdict so a flagged line is never
    silently treated as clean. If expected_ids is provided, ignores context segments
    that may have been inadvertently returned."""
    data = _extract_json(raw)
    if isinstance(data, dict):
        items = data.get("audits") or data.get("segments") or data.get("results") or []
    elif isinstance(data, list):
        items = data
    else:
        items = []
    out = {}
    for item in items:
        if not isinstance(item, dict) or "segment_id" not in item:
            continue
        try:
            sid = int(item["segment_id"])
        except (TypeError, ValueError):
            continue
        if expected_ids is not None and sid not in expected_ids:
            continue
        verdict = str(item.get("verdict", "")).strip().lower()
        reason = str(item.get("reason", "")).strip()
        nf = item.get("needs_fix")
        if isinstance(nf, str):
            needs_fix = nf.strip().lower() in ("true", "1", "yes", "y")
        else:
            needs_fix = bool(nf)
        # Reconcile verdict and needs_fix so the two can never disagree:
        if verdict in ("", "ok", "faithful", "good", "fine", "correct"):
            verdict, needs_fix = "ok", False
        elif nf is None:
            # A problem verdict with needs_fix omitted must still be fixed.
            needs_fix = True
        out[sid] = {"verdict": verdict, "needs_fix": needs_fix, "reason": reason}
    return out


# Closed-class Hindi postpositions and compound-postposition tails. A target
# sentence that ENDS on one of these (before final punctuation), while its own
# english is a complete sentence, has trailed off — grammatically unfinished.
_HI_POSTPOSITIONS = {
    "में", "का", "के", "की", "को", "से", "पर", "ने", "तक", "पे", "द्वारा",
}
_HI_COMPOUND_TAILS = {
    "लिए", "साथ", "बाद", "पास", "ऊपर", "नीचे", "सामने", "बारे", "तरफ", "ओर",
    "अनुसार", "दौरान", "खिलाफ", "रूप", "कारण", "जरिए", "जरिये", "माध्यम", "बजाय",
    "अलावा", "भीतर", "बाहर", "पीछे", "आगे", "बीच",
}
_SENT_SPLIT_RE = re.compile(r"[।.!?]+")


def _source_is_complete_sentence(text: str) -> bool:
    """A source english span is treated as a complete sentence when it ends in
    terminal punctuation. Whisper's resegmentation aims for complete sentences, so
    this is the discriminator that keeps the completeness check from ever trying to
    finish a genuine mid-clause fragment (which would be fabrication)."""
    t = (text or "").strip()
    return bool(t) and t[-1] in ".!?"


def _scan_dangling_targets(segments: list, translated_segments: list, internal_lang: str) -> dict:
    """Deterministic, zero-API completeness backstop for Hindi (Devanagari).

    Flags a segment when BOTH: (a) its own english is a complete sentence, and
    (b) its translation's FINAL sentence ends on a closed-class postposition or a
    compound-postposition tail (के लिए / के बाद / …) — an unambiguous 'trailed off'
    pattern. Returns {segment_id: reason}. High precision by construction: it fires
    only on the closed postposition set, so it never rewrites a valid verb-final
    line. The subtler 'dropped copula on a noun-final clause' case (e.g. '…में
    स्वागत।' missing 'है') is out of reach of a closed-class rule and is left to the
    LLM audit's "incomplete" verdict."""
    if internal_lang != "hindi":
        return {}
    flagged = {}
    for i, seg in enumerate(segments):
        if i >= len(translated_segments):
            break
        if not _source_is_complete_sentence(seg.get("text", "")):
            continue
        tgt = (translated_segments[i].get("text") or "").strip()
        if not tgt:
            continue
        parts = [p.strip() for p in _SENT_SPLIT_RE.split(tgt) if p.strip()]
        if not parts:
            continue
        toks = parts[-1].split()
        if not toks:
            continue
        last = toks[-1]
        prev = toks[-2] if len(toks) >= 2 else ""
        dangling = last in _HI_POSTPOSITIONS or (
            last in _HI_COMPOUND_TAILS and prev in ("के", "की", "का")
        )
        if dangling:
            flagged[i] = (
                f'grammatically incomplete: the sentence trails off on "{last}" without its '
                f"finishing verb/phrase — complete it naturally using only this segment's meaning."
            )
    return flagged


def _audit_once(
    client,
    segments: list,
    translated_segments: list,
    target_language: str,
    log_fn: Optional[Callable[[str], None]] = None,
    filter_ids: Optional[set] = None,
    check_completeness: bool = True,
) -> dict:
    """Run meaning (and, when `check_completeness`, grammatical-completeness) audit on
    the Gemini refine chain. For long transcripts, splits into overlapping windows
    (AUDIT_WINDOW_SIZE with AUDIT_WINDOW_OVERLAP context). If `filter_ids` is provided
    (e.g. during auto-heal), only audits windows containing those flagged segments,
    saving API calls.
    Returns {segment_id: {verdict, needs_fix, reason}}."""
    windows = _slice_audit_windows(
        segments, translated_segments,
        window_size=AUDIT_WINDOW_SIZE,
        overlap=AUDIT_WINDOW_OVERLAP,
        filter_ids=filter_ids,
    )
    if not windows:
        return {}

    total_windows = len(windows)
    merged_audits = {}
    served_models: List[str] = []

    for win in windows:
        w_idx = win["window_idx"]
        w_tot = win["total_windows"]
        target_ids = win["target_ids"]
        expected_ids = set(target_ids)

        if w_tot > 1:
            w_msg = f"  [IsochronyTranslation] Auditing window {w_idx + 1}/{w_tot} (segments {target_ids[0]}–{target_ids[-1]})..."
            if log_fn:
                log_fn(w_msg)
            print(w_msg)

        prompt = _build_audit_prompt(
            win["target_items"],
            win["context_before"],
            win["context_after"],
            target_language,
            window_idx=w_idx,
            total_windows=w_tot,
            check_completeness=check_completeness,
        )

        try:
            raw = _call_gemini(
                client, prompt, temperature=0.1,
                response_schema=TranscriptAuditResponse,
                response_mime_type="application/json",
                log_fn=log_fn,
                models=_refine_models(),
                served=served_models,
                validate_fn=lambda r: _parse_audit(r, expected_ids=expected_ids),
            )
            parsed = _parse_audit(raw, expected_ids=expected_ids)
            if not parsed:
                raise ValueError("no parseable audits in window reply")
            merged_audits.update(parsed)
        except Exception as e:
            err_msg = f"  [IsochronyTranslation] Audit window {w_idx + 1}/{w_tot} failed ({e}); skipping window."
            if log_fn:
                log_fn(err_msg)
            print(err_msg)
            for sid in target_ids:
                merged_audits[sid] = {"verdict": "audit_failed", "needs_fix": False, "reason": str(e)[:60]}

    if served_models and log_fn:
        uniq = ", ".join(dict.fromkeys(served_models))
        log_fn(f"  [IsochronyTranslation] Audit served by: {uniq}")
    return merged_audits


# ---------------------------------------------------------------------------
# Core translation function
# ---------------------------------------------------------------------------

def translate_segments_isochrony(
    segments: list,
    target_language: str,
    api_key: str,
    n_candidates: int = N_CANDIDATES,
    min_score: float = MINIMUM_ACCEPTABLE_SCORE,
    log_fn: Optional[Callable[[str], None]] = None,
    semantic_threshold: float = SEMANTIC_THRESHOLD,
    phoneme_tolerance: float = PHONEME_TOLERANCE,
    max_iterations: int = MAX_ITERATIONS,
    use_cache: bool = True,
) -> list:
    """
    Translate a list of transcribed segments into an Indic language with
    isochrony (duration) constraints AND semantic-fidelity gating.

    Selection per segment: keep candidates whose cross-lingual meaning matches the
    source (IndicSBERT similarity >= `semantic_threshold`), then pick the one whose
    real phoneme count is closest to the duration-grounded budget. The model is
    iterated (Chain-of-Thought + measured feedback) until every segment clears both
    gates or the combined objective stops improving, up to `max_iterations` rounds.

    Backward compatible: the first three args and `n_candidates` / `min_score` /
    `log_fn` are unchanged; the v2.5 knobs (including `use_cache`) are optional
    with sensible defaults.

    Rate-limit strategy (v2.5.1): the iteration-0 BULK batch runs on lenient
    Gemma models; only the few REFINEMENT rounds use gemini-3.1-flash-lite. A
    persistent candidate cache (keyed by language+source) seeds each segment
    before any API call, so a re-run — or a video with repeated phrases — selects
    its final lines with far fewer requests, often zero.
    """
    if not api_key:
        raise ValueError("Gemini API key is required.")

    internal_lang = DISPLAY_TO_INTERNAL.get(target_language, target_language.lower())
    if internal_lang not in LANGUAGE_CODES:
        raise ValueError(
            f"Unsupported language: '{target_language}'. "
            f"Supported: {list(DISPLAY_TO_INTERNAL.keys())}"
        )

    client = _build_client(api_key)
    cache = translation_cache.TranslationCache(enabled=use_cache)

    def _log(msg):
        if log_fn:
            log_fn(msg)
        print(msg)

    # --- Step 1: Compute duration-grounded phoneme budgets for all segments ---
    # Each segment also carries its neighbours' SOURCE text (context_before/after)
    # so the translator can see where a mid-sentence cut is going and never invent
    # a false completion — the root cause of the fabrication errors the audit
    # (Step 5) otherwise has to catch after the fact.
    enriched = []
    n_seg = len(segments)
    for i, seg in enumerate(segments):
        duration = seg["end"] - seg["start"]
        budget = compute_target_budget(seg["text"], internal_lang, source_duration=duration)
        enriched.append({
            "segment_id": i,
            "english_text": seg["text"],
            "context_before": segments[i - 1]["text"] if i > 0 else "",
            "context_after": segments[i + 1]["text"] if i < n_seg - 1 else "",
            "duration_seconds": round(duration, 2),
            "phoneme_budget": budget,
        })

    _log(f"[IsochronyTranslation] Translating {len(segments)} segments → {target_language}")
    # available() lazily loads the IndicSBERT model on first call — a cold model load can
    # take ~30s. Announce it so that stretch reads as "loading", not "stuck".
    _log("[IsochronyTranslation] Initializing semantic gate (IndicSBERT); first-time model "
         "load can take ~30s...")
    if semantic_similarity.available():
        _log(f"[IsochronyTranslation] Semantic gate: IndicSBERT active "
             f"(threshold {semantic_threshold}, ruler {active_ruler()}).")
    else:
        # Surface the ACTUAL cause into this (file-first) log — the distinguishing WARNING is
        # emitted on semantic_similarity's Python logger, which a subprocess/container log may
        # never capture. Distinguishes "package missing" from "model download/load failed".
        reason = semantic_similarity.unavailable_reason()
        _log("[IsochronyTranslation] Semantic gate UNAVAILABLE — selecting on phoneme fit "
             "only. Cause: " + (reason or "unknown (model load returned no error detail)"))

    # Per-segment best-so-far record, tracked across every iteration (global minima).
    best_by_seg: dict = {i: None for i in range(len(segments))}
    satisfied: dict = {i: False for i in range(len(segments))}

    def _merge(seg_id, records):
        """Fold newly measured candidates into a segment's running best. Returns
        True if this segment's combined loss strictly improved."""
        best, sat = _select_best(records, semantic_threshold, phoneme_tolerance)
        if best is None:
            return False
        improved = False
        prev = best_by_seg[seg_id]
        if prev is None or best["loss"] < prev["loss"] - 1e-9:
            best_by_seg[seg_id] = best
            improved = True
        # `satisfied` reflects whether the CURRENT best clears both gates and natural speech rate.
        cur = best_by_seg[seg_id]
        cur_ok = (cur["sem"] is None or cur["sem"] >= semantic_threshold) and \
                 (cur["rel_diff"] <= phoneme_tolerance) and \
                 cur.get("rate_ok", True)
        satisfied[seg_id] = cur_ok
        return improved

    # --- Step 2: Iteration 0 — batch generate initial candidates for all segs ---
    # A smaller batch means a smaller per-call OUTPUT (batch_size * n_candidates
    # translations). The free Gemma endpoints 500 far more often on large
    # generations, and every 500 that survives the retry loop collapses the whole
    # batch to slow per-segment calls — so keep batches modest. Env-tunable.
    try:
        batch_size = max(1, int(os.environ.get("DUBBING_TRANSLATE_BATCH_SIZE", "8")))
    except ValueError:
        batch_size = 8

    def _generate_batch(items, prompt_builder, temperature, models, phase="batch"):
        """Run batched Gemini generation on the given model chain; returns
        {seg_id: [candidate,...]}. `models` selects the phase — Gemma-first for
        the iteration-0 bulk, the Gemini ladder for refinement. `phase` is only a
        label for the "served by" confirmation line.

        The independent per-chunk calls run with bounded concurrency
        (_gemini_concurrency) so a long clip's many batches overlap their response
        waits instead of running strictly one-after-another; _throttle still keeps
        each model under its RPM. DUBBING_GEMINI_CONCURRENCY=1 => the old serial path."""
        items = list(items)
        chunks = [items[b:b + batch_size] for b in range(0, len(items), batch_size)]
        total = len(chunks)

        def run_one(numbered):
            bnum, chunk = numbered
            served_local: List[str] = []
            _log(f"  [IsochronyTranslation] Gemini batch {bnum}/{total} ({len(chunk)} segments)...")
            try:
                prompt = prompt_builder(chunk)
                raw = _call_gemini(
                    client, prompt, temperature=temperature,
                    response_schema=BatchTranslationResponse,
                    response_mime_type="application/json",
                    log_fn=log_fn,
                    models=models,
                    served=served_local,
                    # On an unparseable reply, reask the SAME model for clean JSON
                    # (up to twice) before falling through the chain.
                    validate_fn=_parse_batch,
                )
                # Robust parse: handles the schema shape (Gemini) AND the bare
                # array a schema-less Gemma reply follows from the prompt.
                parsed = _parse_batch(raw)
                if not parsed:
                    raise ValueError("no parseable segments in batch reply")
                return parsed, served_local
            except Exception as e:
                _log(f"    [IsochronyTranslation] Batch {bnum} failed: {e}. Falling back sequentially...")
                # Sequential fallback expects the enriched-item shape.
                seq_items = [
                    it if "phoneme_budget" in it else enriched[it["segment_id"]]
                    for it in chunk
                ]
                part = _translate_sequential(
                    client, seq_items, internal_lang, n_candidates,
                    log_fn=log_fn, models=models, served=served_local,
                )
                return part, served_local

        out, served = _run_batches_concurrent(
            list(enumerate(chunks, 1)), run_one, _gemini_concurrency(), log_fn=log_fn,
        )
        if served:
            # Confirm which model(s) ACTUALLY served this phase — for the bulk
            # phase this is the check that Gemma (not Gemini) took the load.
            uniq = ", ".join(dict.fromkeys(served))
            _log(f"  [IsochronyTranslation] {phase} phase served by: {uniq}")
        return out

    # --- Step 2a: Seed from the persistent cache (free — no API calls) ---------
    # Selection is local, so any cached candidate that already clears both gates
    # removes that segment from the generation batch entirely.
    cache_hits = 0
    for i in range(len(segments)):
        cached = cache.get(internal_lang, segments[i]["text"])
        if cached:
            recs = _evaluate_candidates(
                segments[i]["text"], cached, internal_lang,
                source_duration=enriched[i]["duration_seconds"],
            )
            _merge(i, recs)
            if satisfied[i]:
                cache_hits += 1
    if cache.enabled:
        _log(f"[IsochronyTranslation] Cache: {cache_hits}/{len(segments)} segment(s) "
             f"satisfied from cache before any API call ({cache.stats()['keys']} keys on disk).")

    # --- Step 2b: Iteration 0 — bulk-generate ONLY the still-unsatisfied segs ---
    to_generate = [enriched[i] for i in range(len(segments)) if not satisfied[i]]
    if to_generate:
        _log(f"[IsochronyTranslation] Iteration 0: bulk-generating {n_candidates} "
             f"candidates/segment for {len(to_generate)} segment(s) on the Gemma chain...")
        gen = _generate_batch(
            to_generate,
            lambda chunk: _build_batch_prompt(chunk, internal_lang, n_candidates),
            temperature=0.5,
            models=_bulk_models(),
            phase="Iteration 0 (bulk)",
        )
        for i in range(len(segments)):
            new_cands = gen.get(i, [])
            if not new_cands:
                continue
            cache.add(internal_lang, segments[i]["text"], new_cands)
            recs = _evaluate_candidates(
                segments[i]["text"], new_cands, internal_lang,
                source_duration=enriched[i]["duration_seconds"],
            )
            _merge(i, recs)
    else:
        _log("[IsochronyTranslation] Iteration 0 skipped — every segment satisfied from cache.")

    # --- Step 3: Iterative refinement until both gates pass or global minima ---
    for iteration in range(1, max_iterations + 1):
        pending = [i for i in range(len(segments)) if not satisfied[i]]
        if not pending:
            _log(f"[IsochronyTranslation] All segments cleared both gates after "
                 f"{iteration - 1} refinement round(s).")
            break

        _log(f"[IsochronyTranslation] Iteration {iteration}/{max_iterations}: "
             f"refining {len(pending)} unsatisfied segment(s)...")

        feedback_items = []
        for i in pending:
            best = best_by_seg[i]
            budget = enriched[i]["phoneme_budget"]
            if best is None:
                # No usable candidate yet — re-issue the original ask for this seg.
                feedback_items.append({
                    "segment_id": i,
                    "english_text": segments[i]["text"],
                    "context_before": enriched[i]["context_before"],
                    "context_after": enriched[i]["context_after"],
                    "current_best_translation": "",
                    "semantic_similarity": "not measured",
                    "phoneme_count_now": 0,
                    "ideal_phonemes": budget["ideal_target"],
                    "phoneme_status": "missing",
                    "min_target": budget["min_target"],
                    "max_target": budget["max_target"],
                })
                continue
            status = (
                f"{best['abs_diff']:.0f} phonemes too long" if best["direction"] == "too_long"
                else f"{best['abs_diff']:.0f} phonemes too short" if best["direction"] == "too_short"
                else "on budget"
            )
            feedback_items.append({
                "segment_id": i,
                "english_text": segments[i]["text"],
                "context_before": enriched[i]["context_before"],
                "context_after": enriched[i]["context_after"],
                "current_best_translation": best["text"],
                "semantic_similarity": best["sem"] if best["sem"] is not None else "not measured",
                "phoneme_count_now": best["target_phonemes"],
                "ideal_phonemes": best["ideal_target"],
                "phoneme_status": status,
                "min_target": budget["min_target"],
                "max_target": budget["max_target"],
            })

        gen = _generate_batch(
            feedback_items,
            lambda chunk: _build_feedback_prompt(chunk, internal_lang, n_candidates),
            temperature=0.4,
            models=_refine_models(),
            phase=f"Iteration {iteration} (refine)",
        )

        any_improved = False
        for i in pending:
            new_cands = gen.get(i, [])
            if new_cands:
                cache.add(internal_lang, segments[i]["text"], new_cands)
            recs = _evaluate_candidates(
                segments[i]["text"], new_cands, internal_lang,
                source_duration=enriched[i]["duration_seconds"],
            )
            if _merge(i, recs):
                any_improved = True

        if not any_improved:
            # Combined objective reached a global minimum for every remaining
            # segment — more iterations will not help. Stop early.
            _log(f"[IsochronyTranslation] Iteration {iteration}: no segment improved "
                 f"(global minimum reached). Stopping refinement.")
            break

    # Persist the accumulated candidate pool so future runs get cache hits.
    cache.save()

    # --- Step 4: Assemble output ---
    translated_segments = []
    for i, seg in enumerate(segments):
        best = best_by_seg[i]
        if best is None:
            _log(f"  [Segment {i}] No usable candidate produced. Emitting empty text.")
            translated_segments.append({
                "start": seg["start"],
                "end": seg["end"],
                "text": "",
                "isochrony_score": 0.0,
                "semantic_score": None,
                "phoneme_count": 0,
                "ideal_phonemes": enriched[i]["phoneme_budget"]["ideal_target"],
                "duration": round(seg["end"] - seg["start"], 2),
                "gates_passed": False,
            })
            continue

        sem_str = f"{best['sem']:.3f}" if best["sem"] is not None else "n/a"
        _log(
            f"  [Segment {i}] gates_passed={satisfied[i]} | sim={sem_str} "
            f"| iso={best['isochrony']:.3f} | phonemes={best['target_phonemes']}/"
            f"{best['ideal_target']:.1f} (Δ{best['abs_diff']:.0f}, {best['direction']}) "
            f"| text: {best['text'][:40]}..."
        )
        translated_segments.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": best["text"],
            "isochrony_score": best["isochrony"],
            "semantic_score": best["sem"],
            "phoneme_count": best["target_phonemes"],
            "ideal_phonemes": best["ideal_target"],
            "duration": round(seg["end"] - seg["start"], 2),
            "gates_passed": satisfied[i],
        })

    # --- Step 5: Whole-transcript meaning audit + bounded auto-heal -----------
    # The per-segment gates above were each computed on ONE segment with an
    # embedder, which structurally cannot see negation/inversion or a fabricated
    # completion of a sentence that continues in the NEXT segment. This pass
    # re-reads the FULL source and FULL translation together with an LLM reviewer,
    # flags per-segment meaning errors, and re-translates the flagged ones inside
    # their EXISTING phoneme budget (a meaning fix must not break timing). Anything
    # still flagged after AUDIT_MAX_FIX_ROUNDS is kept as the best candidate and
    # flagged in the log + output — degrade, don't crash.
    for ts in translated_segments:
        ts.setdefault("audit_verdict", "not_audited")
        ts.setdefault("audit_reason", "")
        ts.setdefault("audit_passed", True)

    def _refresh_seg_dict(i):
        """Rewrite translated_segments[i] from the current best_by_seg[i]."""
        nb = best_by_seg[i]
        if nb is None:
            return
        translated_segments[i].update({
            "text": nb["text"],
            "isochrony_score": nb["isochrony"],
            "semantic_score": nb["sem"],
            "phoneme_count": nb["target_phonemes"],
            "ideal_phonemes": nb["ideal_target"],
            "gates_passed": satisfied[i],
        })

    _audit_flag = os.environ.get("DUBBING_TRANSLATION_AUDIT", "1").strip().lower()
    audit_on = _audit_flag not in ("0", "false", "no", "off", "")
    _completeness_flag = os.environ.get("DUBBING_COMPLETENESS_CHECK", "1").strip().lower()
    completeness_on = _completeness_flag not in ("0", "false", "no", "off", "")

    if not audit_on:
        _log("[IsochronyTranslation] Step 5 translation audit DISABLED (DUBBING_TRANSLATION_AUDIT=0).")
    else:
        _log("[IsochronyTranslation] Step 5: whole-transcript meaning audit "
             "(catches inversion / fabricated completions + grammatically unfinished lines "
             "the per-segment gate cannot)...")
        audits = _audit_once(client, segments, translated_segments, target_language,
                             log_fn=log_fn, check_completeness=completeness_on)
        # Deterministic Devanagari backstop (no API cost): force-flag a target that
        # trails off on a postposition though its OWN english is a complete sentence,
        # in case the LLM audit (tuned for fragment-tolerance) let it pass.
        dangling = _scan_dangling_targets(segments, translated_segments, internal_lang) if completeness_on else {}
        if not audits and not dangling:
            _log("[IsochronyTranslation] Audit unavailable — keeping per-segment selections "
                 "unaudited (run not aborted).")
            for ts in translated_segments:
                ts["audit_verdict"] = "audit_unavailable"
        else:
            if not audits:
                audits = {}
            first_flags = {i for i in range(len(segments))
                           if audits.get(i) and audits[i]["needs_fix"]}
            for i, reason in dangling.items():
                if not (audits.get(i) and audits[i]["needs_fix"]):
                    audits[i] = {"verdict": "incomplete", "needs_fix": True, "reason": reason}
                    first_flags.add(i)
            n_flag0 = len(first_flags)
            _log(f"[IsochronyTranslation] Audit pass 1: {n_flag0}/{len(segments)} segment(s) "
                 f"flagged (meaning + completeness)" + (f": {sorted(first_flags)}" if first_flags else "."))

            for fix_round in range(1, AUDIT_MAX_FIX_ROUNDS + 1):
                flagged = [i for i in range(len(segments))
                           if audits.get(i) and audits[i]["needs_fix"]]
                if not flagged:
                    break
                _log(f"[IsochronyTranslation] Audit heal round {fix_round}/{AUDIT_MAX_FIX_ROUNDS}: "
                     f"re-translating {len(flagged)} segment(s) {flagged} within budget...")

                heal_items = []
                for i in flagged:
                    best = best_by_seg[i]
                    budget = enriched[i]["phoneme_budget"]
                    if best is not None:
                        status = (
                            f"{best['abs_diff']:.0f} phonemes too long" if best["direction"] == "too_long"
                            else f"{best['abs_diff']:.0f} phonemes too short" if best["direction"] == "too_short"
                            else "on budget"
                        )
                        sem_val = best["sem"] if best["sem"] is not None else "not measured"
                        phon_now, cur_text = best["target_phonemes"], best["text"]
                    else:
                        status, sem_val, phon_now, cur_text = "missing", "not measured", 0, ""
                    heal_items.append({
                        "segment_id": i,
                        "english_text": segments[i]["text"],
                        "context_before": enriched[i]["context_before"],
                        "context_after": enriched[i]["context_after"],
                        "current_best_translation": cur_text,
                        "audit_issue": audits[i]["reason"] or audits[i]["verdict"],
                        "semantic_similarity": sem_val,
                        "phoneme_count_now": phon_now,
                        "ideal_phonemes": budget["ideal_target"],
                        "phoneme_status": status,
                        "min_target": budget["min_target"],
                        "max_target": budget["max_target"],
                    })

                gen = _generate_batch(
                    heal_items,
                    lambda chunk: _build_feedback_prompt(chunk, internal_lang, n_candidates),
                    temperature=0.4,
                    models=_refine_models(),
                    phase=f"Audit heal {fix_round}",
                )

                healed_any = False
                for i in flagged:
                    new_cands = gen.get(i, [])
                    if not new_cands:
                        continue
                    cache.add(internal_lang, segments[i]["text"], new_cands)
                    new_recs = _evaluate_candidates(
                        segments[i]["text"], new_cands, internal_lang,
                        source_duration=enriched[i]["duration_seconds"],
                    )
                    new_best, new_sat = _select_best(new_recs, semantic_threshold, phoneme_tolerance)
                    if new_best is None:
                        continue
                    # The flagged line was judged meaning-WRONG by the reviewer, so we
                    # do NOT keep it just because its embedder loss was lower (that is
                    # exactly the blindness that let it through). Replace it with the
                    # best fresh attempt — unless that attempt scores BELOW the semantic
                    # gate while the flagged line was above it (guard against replacing a
                    # faithful line on a false-positive flag).
                    old_best = best_by_seg[i]
                    old_sem_ok = old_best is not None and (
                        old_best["sem"] is None or old_best["sem"] >= semantic_threshold)
                    new_sem_ok = new_best["sem"] is None or new_best["sem"] >= semantic_threshold
                    if old_sem_ok and not new_sem_ok:
                        _log(f"  [Segment {i}] heal candidate below semantic gate; keeping prior "
                             f"line for re-audit.")
                        continue
                    best_by_seg[i] = new_best
                    satisfied[i] = new_sat
                    _refresh_seg_dict(i)
                    healed_any = True

                if not healed_any:
                    _log(f"[IsochronyTranslation] Audit heal round {fix_round}: no acceptable "
                         f"replacement produced; stopping heal loop.")
                    break

                new_audits = _audit_once(client, segments, translated_segments,
                                         target_language, log_fn=log_fn,
                                         filter_ids=set(flagged),
                                         check_completeness=completeness_on)
                if not new_audits:
                    _log("[IsochronyTranslation] Re-audit unavailable; keeping prior verdicts, "
                         "stopping heal loop.")
                    break
                audits.update(new_audits)

            # Finalize verdicts from the last successful audit; keep + flag residual.
            residual = []          # segment ids still flagged after all heal rounds
            fixed = []             # first-pass flags that are now meaning-faithful
            for i in range(len(segments)):
                a = audits.get(i)
                still = bool(a and a["needs_fix"])
                translated_segments[i]["audit_reason"] = a["reason"] if a else ""
                translated_segments[i]["audit_passed"] = not still
                if still:
                    translated_segments[i]["audit_verdict"] = "flagged_" + (a["verdict"] if a else "drift")
                    residual.append(i)
                    _log(f"  [Segment {i}] AUDIT FLAG ({a['verdict']}): {a['reason']} "
                         f"| kept best: {translated_segments[i]['text'][:40]}...")
                elif i in first_flags:
                    translated_segments[i]["audit_verdict"] = "fixed"
                    fixed.append(i)
                    _log(f"  [Segment {i}] audit FIXED | now: {translated_segments[i]['text'][:40]}...")
                else:
                    translated_segments[i]["audit_verdict"] = "ok"
            # A residual id NOT in first_flags was newly surfaced by a heal round —
            # report it distinctly rather than letting it dent the "fixed" tally.
            newly = [i for i in residual if i not in first_flags]
            summary = (f"[IsochronyTranslation] Audit done: flagged {n_flag0} → "
                       f"fixed {len(fixed)}, still-flagged {len(residual)}")
            if newly:
                summary += f" ({len(newly)} newly surfaced during healing: {newly})"
            _log(summary + " (residual kept as best candidate).")

    avg_iso = (
        sum(s["isochrony_score"] for s in translated_segments) / len(translated_segments)
        if translated_segments else 0.0
    )
    sem_vals = [s["semantic_score"] for s in translated_segments if s["semantic_score"] is not None]
    avg_sem = sum(sem_vals) / len(sem_vals) if sem_vals else None
    passed = sum(1 for s in translated_segments if s["gates_passed"])
    sem_report = f"{avg_sem:.3f}" if avg_sem is not None else "n/a (gate disabled)"
    # Transcript-level meaning-audit tally (the safety net's bottom line).
    audit_flagged = sum(1 for s in translated_segments if s.get("audit_passed") is False)
    audit_verdicts = {s.get("audit_verdict", "not_audited") for s in translated_segments}
    if "not_audited" in audit_verdicts or "audit_unavailable" in audit_verdicts:
        audit_report = "not run" if "not_audited" in audit_verdicts else "unavailable"
    else:
        audit_report = f"{len(translated_segments) - audit_flagged}/{len(translated_segments)} meaning-faithful"
        if audit_flagged:
            audit_report += f", {audit_flagged} still flagged"
    _log(
        f"[IsochronyTranslation] Done. Avg isochrony: {avg_iso:.3f} | Avg semantic: {sem_report} "
        f"| Both gates passed: {passed}/{len(translated_segments)} | Audit: {audit_report} "
        f"| ruler: {active_ruler()}"
    )
    return translated_segments


# ---------------------------------------------------------------------------
# Sequential fallback (used if batch JSON parsing fails)
# ---------------------------------------------------------------------------

def _translate_sequential(
    client,
    enriched: list,
    internal_lang: str,
    n_candidates: int,
    log_fn: Optional[Callable[[str], None]] = None,
    models: Optional[List[str]] = None,
    served: Optional[List[str]] = None,
) -> dict:
    """Translate one segment at a time. Returns candidates_map dict.

    `served`, if given, collects the model that served each call (forwarded to
    `_call_gemini`) so a batch's "served by" line stays accurate even when it
    falls back to the sequential path.

    Pacing is handled centrally by `_call_gemini`'s per-model throttle, so there
    is no fixed sleep here — the client-side RPM limiter already spaces requests
    to the active model."""
    candidates_map = {}
    lang_cap = internal_lang.capitalize()
    for idx, item in enumerate(enriched):
        i = item["segment_id"]
        budget = item["phoneme_budget"]

        prompt = (
            f"Translate this English dubbing segment into {lang_cap}, preserving the meaning.\n"
            f"English: \"{item['english_text']}\"\n"
            f"Duration: {item['duration_seconds']}s | "
            f"Phoneme target: {budget['min_target']}–{budget['max_target']} (ideal {budget['ideal_target']})\n\n"
            f"Generate exactly {n_candidates} translation candidates."
        )

        seq_msg = f"  [IsochronyTranslation] Sequentially translating segment {i} ({item['english_text'][:30]}...)"
        if log_fn:
            log_fn(seq_msg)
        print(seq_msg)

        try:
            raw = _call_gemini(
                client,
                prompt,
                temperature=0.5,
                response_schema=SegmentCandidatesResponse,
                response_mime_type="application/json",
                log_fn=log_fn,
                models=models,
                served=served,
            )
            candidates = _parse_candidates(raw)
            if not candidates:
                raise ValueError("no parseable candidates in reply")
        except Exception:
            # Last resort: return a single direct translation
            try:
                simple = _call_gemini(
                    client,
                    f"Translate to {lang_cap}: \"{item['english_text']}\". Return only the translation.",
                    temperature=0.3,
                    log_fn=log_fn,
                    models=models,
                    served=served,
                )
                candidates = [simple.strip()] if simple.strip() else [item["english_text"]]
            except Exception:
                candidates = [item["english_text"]]
        candidates_map[i] = candidates
    return candidates_map
