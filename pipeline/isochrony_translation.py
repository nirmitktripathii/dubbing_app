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

# Combined objective used for global-minima tracking when a candidate clears
# neither/one gate. Lower is better:  loss = w_sem*(1 - sim) + w_phon*rel_diff.
# Semantics is weighted higher — a mistranslation that fits the timing is worse
# than a faithful translation that is slightly off timing (TTS can absorb a
# small timing gap; it cannot fix wrong words).
SEMANTIC_WEIGHT = 0.6
PHONEME_WEIGHT = 0.4


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
_GEMINI_FALLBACK = ["gemini-3.1-flash-lite", "gemini-2.5-flash",
                    "gemini-2.0-flash", "gemini-1.5-flash"]
_GEMMA_BULK_DEFAULT = ["gemma-4-31b", "gemma-4-26b"]


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


def _call_gemini(
    client,
    prompt: str,
    temperature: float = 0.4,
    response_schema=None,
    response_mime_type=None,
    log_fn: Optional[Callable[[str], None]] = None,
    models: Optional[List[str]] = None,
    served: Optional[List[str]] = None,
) -> str:
    """Call Gemini with logging, throttling, rate-limit-aware retries, and a
    model fallback chain.

    If `served` is provided, the name of the model that actually returned the
    response is appended to it — so the caller can report which model (Gemma vs
    Gemini) really served a phase, rather than which chain it *intended* to use.

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
    total_tries = 0
    max_total_tries = 10
    max_rate_retries = 4

    while total_tries < max_total_tries and idx < len(chain):
        model_name = chain[idx]

        # Daily-cap guard: skip a model that is already over its RPD for today.
        if rpd is not None and translation_cache.count_today(model_name) >= rpd:
            _emit(f"  [Gemini API] '{model_name}' hit daily cap ({rpd}); advancing model.")
            idx += 1
            rate_retries = 0
            continue

        total_tries += 1
        _throttle(model_name)
        _emit(f"  [Gemini API] Calling '{model_name}' "
              f"(try {total_tries}/{max_total_tries}, prompt len: {len(prompt)})...")
        start_time = time.time()

        try:
            config_args = {"temperature": temperature}
            # Gemma on the Gemini API rejects response_schema / json mime — send
            # a plain text-mode request and let the caller parse the JSON.
            if not _is_gemma(model_name):
                if response_schema is not None:
                    config_args["response_schema"] = response_schema
                if response_mime_type is not None:
                    config_args["response_mime_type"] = response_mime_type

            translation_cache.record_request(model_name)
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(**config_args),
            )
            elapsed = time.time() - start_time
            resp_text = response.text.strip() if response.text else ""
            _emit(f"  [Gemini API] ✓ '{model_name}' in {elapsed:.2f}s. "
                  f"Response len: {len(resp_text)} chars.")
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
            is_transient = any(code in err for code in ["503", "UNAVAILABLE", "overloaded"])

            if is_model_error:
                # Wrong/renamed id → next model immediately (self-heal to Gemini).
                _emit(f"  [Gemini API] '{model_name}' unavailable; advancing to next model.")
                idx += 1
                rate_retries = 0
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

            if (is_rate or is_transient) and idx < len(chain) - 1:
                # Exhausted same-model retries (or transient) → next model.
                _emit(f"  [Gemini API] advancing from '{model_name}' to next model.")
                idx += 1
                rate_retries = 0
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

For EACH segment, generate exactly {n_candidates} NEW improved {lang_cap} candidates that:
1. Fix meaning first: if semantic_similarity is low, rewrite so the translation
   means exactly what the English says.
2. Then fit timing: move the phoneme count toward "ideal_phonemes" and inside
   [min_target, max_target], in the direction given by "phoneme_status".
3. Stay natural and idiomatic — these lines will be spoken aloud.

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
        records.append({
            "text": cand,
            "sem": sem,
            "isochrony": iso,
            "rel_diff": diff["rel_diff"],
            "abs_diff": diff["abs_diff"],
            "target_phonemes": diff["target_phonemes"],
            "ideal_target": diff["ideal_target"],
            "direction": diff["direction"],
            # Repetition penalty folded into the objective so degenerate output
            # never wins on a lucky phoneme count.
            "loss": _combined_loss(sem, diff["rel_diff"]) + rep,
        })
    return records


def _select_best(records: list, semantic_threshold: float, phoneme_tolerance: float):
    """Two-stage selection: semantic gate, then closest phoneme count.

    Returns (best_record, satisfied) where `satisfied` is True iff the chosen
    candidate clears BOTH gates (or clears the phoneme gate when semantics are
    unmeasured this run).
    """
    if not records:
        return None, False

    measured = [r for r in records if r["sem"] is not None]

    if measured:
        gated = [r for r in measured if r["sem"] >= semantic_threshold]
        if gated:
            # Stage B: among meaning-faithful candidates, closest to the phoneme
            # budget wins; combined loss breaks ties.
            gated.sort(key=lambda r: (r["rel_diff"], r["loss"]))
            best = gated[0]
        else:
            # Nothing cleared the meaning gate — keep the MOST faithful candidate
            # (highest similarity), phoneme closeness as tie-break. Better a
            # slightly-off-timing faithful line than an on-time mistranslation.
            measured.sort(key=lambda r: (-r["sem"], r["rel_diff"]))
            best = measured[0]
    else:
        # Semantics unavailable this run: phoneme-fit only (visible degradation).
        records.sort(key=lambda r: (r["rel_diff"], -r["isochrony"]))
        best = records[0]

    sem_ok = best["sem"] is None or best["sem"] >= semantic_threshold
    phon_ok = best["rel_diff"] <= phoneme_tolerance
    return best, (sem_ok and phon_ok)


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
    enriched = []
    for i, seg in enumerate(segments):
        duration = seg["end"] - seg["start"]
        budget = compute_target_budget(seg["text"], internal_lang, source_duration=duration)
        enriched.append({
            "segment_id": i,
            "english_text": seg["text"],
            "duration_seconds": round(duration, 2),
            "phoneme_budget": budget,
        })

    _log(f"[IsochronyTranslation] Translating {len(segments)} segments → {target_language}")
    if semantic_similarity.available():
        _log(f"[IsochronyTranslation] Semantic gate: IndicSBERT active "
             f"(threshold {semantic_threshold}, ruler {active_ruler()}).")
    else:
        _log("[IsochronyTranslation] Semantic gate UNAVAILABLE — selecting on phoneme fit "
             "only (see WARNING above). Install sentence-transformers to enable it.")

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
        # `satisfied` reflects whether the CURRENT best clears both gates.
        cur = best_by_seg[seg_id]
        cur_ok = (cur["sem"] is None or cur["sem"] >= semantic_threshold) and \
                 (cur["rel_diff"] <= phoneme_tolerance)
        satisfied[seg_id] = cur_ok
        return improved

    # --- Step 2: Iteration 0 — batch generate initial candidates for all segs ---
    batch_size = 15

    def _generate_batch(items, prompt_builder, temperature, models, phase="batch"):
        """Run one batched Gemini generation on the given model chain; returns
        {seg_id: [candidate,...]}. `models` selects the phase — Gemma-first for
        the iteration-0 bulk, the Gemini ladder for refinement. `phase` is only a
        label for the "served by" confirmation line."""
        out = {}
        served: List[str] = []
        for bstart in range(0, len(items), batch_size):
            chunk = items[bstart:bstart + batch_size]
            bnum = (bstart // batch_size) + 1
            total = -(-len(items) // batch_size)
            _log(f"  [IsochronyTranslation] Gemini batch {bnum}/{total} ({len(chunk)} segments)...")
            try:
                prompt = prompt_builder(chunk)
                raw = _call_gemini(
                    client, prompt, temperature=temperature,
                    response_schema=BatchTranslationResponse,
                    response_mime_type="application/json",
                    log_fn=log_fn,
                    models=models,
                    served=served,
                )
                # Robust parse: handles the schema shape (Gemini) AND the bare
                # array a schema-less Gemma reply follows from the prompt.
                parsed = _parse_batch(raw)
                if not parsed:
                    raise ValueError("no parseable segments in batch reply")
                out.update(parsed)
            except Exception as e:
                _log(f"    [IsochronyTranslation] Batch {bnum} failed: {e}. Falling back sequentially...")
                # Sequential fallback expects the enriched-item shape.
                seq_items = [
                    it if "phoneme_budget" in it else enriched[it["segment_id"]]
                    for it in chunk
                ]
                out.update(_translate_sequential(
                    client, seq_items, internal_lang, n_candidates,
                    log_fn=log_fn, models=models, served=served,
                ))
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

    avg_iso = (
        sum(s["isochrony_score"] for s in translated_segments) / len(translated_segments)
        if translated_segments else 0.0
    )
    sem_vals = [s["semantic_score"] for s in translated_segments if s["semantic_score"] is not None]
    avg_sem = sum(sem_vals) / len(sem_vals) if sem_vals else None
    passed = sum(1 for s in translated_segments if s["gates_passed"])
    sem_report = f"{avg_sem:.3f}" if avg_sem is not None else "n/a (gate disabled)"
    _log(
        f"[IsochronyTranslation] Done. Avg isochrony: {avg_iso:.3f} | Avg semantic: {sem_report} "
        f"| Both gates passed: {passed}/{len(translated_segments)} | ruler: {active_ruler()}"
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
