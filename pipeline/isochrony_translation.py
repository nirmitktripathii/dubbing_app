"""
isochrony_translation.py — Stage 3 of the Indic Dubbing Pipeline

Isochrony-Aware Translation: translates English segments into Indic languages
while constraining the output to fit the source audio duration budget.

Core insight: "The real fix is upstream at translation."
Most dubbing pipelines generate a semantically correct translation, then try
to fix timing at the TTS stage. We fix it HERE — before audio is ever generated.

Two-pass approach:
  Pass 1: Generate N candidate translations via Gemini using Chain-of-Thought
           prompting that explicitly instructs the model to count phonemes and
           stay within the duration budget (±15% of source phoneme count).
  Pass 2: Score each candidate using phoneme_counter.isochrony_score().
           The best-scoring candidate (closest to ideal phoneme count) is chosen.

If the best candidate still scores below MINIMUM_ACCEPTABLE_SCORE, a final
Gemini call requests a shorter paraphrase within the remaining budget.

Language support: All 11 Indic languages supported by IndicF5.
"""

import json
import time
import ssl
import re
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
)

# Minimum isochrony score to accept without further refinement
MINIMUM_ACCEPTABLE_SCORE = 0.75
# Number of candidate translations to generate per segment (MBR-style)
N_CANDIDATES = 3


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
    log_fn: Optional[Callable[[str], None]] = None
) -> str:
    """Call Gemini with logging, latency measurement, and fallback models."""
    models_to_try = ["gemini-3.1-flash-lite", "gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
    
    for attempt in range(5):
        # Pick model based on attempt count or fallback requirements
        model_name = models_to_try[min(attempt, len(models_to_try) - 1)]
        msg = f"  [Gemini API] Calling model '{model_name}' (attempt {attempt+1}/5, prompt len: {len(prompt)})..."
        if log_fn:
            log_fn(msg)
        print(msg)
        start_time = time.time()
        
        try:
            config_args = {"temperature": temperature}
            if response_schema is not None:
                config_args["response_schema"] = response_schema
            if response_mime_type is not None:
                config_args["response_mime_type"] = response_mime_type

            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(**config_args),
            )
            elapsed = time.time() - start_time
            resp_text = response.text.strip() if response.text else ""
            success_msg = f"  [Gemini API] ✓ Success in {elapsed:.2f}s. Response len: {len(resp_text)} chars."
            if log_fn:
                log_fn(success_msg)
            print(success_msg)
            return resp_text
        except Exception as e:
            err = str(e)
            elapsed = time.time() - start_time
            err_msg = f"  [Gemini API] ⚠️ Attempt {attempt+1}/5 failed in {elapsed:.2f}s: {err[:120]}"
            if log_fn:
                log_fn(err_msg)
            print(err_msg)
            
            # Check if error is model-not-found (404/INVALID_ARGUMENT for unsupported model)
            is_model_error = "not found" in err.lower() or "not support" in err.lower() or "404" in err
            transient = any(
                code in err
                for code in ["503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "overloaded"]
            )
            
            if (transient or is_model_error) and attempt < 4:
                # If it's a model error, try next model immediately without waiting
                wait = 0 if is_model_error else (attempt + 1) * 3
                if wait > 0:
                    wait_msg = f"  [Gemini API] Retrying in {wait}s..."
                    if log_fn:
                        log_fn(wait_msg)
                    print(wait_msg)
                    time.sleep(wait)
                else:
                    fallback_msg = f"  [Gemini API] Falling back to next model immediately..."
                    if log_fn:
                        log_fn(fallback_msg)
                    print(fallback_msg)
            else:
                raise



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
    per segment, each phoneme-count compliant.
    """
    lang_cap = target_language.capitalize()
    examples_json = json.dumps(segments_with_budgets, ensure_ascii=False, indent=2)

    return f"""You are an expert {lang_cap} dubbing translator with deep knowledge of phonetics.
Your task is to translate English video segments into {lang_cap} for audio dubbing.

CRITICAL DUBBING CONSTRAINT (Isochrony):
Each translated segment must be naturally speakable in the same duration as the original English.
The "phoneme_budget" field tells you exactly how many {lang_cap} syllables/phonemes to target.
- "min_target" and "max_target" define the acceptable range.
- Staying within this range ensures the dubbed audio fits the original video timing.

For EACH segment, generate exactly {n_candidates} candidate translations:
- Candidate 1: Direct translation (prioritize accuracy)
- Candidate 2: Paraphrased translation (prioritize fitting the phoneme budget)
- Candidate 3: Minimal translation (shortest natural phrasing that preserves meaning)

Chain-of-Thought Instructions (apply silently for each segment):
1. Read the English text and its phoneme budget.
2. Draft {n_candidates} translations in {lang_cap}.
3. Mentally count the syllables of each translation.
4. Ensure at least one candidate falls within [min_target, max_target] range.
5. If a direct translation is too long, use synonyms, drop articles, or restructure.

Return ONLY a valid JSON array. Each element must be an object:
{{
  "segment_id": <int>,
  "candidates": ["candidate1 text", "candidate2 text", "candidate3 text"]
}}

Do NOT include markdown fences, explanations, or any text outside the JSON array.

Segments to translate:
{examples_json}"""


def _build_refinement_prompt(
    text: str,
    target_language: str,
    budget: dict,
    current_translation: str,
) -> str:
    """Ask Gemini to shorten or expand a translation to better fit the budget."""
    lang_cap = target_language.capitalize()
    return f"""You are an expert {lang_cap} dubbing translator.

The following {lang_cap} translation for dubbing is outside the phoneme budget:
  Original English: "{text}"
  Current {lang_cap} translation: "{current_translation}"
  Phoneme budget: {budget['min_target']}–{budget['max_target']} syllables
  Ideal: {budget['ideal_target']} syllables

Please rewrite the translation to fit the budget while preserving the core meaning.
Return ONLY the new {lang_cap} translation text, nothing else."""


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
) -> list:
    """
    Translate a list of transcribed segments into an Indic language with
    isochrony (duration) constraints.
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

    # --- Step 1: Compute phoneme budgets for all segments ---
    enriched = []
    for i, seg in enumerate(segments):
        duration = seg["end"] - seg["start"]
        budget = compute_target_budget(seg["text"], internal_lang)
        enriched.append({
            "segment_id": i,
            "english_text": seg["text"],
            "duration_seconds": round(duration, 2),
            "phoneme_budget": budget,
        })

    # --- Step 2: Batch translate in chunks with CoT prompt ---
    start_msg = f"[IsochronyTranslation] Translating {len(segments)} segments → {target_language}"
    if log_fn:
        log_fn(start_msg)
    print(start_msg)
    
    batch_msg = f"[IsochronyTranslation] Generating {n_candidates} candidates per segment in batches..."
    if log_fn:
        log_fn(batch_msg)
    print(batch_msg)
    
    candidates_map = {}
    batch_size = 15
    
    for batch_start in range(0, len(enriched), batch_size):
        batch = enriched[batch_start:batch_start + batch_size]
        batch_num = (batch_start // batch_size) + 1
        total_batches = -(-len(enriched) // batch_size)
        proc_msg = f"  [IsochronyTranslation] Processing batch {batch_num}/{total_batches} ({len(batch)} segments)..."
        if log_fn:
            log_fn(proc_msg)
        print(proc_msg)
        
        try:
            prompt = _build_batch_prompt(batch, internal_lang, n_candidates)
            raw = _call_gemini(
                client, 
                prompt, 
                temperature=0.5,
                response_schema=BatchTranslationResponse,
                response_mime_type="application/json",
                log_fn=log_fn
            )
            
            data = json.loads(raw)
            translations_list = data.get("translations", [])
            for item in translations_list:
                candidates_map[item["segment_id"]] = item["candidates"]
        except Exception as e:
            fail_msg = f"    [IsochronyTranslation] Batch {batch_num} failed: {e}. Translating batch sequentially..."
            if log_fn:
                log_fn(fail_msg)
            print(fail_msg)
            # Fallback to sequential translation just for the failed batch
            batch_map = _translate_sequential(client, batch, internal_lang, n_candidates, log_fn=log_fn)
            candidates_map.update(batch_map)

    # --- Step 3: Score candidates and pick best ---
    translated_segments = []
    for i, seg in enumerate(segments):
        budget = enriched[i]["phoneme_budget"]
        candidates = candidates_map.get(i, [])

        # Deduplicate candidates to avoid redundant calculations
        candidates = list(set(c.strip() for c in candidates if c.strip()))

        if not candidates:
            no_cand_msg = f"  [Segment {i}] No candidates returned. Using empty string."
            if log_fn:
                log_fn(no_cand_msg)
            print(no_cand_msg)
            best = ""
            best_score = 0.0
        else:
            # Score each candidate using MBR-style utility function
            scored = []
            for c in candidates:
                # 1. Base isochrony score (phoneme closeness)
                base_score = isochrony_score(seg["text"], c, internal_lang)
                phonemes = count_indic_phonemes(c, internal_lang)

                # 2. Penality for repetitive patterns (sign of LLM hallucination)
                repetition_penalty = 0.0
                words = c.split()
                if len(words) > 3:
                    # Check for adjacent duplicate words or word pairs
                    dupes = 0
                    for w_idx in range(len(words) - 1):
                        if words[w_idx] == words[w_idx + 1]:
                            dupes += 1
                    repetition_penalty = 0.25 * dupes

                # 3. Penalty for extreme length ratio deviation (guardrail)
                length_ratio = len(c) / max(1, len(seg["text"]))
                length_penalty = 0.0
                if length_ratio < 0.25 or length_ratio > 3.0:
                    length_penalty = 0.4

                # Calculate final utility score
                utility = base_score - repetition_penalty - length_penalty
                scored.append((utility, base_score, phonemes, c))

            # Sort by utility first, then base isochrony score
            scored.sort(reverse=True, key=lambda x: (x[0], x[1]))
            best_utility, best_score, best_phonemes, best = scored[0]

            score_msg = (
                f"  [Segment {i}] Candidates: {len(candidates)} | Best Utility: {best_utility:.3f} "
                f"| Base Score: {best_score:.3f} | phonemes: {best_phonemes}/{budget['ideal_target']:.1f} "
                f"| text: {best[:40]}..."
            )
            if log_fn:
                log_fn(score_msg)
            print(score_msg)


        # --- Step 4: Refinement if score is below threshold ---
        if best_score < min_score and best:
            refine_msg = f"  [Segment {i}] Score {best_score:.3f} < {min_score}. Requesting refinement..."
            if log_fn:
                log_fn(refine_msg)
            print(refine_msg)
            try:
                refine_prompt = _build_refinement_prompt(
                    seg["text"], internal_lang, budget, best
                )
                refined = _call_gemini(client, refine_prompt, temperature=0.3, log_fn=log_fn).strip()
                refined_score = isochrony_score(seg["text"], refined, internal_lang)
                if refined_score > best_score:
                    imp_msg = f"  [Segment {i}] Refinement improved: {best_score:.3f} → {refined_score:.3f}"
                    if log_fn:
                        log_fn(imp_msg)
                    print(imp_msg)
                    best = refined
                    best_score = refined_score
                else:
                    keep_msg = f"  [Segment {i}] Refinement did not improve score. Keeping original."
                    if log_fn:
                        log_fn(keep_msg)
                    print(keep_msg)
            except Exception as e:
                ref_fail_msg = f"  [Segment {i}] Refinement failed: {e}. Keeping best candidate."
                if log_fn:
                    log_fn(ref_fail_msg)
                print(ref_fail_msg)

        translated_segments.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": best,
            "isochrony_score": best_score,
            "duration": round(seg["end"] - seg["start"], 2),
        })

    avg_score = (
        sum(s["isochrony_score"] for s in translated_segments) / len(translated_segments)
        if translated_segments else 0.0
    )
    done_msg = f"[IsochronyTranslation] Done. Average isochrony score: {avg_score:.3f}"
    if log_fn:
        log_fn(done_msg)
    print(done_msg)
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
) -> dict:
    """Translate one segment at a time. Returns candidates_map dict."""
    candidates_map = {}
    lang_cap = internal_lang.capitalize()
    for idx, item in enumerate(enriched):
        i = item["segment_id"]
        budget = item["phoneme_budget"]
        
        # 5.0 seconds delay between sequential fallback requests to stay under 15 RPM
        if idx > 0:
            delay_msg = "  [IsochronyTranslation] Rate-limit safeguard: sleeping 5.0 seconds..."
            if log_fn:
                log_fn(delay_msg)
            print(delay_msg)
            time.sleep(5.0)

        prompt = (
            f"Translate this English dubbing segment into {lang_cap}.\n"
            f"English: \"{item['english_text']}\"\n"
            f"Duration: {item['duration_seconds']}s | "
            f"Phoneme target: {budget['min_target']}–{budget['max_target']} syllables\n\n"
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
                log_fn=log_fn
            )
            data = json.loads(raw)
            candidates = data.get("candidates", [])
            if not isinstance(candidates, list):
                candidates = [str(candidates)]
        except Exception as e:
            # Last resort: return a single direct translation
            try:
                simple = _call_gemini(
                    client,
                    f"Translate to {lang_cap}: \"{item['english_text']}\". Return only the translation.",
                    temperature=0.3,
                    log_fn=log_fn
                )
                candidates = [simple]
            except Exception:
                candidates = [item["english_text"]]
        candidates_map[i] = candidates
    return candidates_map
