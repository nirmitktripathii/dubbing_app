"""
semantic_similarity.py — Cross-lingual semantic scoring for isochrony candidate selection.

v2.5 addition
-------------
Isochrony alone is not enough: a translation can hit the phoneme budget perfectly and still
say the wrong thing. This module scores how well an Indic candidate preserves the MEANING of
the English source, so the isochrony loop can gate on semantics FIRST (keep only candidates
that actually mean the same thing) and only then choose among them by phoneme fit.

Model
-----
`l3cube-pune/indic-sentence-similarity-sbert` (IndicSBERT) — a sentence-transformer trained
for cross-lingual similarity across the major Indic languages plus English. It embeds the
English source and the Indic candidate into a shared space; cosine similarity of the two
embeddings is the semantic score in [0, 1] (higher = closer meaning).

WHY NOT plain MiniLM: multilingual-MiniLM collapses most non-Devanagari Indic scripts into
near-identical embeddings (validated: negative-pair p95 ≈ 0.99 for 8 of 11 languages), so it
cannot tell a good Tamil/Telugu/Malayalam translation from a bad one. IndicSBERT is trained
on Indic data specifically. Override with env var DUBBING_SBERT_MODEL if needed.

DEGRADE VISIBLY, NEVER SILENTLY
-------------------------------
Like the phoneme counter, this sits in a live pipeline. If sentence-transformers or the model
weights are unavailable (offline, gated download, the huggingface_hub segfault seen on some
machines), `score()` returns None rather than a fake 0.0 or 1.0 — a None tells the caller
"semantics could not be measured, fall back to phoneme-only selection" instead of silently
scoring every candidate identically. The first failure logs one loud WARNING.
"""

from __future__ import annotations

import logging
import os
import threading

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "l3cube-pune/indic-sentence-similarity-sbert"


def _model_name() -> str:
    return os.environ.get("DUBBING_SBERT_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL


# Lazy, process-wide singleton. Loading the model is expensive (weights + torch), so we do it
# once, on first use, behind a lock. `_load_failed` latches so we don't retry a broken load
# (and re-emit the warning) on every segment.
_model = None
_load_failed = False
_load_lock = threading.Lock()
_warned = False


def _warn_once(msg: str, exc: Exception = None):
    global _warned
    if not _warned:
        _warned = True
        if exc is not None:
            logger.warning("%s: %s", msg, exc)
        else:
            logger.warning("%s", msg)


def _get_model():
    """Return the loaded SentenceTransformer, or None if it cannot be loaded (latched)."""
    global _model, _load_failed
    if _model is not None:
        return _model
    if _load_failed:
        return None
    with _load_lock:
        if _model is not None:
            return _model
        if _load_failed:
            return None
        name = _model_name()
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as e:  # noqa: BLE001
            _load_failed = True
            _warn_once(
                "SEMANTIC GATING DISABLED: `sentence-transformers` is not importable, so "
                "cross-lingual similarity cannot be measured. Candidate selection will fall "
                "back to phoneme-fit only. Install with: pip install sentence-transformers",
                e,
            )
            return None
        try:
            _model = SentenceTransformer(name)
            logger.info("Loaded IndicSBERT semantic model: %s", name)
            return _model
        except Exception as e:  # noqa: BLE001
            _load_failed = True
            _warn_once(
                f"SEMANTIC GATING DISABLED: could not load semantic model {name!r} "
                f"(offline, gated, or download failure). Candidate selection will fall back "
                f"to phoneme-fit only",
                e,
            )
            return None


def available() -> bool:
    """True if the semantic model is loaded and usable. Triggers the lazy load."""
    return _get_model() is not None


def score(source_text: str, target_text: str):
    """Cosine semantic similarity between an English source and an Indic candidate.

    Returns a float in [0.0, 1.0] (higher = closer meaning), or None if the semantic model
    is unavailable — the caller must treat None as "unmeasured", not as zero similarity.
    """
    if not source_text or not source_text.strip() or not target_text or not target_text.strip():
        return None
    model = _get_model()
    if model is None:
        return None
    try:
        from sentence_transformers import util
        embs = model.encode(
            [source_text, target_text],
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        sim = util.cos_sim(embs[0], embs[1]).item()
        # Cosine of normalised embeddings is in [-1, 1]; clamp to [0, 1] for a clean gate.
        return round(max(0.0, min(1.0, sim)), 4)
    except Exception as e:  # noqa: BLE001
        _warn_once("Semantic scoring failed on a segment; treating as unmeasured", e)
        return None


def score_many(source_text: str, candidates):
    """Semantic similarity of one English source against many Indic candidates.

    One batched encode for the whole set (much faster than looping). Returns a list of
    floats aligned to `candidates`, or None if the model is unavailable (so the caller can
    skip the semantic gate entirely for this segment).
    """
    cands = list(candidates)
    if not source_text or not source_text.strip() or not cands:
        return None
    model = _get_model()
    if model is None:
        return None
    try:
        from sentence_transformers import util
        texts = [source_text] + [c if (c and c.strip()) else " " for c in cands]
        embs = model.encode(
            texts,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        sims = util.cos_sim(embs[0:1], embs[1:])[0]
        return [round(max(0.0, min(1.0, float(s))), 4) for s in sims]
    except Exception as e:  # noqa: BLE001
        _warn_once("Batched semantic scoring failed; treating as unmeasured", e)
        return None
