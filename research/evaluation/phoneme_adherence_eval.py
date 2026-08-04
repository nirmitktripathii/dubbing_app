"""
evaluation/phoneme_adherence_eval.py
=====================================
Checkpoint-trajectory evaluation for the length-constrained translation fine-tune.

WHAT CHANGED IN THIS REVISION, AND WHY
---------------------------------------
The previous harness produced numbers that could not be trusted, in four separate ways.
Each fix below corresponds to a defect that was found in its output, not to a style
preference.

**1. chrF++ was being swallowed.** `import sacrebleu` sat inside the same `try` as the
scoring call, under a bare `except Exception: return None`. When the pip install failed in
a Kaggle session, every one of 440 calls returned None without a single log line, the run
completed, a report was written, and the fidelity column came out blank — the one column
that would have told us whether the length constraint was being paid for out of meaning.
The import now happens once at module scope and its failure is logged loudly and recorded
in the report header, so an absent metric is visible rather than merely empty.

**2. There were two different slope estimators and the report used the wrong one.**
They are not duplicates; they measure different things, and the distinction is the whole
point of the metric:

  - *population slope* regresses generated length against requested length across
    **different sentences**, whose budgets differ because the sentences differ. A model
    that ignores the budget entirely still scores high on it — longer English produces
    longer Hindi regardless. It measures whether translations are appropriately scaled,
    which is a fluency property, not budget obedience.
  - *probe slope* holds the sentence **fixed** and sweeps only the requested budget across
    0.6-1.4x. The sentence is constant, so the only thing that can move the output length
    is the budget. This is the capability probe.

The old report quoted the population slope while the surrounding prose described the
probe. Both are now computed, named distinctly, and the **probe** is what the report
leads with.

**3. The probe's budgets were derived from the corpus labels**, which are known to be
mislabelled (see `common/phonemes.py`). Natural length is now measured from the reference
text with the canonical counter, so the probe is independent of the corpus labels and
measures capability against the true ruler.

**4. `summarize` used the population variance divisor and returned the upper-middle value
as the "median"** for even-length samples. Small, systematic, and in every median in every
report. Now sample variance (n-1) and a true median.

Two things were also missing rather than broken:

**Semantic fidelity was never measured.** chrF++ needs a reference translation, which
exists here and never exists at dub time. The production-side question — "how much meaning
did compression cost, measured against something available at inference?" — is answered by
scoring each generation against the full-budget candidate using the same embedder the
inference gate uses. That produces a degraded-segment rate per language, which is the
number that converts an architectural worry into evidence.

**The stopping rule lived in prose.** It is now `stopping_verdict()`, computed from the
trajectory and printed in the report, so the decision cannot be re-argued after the fact.

READING THE OUTPUT
------------------
Read the **per-language** table first, then the aggregate. Eleven languages across two
families do not plateau together — Dravidian languages are agglutinative, so their
token-to-phoneme relationship differs and they learn this task on a different schedule.
Every aggregate number hides that. Building the decomposition and then reading the
aggregate column anyway is the mistake this harness was already capable of preventing.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import math
import os
import re
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.languages import LANGUAGES, get_language  # noqa: E402
from common.phonemes import (  # noqa: E402
    PhonemizationError, assert_g2p_available, count_phonemes, ruler_id,
)

logger = logging.getLogger("phoneme_adherence_eval")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

PROMPT_RE = re.compile(r"\[Target Phonemes:\s*(\d+)\]")
LENGTH_SWEEP_FACTORS = [0.6, 0.8, 1.0, 1.2, 1.4]

# --- chrF++ availability, resolved once, loudly -----------------------------------------
# Imported at module scope precisely so its absence is a visible fact about the run rather
# than 440 silent Nones.
try:
    import sacrebleu as _sacrebleu
    CHRF_AVAILABLE = True
    CHRF_UNAVAILABLE_REASON = None
except Exception as _e:  # noqa: BLE001
    _sacrebleu = None
    CHRF_AVAILABLE = False
    CHRF_UNAVAILABLE_REASON = repr(_e)
    logger.error(
        "sacrebleu is NOT importable (%s). chrF++ will be absent from this report. "
        "Fidelity is the axis a length-targeted fine-tune puts at risk, so a run without "
        "it answers a strictly smaller question. Install with: pip install sacrebleu",
        CHRF_UNAVAILABLE_REASON,
    )


# ========================================================================================
# Corpus IO
# ========================================================================================

def read_val(path: str) -> list:
    rows, bad = [], 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
    if bad:
        logger.warning("%d unparseable lines skipped in %s", bad, path)
    return rows


def group_by_language(rows: list) -> dict:
    g = defaultdict(list)
    for r in rows:
        g[r.get("language", "unknown")].append(r)
    return g


def requested_n(row: dict) -> int:
    """The budget the row's own prompt states — i.e. what the model was actually asked for.

    Read from the prompt text first, not from the `n_phonemes` field, because the prompt
    is what the model sees. If a relabelling ever updates one and not the other, this
    reports the number that actually conditioned the generation.
    """
    m = PROMPT_RE.search(row.get("prompt", ""))
    if m:
        return int(m.group(1))
    return int(row.get("n_phonemes") or 0)


def true_phoneme_len(text: str, lang: str) -> Optional[int]:
    try:
        return count_phonemes(text, lang)
    except PhonemizationError as e:
        logger.warning("phonemization failed (%s): %s", lang, e)
        return None


# ========================================================================================
# Statistics
# ========================================================================================

def summarize(xs: list) -> dict:
    """Sample statistics. Sample stdev (n-1) and a true median — the previous version used
    the population divisor and `xs_sorted[n // 2]`, which returns the upper middle value
    for even n."""
    if not xs:
        return {"n": 0}
    n = len(xs)
    return {
        "n": n,
        "mean": statistics.fmean(xs),
        "median": statistics.median(xs),
        "std": statistics.stdev(xs) if n > 1 else 0.0,
    }


def linfit_slope(xs: list, ys: list):
    n = len(xs)
    if n < 2:
        return None, None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx == 0:
        return None, None
    slope = sxy / sxx
    syy = sum((y - my) ** 2 for y in ys)
    r2 = (sxy * sxy) / (sxx * syy) if syy > 0 else None
    return slope, r2


def chrf_pp(hypothesis: str, reference: str) -> Optional[float]:
    """chrF++ against the reference. Returns None only when sacrebleu is genuinely absent
    — which is recorded once, at module import, rather than per call."""
    if not CHRF_AVAILABLE:
        return None
    return float(_sacrebleu.sentence_chrf(hypothesis, [reference], word_order=2).score)


# ========================================================================================
# Semantic scoring — the production-side fidelity check
# ========================================================================================

class SemanticScorer:
    """Cosine similarity with the same embedder the inference gate uses.

    Deliberately the same model as `translation/semantic_gate.py` and
    `training/length_augmentation.py`: a threshold validated on one embedder means nothing
    on another, so an eval that used a different one could not be compared against the
    gate that ships.
    """

    def __init__(self, model_id: Optional[str] = None, device: Optional[str] = None):
        from translation.semantic_gate import DEFAULT_EMBEDDER_MODEL_ID
        self.model_id = model_id or DEFAULT_EMBEDDER_MODEL_ID
        self.device = device
        self._model = None
        self.available = True
        self.reason = None

    def _lazy(self):
        if self._model is None and self.available:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info("Loading semantic embedder %s ...", self.model_id)
                self._model = SentenceTransformer(self.model_id, device=self.device)
            except Exception as e:  # noqa: BLE001
                self.available = False
                self.reason = repr(e)
                logger.error("Semantic scoring DISABLED — embedder failed to load: %s", e)
        return self._model

    def similarity(self, a: str, b: str) -> Optional[float]:
        model = self._lazy()
        if model is None or not a or not b:
            return None
        import numpy as np
        emb = model.encode([a, b], normalize_embeddings=True)
        return float(np.dot(emb[0], emb[1]))


# ========================================================================================
# Model loading and generation
# ========================================================================================

def load_model(adapter_path: Optional[str], base_model_id: str, max_seq_length: int = 512):
    from unsloth import FastLanguageModel
    src = adapter_path if adapter_path else base_model_id
    model, tok = FastLanguageModel.from_pretrained(
        src, max_seq_length=max_seq_length, load_in_4bit=True, dtype=None,
    )
    FastLanguageModel.for_inference(model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


def _clean(text: str) -> str:
    text = text.strip()
    for prefix in ("Translation:", "Output:", "Target:"):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    if len(text) >= 2 and text[0] in "\"'" and text[-1] in "\"'":
        text = text[1:-1].strip()
    return text


def generate_batch(model, tok, prompts: list[str], max_new_tokens: int = 128,
                   temperature: float = 0.3, batch_size: int = 8) -> list[str]:
    """Batched generation.

    The probe needs 5 generations per sentence per language per checkpoint; unbatched that
    dominates the entire session's wall clock and is the reason the probe was previously
    run at 10 sentences per language, where per-language orderings are not trustworthy.
    Left padding is required — decoder-only models continue from the rightmost token, and
    right padding would have the model continue from pad tokens.
    """
    import torch
    outs: list[str] = []
    prev_side = tok.padding_side
    tok.padding_side = "left"
    try:
        for i in range(0, len(prompts), batch_size):
            chunk = prompts[i:i + batch_size]
            texts = [
                tok.apply_chat_template([{"role": "user", "content": p}],
                                        tokenize=False, add_generation_prompt=True)
                for p in chunk
            ]
            enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            with torch.no_grad():
                gen = model.generate(
                    **enc, max_new_tokens=max_new_tokens,
                    do_sample=temperature > 0, temperature=max(temperature, 1e-4),
                    top_p=0.9, pad_token_id=tok.pad_token_id,
                )
            for j in range(len(chunk)):
                new = gen[j][enc["input_ids"].shape[1]:]
                outs.append(_clean(tok.decode(new, skip_special_tokens=True)))
    finally:
        tok.padding_side = prev_side
    return outs


def completion_ce(model, tok, prompt: str, completion: str, max_seq_length: int = 512) -> Optional[float]:
    import torch
    ids = tok.apply_chat_template(
        [{"role": "user", "content": prompt}, {"role": "assistant", "content": completion}],
        return_tensors="pt", add_generation_prompt=False,
    )
    prompt_ids = tok.apply_chat_template(
        [{"role": "user", "content": prompt}], return_tensors="pt", add_generation_prompt=True,
    )
    if ids.shape[1] > max_seq_length:
        return None
    labels = ids.clone()
    labels[:, : prompt_ids.shape[1]] = -100
    ids, labels = ids.to(model.device), labels.to(model.device)
    with torch.no_grad():
        loss = model(input_ids=ids, labels=labels).loss
    return float(loss.item())


# ========================================================================================
# Metric passes
# ========================================================================================

def eval_checkpoint(model, tok, rows_by_lang: dict, adherence_per_lang: int,
                    ce_per_lang: int, do_generation: bool, do_ce: bool,
                    semantic: Optional[SemanticScorer] = None,
                    semantic_threshold: float = 0.80,
                    budget_scale: Optional[dict] = None,
                    batch_size: int = 8,
                    dump_langs: Optional[set] = None, dump_n: int = 0,
                    dump_sink: Optional[list] = None, ckpt_label: str = "") -> dict:
    results = {}
    dump_langs = dump_langs or set()

    for lang, rows in rows_by_lang.items():
        if lang not in LANGUAGES:
            continue
        lang_res = {"language": lang}

        if do_ce:
            ce_vals = []
            for r in rows[:ce_per_lang]:
                c = completion_ce(model, tok, r["prompt"], r.get("completion") or r["target"])
                if c is not None:
                    ce_vals.append(c)
            s = summarize(ce_vals)
            lang_res["ce_mean"] = s.get("mean")
            lang_res["ce_perplexity"] = math.exp(s["mean"]) if s.get("mean") is not None else None
            lang_res["ce_n"] = s["n"]

        if do_generation:
            subset = [r for r in rows[:adherence_per_lang] if requested_n(r) > 0]
            prompts = [_rescaled_prompt(r, lang, budget_scale) for r in subset]
            gens = generate_batch(model, tok, prompts, temperature=0.3, batch_size=batch_size)

            rel_errs, signed_errs, chrfs, sims = [], [], [], []
            req_ns, gen_ns = [], []
            degraded = 0
            dumped = 0
            for r, gen in zip(subset, gens):
                if not gen:
                    continue
                # The budget we hold the model to is always in TRUE phonemes, whatever unit
                # the corpus label happened to be written in.
                N = _true_budget(r, lang)
                if not N:
                    continue
                n_gen = true_phoneme_len(gen, lang)
                if n_gen is None:
                    continue
                ref = r.get("target") or r.get("completion") or ""
                rel_errs.append(abs(n_gen - N) / N)
                signed_errs.append((n_gen - N) / N)
                req_ns.append(float(N))
                gen_ns.append(float(n_gen))

                c = chrf_pp(gen, ref) if ref else None
                if c is not None:
                    chrfs.append(c)

                sim = None
                if semantic is not None:
                    # Anchor = the human reference. At dub time no reference exists and the
                    # anchor is the full-budget candidate instead (see probe below); here the
                    # reference is the stronger anchor and is free.
                    sim = semantic.similarity(ref, gen) if ref else None
                    if sim is not None:
                        sims.append(sim)
                        if sim < semantic_threshold:
                            degraded += 1

                if lang in dump_langs and dump_sink is not None and dumped < dump_n:
                    dump_sink.append({
                        "checkpoint": ckpt_label, "language": lang, "english": r.get("english"),
                        "requested_n": N, "generated_n": n_gen, "generated": gen,
                        "reference": ref, "chrf": c, "semantic_similarity": sim,
                    })
                    dumped += 1

            ra, sa, ca, si = summarize(rel_errs), summarize(signed_errs), summarize(chrfs), summarize(sims)
            lang_res["adherence_rel_mean"] = ra.get("mean")
            lang_res["adherence_rel_median"] = ra.get("median")
            lang_res["adherence_signed_mean"] = sa.get("mean")
            lang_res["adherence_signed_median"] = sa.get("median")
            lang_res["chrf_mean"] = ca.get("mean")
            lang_res["semantic_mean"] = si.get("mean")
            lang_res["semantic_degraded_frac"] = (degraded / si["n"]) if si.get("n") else None
            lang_res["adherence_n"] = ra["n"]
            slope, r2 = linfit_slope(req_ns, gen_ns)
            # Named for what it is. Across DIFFERENT sentences, so it is confounded by
            # sentence length and is a fluency proxy, not a budget-obedience measurement.
            lang_res["length_slope_population"] = slope
            lang_res["length_r2_population"] = r2

        results[lang] = lang_res
    return results


def _true_budget(row: dict, lang: str) -> Optional[int]:
    """The budget in true phonemes.

    If the row was written by the repaired labeller it carries `ruler`, and `n_phonemes`
    is already correct. Otherwise the reference text is re-counted, so a legacy
    character-ruled corpus is still evaluated on the right ruler.
    """
    if str(row.get("ruler", "")).startswith("phonemes:"):
        n = int(row.get("n_phonemes") or 0)
        if n > 0:
            return n
    ref = row.get("target") or row.get("completion") or ""
    return true_phoneme_len(ref, lang) if ref else None


def _rescaled_prompt(row: dict, lang: str, budget_scale: Optional[dict]) -> str:
    """The prompt to send.

    With `--budget_scale_json`, the true-phoneme budget is divided by that language's
    phonemes-per-character constant before being written into the prompt. That converts
    "the budget I want, in phonemes" into "the budget this model was actually taught, in
    characters" — the salvage path for a checkpoint trained on character-ruled labels,
    which needs no retraining. Without the flag the row's own prompt is used unchanged.
    """
    if not budget_scale or lang not in budget_scale:
        return row["prompt"]
    N = _true_budget(row, lang)
    if not N:
        return row["prompt"]
    k = budget_scale[lang]
    asked = max(1, round(N / k)) if k else N
    return PROMPT_RE.sub(f"[Target Phonemes: {asked}]", row["prompt"])


def length_response_probe(model, tok, rows_by_lang: dict, sentences_per_lang: int,
                          semantic: Optional[SemanticScorer] = None,
                          semantic_threshold: float = 0.80,
                          budget_scale: Optional[dict] = None,
                          batch_size: int = 8,
                          points_sink: Optional[list] = None) -> dict:
    """The capability probe: one sentence, five budgets, only the budget varies.

    Natural length is measured from the reference text with the canonical counter rather
    than read from the corpus label, so the probe is unaffected by how the corpus was
    labelled.

    The semantic anchor here is the model's own 1.0x generation — the same anchor the
    inference gate uses, because at dub time no reference exists. That makes the degraded
    rate reported here directly comparable to what the shipped gate will see.

    `points_sink` collects the raw (requested, produced) pairs. Session 02i stored only the
    fitted slope and R², and that turned out to be too little: five languages came back at
    slope ~0.4 with R² ~0.55, and a summary statistic cannot distinguish "flat across the
    whole sweep" from "follows the budget down to 0.8x and then saturates". Those are
    different defects with different fixes.
    """
    out = {}
    for lang, rows in rows_by_lang.items():
        if lang not in LANGUAGES:
            continue
        lang_name = get_language(lang).name

        specs = []          # (sentence_index, factor, prompt)
        naturals = []
        for si, r in enumerate(rows[:sentences_per_lang]):
            eng = r.get("english") or ""
            ref = r.get("target") or r.get("completion") or ""
            if not eng or not ref:
                continue
            natural = true_phoneme_len(ref, lang)
            if not natural:
                continue
            naturals.append(natural)
            k = (budget_scale or {}).get(lang)
            for f in LENGTH_SWEEP_FACTORS:
                want = max(1, round(natural * f))          # what we want, in phonemes
                asked = max(1, round(want / k)) if k else want   # what we write in the prompt
                specs.append((len(naturals) - 1, f, want,
                              f'[Translate to {lang_name}] [Target Phonemes: {asked}] "{eng}"'))

        if not specs:
            out[lang] = {"language": lang, "length_slope_probe": None, "n_points": 0}
            continue

        gens = generate_batch(model, tok, [s[3] for s in specs], temperature=0.3,
                              batch_size=batch_size)

        req_all, gen_all = [], []
        by_sentence: dict[int, dict[float, str]] = defaultdict(dict)
        for (si, f, want, _), gen in zip(specs, gens):
            if not gen:
                continue
            n_gen = true_phoneme_len(gen, lang)
            if n_gen is None:
                continue
            req_all.append(float(want))
            gen_all.append(float(n_gen))
            by_sentence[si][f] = gen
            if points_sink is not None:
                points_sink.append({
                    "language": lang, "sentence_idx": si, "scale": f,
                    "natural_n": naturals[si], "requested_n": want,
                    "produced_n": n_gen,
                })

        slope, r2 = linfit_slope(req_all, gen_all)

        # Semantic cost of compression, anchored the way production anchors it.
        sims, degraded, n_scored = [], 0, 0
        if semantic is not None:
            for si, byf in by_sentence.items():
                anchor = byf.get(1.0)
                if not anchor:
                    continue
                for f, gen in byf.items():
                    if f >= 1.0:
                        continue
                    s = semantic.similarity(anchor, gen)
                    if s is None:
                        continue
                    sims.append(s)
                    n_scored += 1
                    if s < semantic_threshold:
                        degraded += 1

        out[lang] = {
            "language": lang,
            "length_slope_probe": slope,
            "length_r2_probe": r2,
            "n_points": len(req_all),
            "n_sentences": len(naturals),
            "compressed_semantic_mean": statistics.fmean(sims) if sims else None,
            "compressed_degraded_frac": (degraded / n_scored) if n_scored else None,
        }
    return out


# ========================================================================================
# The stopping rule, as code
# ========================================================================================

def stopping_verdict(traj: list[dict], ce_flat_tol: float = 0.005,
                     slope_move_tol: float = 0.01) -> dict:
    """CE-flat + slope still moving => keep spending quota.
       CE-flat + slope flat        => genuine plateau; early-stop loses nothing.

    Written as code rather than kept in prose because the whole point of the rule is that
    it must survive the moment when stopping looks attractive. Uses the probe slope; the
    population slope is not a capability measurement.
    """
    pts = sorted([t for t in traj if t.get("step", -1) >= 0], key=lambda x: x["step"])
    if len(pts) < 2:
        return {"verdict": "INSUFFICIENT_DATA", "reason": "need at least two checkpoints"}

    def last_valid(key):
        vals = [(t["step"], t[key]) for t in pts if t.get(key) is not None]
        return vals[-2:] if len(vals) >= 2 else None

    ce = last_valid("ce_mean")
    sl = last_valid("length_slope_probe") or last_valid("length_slope_population")
    if sl is None:
        return {"verdict": "INSUFFICIENT_DATA", "reason": "no slope measured on >=2 checkpoints"}

    ce_delta = (ce[1][1] - ce[0][1]) if ce else None
    slope_delta = sl[1][1] - sl[0][1]
    ce_flat = ce_delta is None or abs(ce_delta) < ce_flat_tol
    slope_moving = slope_delta > slope_move_tol

    if ce_flat and slope_moving:
        verdict, reason = "CONTINUE", (
            f"CE flat (delta {ce_delta:+.4f}) but probe slope still climbing "
            f"({sl[0][1]:.3f} -> {sl[1][1]:.3f}, delta {slope_delta:+.3f}). Length control "
            f"is still being learned after the loss curve went quiet. Keep spending quota."
        )
    elif ce_flat:
        verdict, reason = "STOP", (
            f"CE flat (delta {ce_delta if ce_delta is None else f'{ce_delta:+.4f}'}) and probe "
            f"slope flat ({sl[0][1]:.3f} -> {sl[1][1]:.3f}, delta {slope_delta:+.3f}). "
            f"Genuine plateau; early-stopping loses nothing measurable."
        )
    else:
        verdict, reason = "CONTINUE", (
            f"CE still moving (delta {ce_delta:+.4f}). Not a plateau."
        )
    return {"verdict": verdict, "reason": reason,
            "ce_delta": ce_delta, "slope_delta": slope_delta,
            "steps_compared": [sl[0][0], sl[1][0]]}


# ========================================================================================
# Driver
# ========================================================================================

def checkpoint_step(path: str) -> int:
    m = re.search(r"checkpoint-(\d+)", os.path.basename(str(path).rstrip("/")))
    return int(m.group(1)) if m else -1


def run(args):
    # An eval that silently used a character fallback would report a mislabelled corpus as
    # correctly labelled. Refuse to start.
    g2p = assert_g2p_available()
    logger.info("G2P ruler: %s", g2p["ruler"])

    budget_scale = None
    if args.budget_scale_json:
        budget_scale = json.loads(Path(args.budget_scale_json).read_text(encoding="utf-8"))
        if "per_language" in budget_scale:  # accept a ruler_audit report directly
            budget_scale = {k: v["ols_k_through_origin"]
                            for k, v in budget_scale["per_language"].items()}
        logger.info("Budget rescale ACTIVE: %s", budget_scale)

    rows = read_val(args.val_jsonl)
    rows_by_lang = group_by_language(rows)
    logger.info("Loaded %d val rows across %d languages", len(rows), len(rows_by_lang))

    semantic = None
    if not args.no_semantic:
        semantic = SemanticScorer(device=args.semantic_device)

    targets = []
    if args.base_baseline:
        targets.append(("base_model", None))
    ckpts = list(args.checkpoints or [])
    if args.checkpoints_glob:
        found = glob.glob(args.checkpoints_glob)
        if not found:
            # The silent-resume bug's twin: a glob that matches nothing produces a run that
            # looks healthy and evaluates nothing.
            raise SystemExit(
                f"--checkpoints_glob {args.checkpoints_glob!r} matched NOTHING. "
                f"Check the directory depth before spending a session on it."
            )
        ckpts += found
    ckpts = sorted(set(ckpts), key=checkpoint_step)
    for c in ckpts:
        targets.append((os.path.basename(str(c).rstrip("/")), c))
    if not targets:
        raise SystemExit("Nothing to evaluate: pass --checkpoints/--checkpoints_glob and/or --base_baseline")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    do_gen = args.mode in ("all", "adherence", "length")
    do_ce = args.mode in ("all", "ce")
    do_probe = args.mode in ("all", "length")

    per_ckpt_rows, traj_rows, lr_rows, sample_rows = [], [], [], []
    probe_point_rows: list[dict] = []
    dump_langs = set(args.dump_samples_langs or [])

    wb = wandb_init({"ruler": g2p["ruler"], "val_jsonl": args.val_jsonl, "mode": args.mode,
                     "checkpoints": [t[0] for t in targets],
                     "budget_scale": budget_scale}, args.wandb_project,
                    args.wandb_entity, args.wandb_run_name,
                    required=args.wandb_required) if args.wandb else None

    for label, adapter in targets:
        logger.info("=== Evaluating %s ===", label)
        model, tok = load_model(adapter, args.base_model_id, args.max_seq_length)
        res = eval_checkpoint(
            model, tok, rows_by_lang, args.adherence_samples_per_lang,
            args.ce_samples_per_lang, do_gen, do_ce,
            semantic=semantic, semantic_threshold=args.semantic_threshold,
            budget_scale=budget_scale, batch_size=args.batch_size,
            dump_langs=dump_langs, dump_n=args.dump_samples_n,
            dump_sink=sample_rows, ckpt_label=label,
        )
        ckpt_points: list[dict] = []
        probe = length_response_probe(
            model, tok, rows_by_lang, args.probe_sentences_per_lang,
            semantic=semantic, semantic_threshold=args.semantic_threshold,
            budget_scale=budget_scale, batch_size=args.batch_size,
            points_sink=ckpt_points,
        ) if do_probe else {}
        probe_point_rows += [{"checkpoint": label, "step": checkpoint_step(adapter)
                              if adapter else -1, **p} for p in ckpt_points]

        def agg(source: dict, key: str):
            vals = [v[key] for v in source.values() if v.get(key) is not None]
            return sum(vals) / len(vals) if vals else None

        step = checkpoint_step(adapter) if adapter else -1
        traj_rows.append({
            "checkpoint": label, "step": step,
            "ce_mean": agg(res, "ce_mean"), "ce_perplexity": agg(res, "ce_perplexity"),
            "adherence_rel_mean": agg(res, "adherence_rel_mean"),
            "adherence_signed_mean": agg(res, "adherence_signed_mean"),
            "chrf_mean": agg(res, "chrf_mean"),
            "semantic_mean": agg(res, "semantic_mean"),
            "semantic_degraded_frac": agg(res, "semantic_degraded_frac"),
            "length_slope_population": agg(res, "length_slope_population"),
            "length_slope_probe": agg(probe, "length_slope_probe") if probe else None,
            "compressed_degraded_frac": agg(probe, "compressed_degraded_frac") if probe else None,
        })
        for lang, v in res.items():
            merged = {**v, **{k: val for k, val in (probe.get(lang) or {}).items()
                              if k != "language"}}
            per_ckpt_rows.append({"checkpoint": label, "step": step, **merged})
        for lang, v in probe.items():
            lr_rows.append({"checkpoint": label, "step": step, **v})

        _write_csv(out_dir / "per_checkpoint_metrics.csv", per_ckpt_rows)
        _write_csv(out_dir / "trajectory_summary.csv", traj_rows)
        if lr_rows:
            _write_csv(out_dir / "length_response.csv", lr_rows)
        if probe_point_rows:
            _write_csv(out_dir / "length_response_points.csv", probe_point_rows)
        # Stream to W&B on the same cadence as the CSVs — this is the only signal
        # visible while the session is still running.
        wandb_log_checkpoint(wb, traj_rows[-1],
                             [r for r in per_ckpt_rows if r["step"] == step])

        del model
        try:
            import gc
            import torch
            gc.collect(); torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    if sample_rows:
        with open(out_dir / "samples.jsonl", "w", encoding="utf-8") as f:
            for row in sample_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "ruler": g2p["ruler"],
        "val_jsonl": args.val_jsonl,
        "n_val_rows": len(rows),
        "mode": args.mode,
        "adherence_samples_per_lang": args.adherence_samples_per_lang,
        "probe_sentences_per_lang": args.probe_sentences_per_lang,
        "chrf_available": CHRF_AVAILABLE,
        "chrf_unavailable_reason": CHRF_UNAVAILABLE_REASON,
        "semantic_available": bool(semantic and semantic.available),
        "semantic_threshold": args.semantic_threshold,
        "budget_scale": budget_scale,
        "checkpoints": [t[0] for t in targets],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_report(out_dir / "eval_report.md", traj_rows, per_ckpt_rows, lr_rows, manifest)
    if wb is not None:
        try:
            import wandb as _wb
            cols = ["checkpoint", "step", "language", "adherence_rel_mean",
                    "adherence_signed_mean", "length_slope_probe",
                    "length_slope_population", "chrf_mean", "semantic_mean",
                    "semantic_degraded_frac", "ce_mean"]
            tbl = _wb.Table(columns=cols)
            for r in per_ckpt_rows:
                tbl.add_data(*[r.get(c) for c in cols])
            wb.log({"per_language": tbl})
            v = stopping_verdict(traj_rows)
            wb.summary["stopping_verdict"] = v["verdict"]
            wb.summary["stopping_reason"] = v.get("reason", "")
            wb.summary["ruler"] = manifest.get("ruler")
            wb.finish()
            logger.info("wandb run CLOSED")
        except Exception as e:  # noqa: BLE001
            logger.error("wandb finalisation failed: %s", e)
    logger.info("Wrote outputs to %s", out_dir)


def wandb_init(manifest: dict, project: str, entity: Optional[str],
               run_name: Optional[str], required: bool = False):
    """Opens the W&B run BEFORE evaluation starts, so metrics can stream.

    The first version of this logged everything in one call at the end of `run()`. That
    makes W&B useless for its actual job here: Kaggle publishes a notebook's log only when
    the session ends, so W&B is the only live signal during a multi-hour run — and a
    channel that reports nothing until the run is over is not a live signal. Metrics are
    now logged after each checkpoint completes, matching the CSV writes.

    `required=True` turns every failure below into a hard exit. Session 02i is why: Kaggle's
    secrets service returned a connection error, the notebook logged a warning and carried
    on, and 12 GPU-hours ran with no observability at all. "The artifacts land on disk
    anyway" is a fair argument for a five-minute local run and a bad one for a session that
    costs a third of the weekly quota. Pass it for anything running on Kaggle.
    """
    def _fail(msg: str):
        if required:
            raise SystemExit(
                f"W&B is required for this run and could not start: {msg}\n"
                f"  This is fatal by design — a multi-hour GPU session with no live channel\n"
                f"  is unobservable until it ends. Fix the credential and re-push, or drop\n"
                f"  --wandb_required if you accept running blind."
            )
        logger.error("%s — continuing without it. The evaluation artifacts on disk are "
                     "unaffected.", msg)
        return None

    try:
        import wandb
    except ImportError:
        return _fail("wandb not installed (pip install wandb)")

    # Check the credential before anything expensive. Without this the run discovers the
    # problem after loading an 8B model, or not at all.
    if required and not (os.environ.get("WANDB_API_KEY") or
                         (Path.home() / ".netrc").exists()):
        return _fail("no WANDB_API_KEY in the environment and no ~/.netrc")

    try:
        run = wandb.init(project=project, entity=entity, name=run_name,
                         job_type="evaluation", config=manifest, reinit=True)
        logger.info("wandb run OPEN: %s", getattr(run, "url", ""))
        return run
    except Exception as e:  # noqa: BLE001
        return _fail(f"wandb.init failed ({e})")


def wandb_log_checkpoint(run, traj_row: dict, lang_rows: list[dict]) -> None:
    """Streams one checkpoint's metrics as soon as it finishes."""
    if run is None:
        return
    step = traj_row["step"] if traj_row["step"] >= 0 else 0
    payload = {f"agg/{k}": v for k, v in traj_row.items()
               if k not in ("checkpoint", "step") and v is not None}
    for r in lang_rows:
        lang = r.get("language")
        for k in ("adherence_rel_mean", "adherence_signed_mean", "length_slope_probe",
                  "length_slope_population", "chrf_mean", "semantic_mean",
                  "semantic_degraded_frac", "ce_mean"):
            if r.get(k) is not None:
                payload[f"{lang}/{k}"] = r[k]
    try:
        run.log(payload, step=step)
        logger.info("wandb: logged %d metrics at step %d", len(payload), step)
    except Exception as e:  # noqa: BLE001
        logger.error("wandb log failed at step %s: %s", step, e)


def log_to_wandb(traj: list[dict], per_ckpt: list[dict], manifest: dict,
                 project: str, entity: Optional[str], run_name: Optional[str]) -> None:
    """Mirrors the report into Weights & Biases.

    What gets logged is deliberately not what was logged last time. CE and perplexity go up
    as *diagnostics*; the panels that matter are `length_slope_probe`, signed adherence, and
    the semantic degraded rate — the metrics that measure the objective. Logging CE
    prominently is how a run gets stopped on the wrong signal, and this project has already
    paid for that once.

    Per-language series are logged individually (`as/length_slope_probe`, …) because the
    aggregate hides that eleven languages across two families do not plateau together.
    """
    try:
        import wandb
    except ImportError:
        logger.warning("wandb not installed — skipping (pip install wandb)")
        return

    try:
        run = wandb.init(project=project, entity=entity, name=run_name,
                         job_type="evaluation", config=manifest, reinit=True)
    except Exception as e:  # noqa: BLE001
        logger.error("wandb.init failed (%s) — continuing without it. The evaluation "
                     "artifacts on disk are unaffected.", e)
        return

    for t in sorted(traj, key=lambda x: x["step"]):
        step = t["step"] if t["step"] >= 0 else 0
        payload = {f"agg/{k}": v for k, v in t.items()
                   if k not in ("checkpoint", "step") and v is not None}
        for r in per_ckpt:
            if r["step"] != t["step"]:
                continue
            lang = r.get("language")
            for k in ("adherence_rel_mean", "adherence_signed_mean", "length_slope_probe",
                      "length_slope_population", "chrf_mean", "semantic_mean",
                      "semantic_degraded_frac", "ce_mean"):
                if r.get(k) is not None:
                    payload[f"{lang}/{k}"] = r[k]
        run.log(payload, step=step)

    cols = ["checkpoint", "step", "language", "adherence_rel_mean", "adherence_signed_mean",
            "length_slope_probe", "length_slope_population", "chrf_mean", "semantic_mean",
            "semantic_degraded_frac", "ce_mean"]
    tbl = wandb.Table(columns=cols)
    for r in per_ckpt:
        tbl.add_data(*[r.get(c) for c in cols])
    run.log({"per_language": tbl})

    v = stopping_verdict(traj)
    run.summary["stopping_verdict"] = v["verdict"]
    run.summary["stopping_reason"] = v.get("reason", "")
    run.summary["ruler"] = manifest.get("ruler")
    run.finish()
    logger.info("wandb run: %s", getattr(run, "url", ""))


def _write_csv(path, rows):
    if not rows:
        return
    cols = list({k for r in rows for k in r.keys()})
    order = ["checkpoint", "step", "language"]
    cols = [c for c in order if c in cols] + sorted(c for c in cols if c not in order)
    # csv.writer rather than manual joining: any value containing a comma — a ruler string,
    # a language name, a failure message — silently shifts every subsequent column when you
    # join by hand, and the file still parses, just wrongly.
    import csv
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow(["" if r.get(c) is None else r.get(c) for c in cols])


def _fmt(x, p=3):
    return "-" if x is None else f"{x:.{p}f}"


def _write_report(path, traj, per_ckpt, lr, manifest):
    L = ["# Phoneme-Adherence Evaluation Report", ""]
    L += [f"- **Ruler:** `{manifest['ruler']}`",
          f"- **Val set:** `{manifest['val_jsonl']}` ({manifest['n_val_rows']} rows)",
          f"- **Generated:** {manifest['generated_utc']}",
          f"- **Adherence samples/lang:** {manifest['adherence_samples_per_lang']}  "
          f"**Probe sentences/lang:** {manifest['probe_sentences_per_lang']}"]
    if manifest.get("budget_scale"):
        L.append(f"- **Budget rescale ACTIVE** (phoneme budget converted to the character "
                 f"budget the model was taught): `{manifest['budget_scale']}`")
    if not manifest["chrf_available"]:
        L.append(f"- **chrF++ UNAVAILABLE** — `{manifest['chrf_unavailable_reason']}`. "
                 f"The fidelity column is absent, not zero.")
    if not manifest["semantic_available"]:
        L.append("- **Semantic scoring UNAVAILABLE** — embedder failed to load. "
                 "Semantic columns are absent, not zero.")
    L += ["", "> Read the per-language table first. Eleven languages across two families do "
          "not plateau together; every aggregate number below hides that.", ""]

    # --- per-language first, by design ---
    L += ["## Per-language", ""]
    by_ckpt = defaultdict(list)
    for r in per_ckpt:
        by_ckpt[(r["step"], r["checkpoint"])].append(r)
    for (step, ckpt) in sorted(by_ckpt):
        L += [f"### {ckpt} (step {step})", "",
              "| Lang | relErr | signed | probe slope | pop slope | chrF++ | semantic | degraded |",
              "|---|---|---|---|---|---|---|---|"]
        for r in sorted(by_ckpt[(step, ckpt)], key=lambda x: x.get("language") or ""):
            L.append(
                f"| {r.get('language')} | {_fmt(r.get('adherence_rel_mean'))} | "
                f"{_fmt(r.get('adherence_signed_mean'))} | {_fmt(r.get('length_slope_probe'))} | "
                f"{_fmt(r.get('length_slope_population'))} | {_fmt(r.get('chrf_mean'), 1)} | "
                f"{_fmt(r.get('semantic_mean'))} | {_fmt(r.get('semantic_degraded_frac'))} |")
        L.append("")

    # --- aggregate ---
    L += ["## Aggregate (read second)", "",
          "| Checkpoint | Step | CE | PPL | relErr | signed | chrF++ | semantic | probe slope | pop slope |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for t in sorted(traj, key=lambda x: x["step"]):
        L.append(f"| {t['checkpoint']} | {t['step']} | {_fmt(t['ce_mean'], 4)} | "
                 f"{_fmt(t['ce_perplexity'])} | {_fmt(t['adherence_rel_mean'])} | "
                 f"{_fmt(t['adherence_signed_mean'])} | {_fmt(t['chrf_mean'], 1)} | "
                 f"{_fmt(t.get('semantic_mean'))} | {_fmt(t.get('length_slope_probe'))} | "
                 f"{_fmt(t['length_slope_population'])} |")

    v = stopping_verdict(traj)
    L += ["", "## Stopping verdict", "", f"**{v['verdict']}** — {v.get('reason', '')}", "",
          "> `length_slope_probe` holds the sentence fixed and sweeps only the budget: it is "
          "the capability measurement. `length_slope_population` regresses across different "
          "sentences and is confounded by sentence length — a model that ignores the budget "
          "entirely still scores high on it. Never select a checkpoint on the population slope.", ""]

    Path(path).write_text("\n".join(L) + "\n", encoding="utf-8")


def _cli():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--val_jsonl", required=True)
    p.add_argument("--checkpoints", nargs="*", default=[])
    p.add_argument("--checkpoints_glob", default=None)
    p.add_argument("--base_baseline", action="store_true")
    p.add_argument("--base_model_id", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--mode", choices=["all", "ce", "adherence", "length"], default="all")
    p.add_argument("--max_seq_length", type=int, default=512)
    p.add_argument("--adherence_samples_per_lang", type=int, default=40)
    p.add_argument("--ce_samples_per_lang", type=int, default=150)
    p.add_argument("--probe_sentences_per_lang", type=int, default=30,
                   help="Sentences per language for the capability probe; each costs 5 "
                        "generations. The previous default of 10 gave 50 points per "
                        "language, at which per-language orderings are not trustworthy.")
    p.add_argument("--batch_size", type=int, default=8,
                   help="Generation batch size. The probe is generation-bound; batching is "
                        "what makes a trustworthy sentence count affordable.")
    p.add_argument("--budget_scale_json", default=None,
                   help="Path to a tools/ruler_audit.py report (or a plain {lang: k} map). "
                        "Converts the true-phoneme budget into the character budget a "
                        "character-ruled checkpoint was actually taught — the salvage path.")
    p.add_argument("--semantic_threshold", type=float, default=0.80)
    p.add_argument("--semantic_device", default=None)
    p.add_argument("--no_semantic", action="store_true",
                   help="Skip semantic scoring (faster; loses the production-side fidelity axis).")
    p.add_argument("--wandb", action="store_true", help="Mirror the report into W&B.")
    p.add_argument("--wandb_project", default="indic-dubbing-v3")
    # The personal namespace `nktthegreat` holds ZERO projects; everything lives
    # under the team entity. Getting this wrong sends metrics to a namespace nobody
    # looks at, and the run appears to have logged nothing.
    p.add_argument("--wandb_entity", default="nktthegreat-soccernet")
    p.add_argument("--wandb_run_name", default=None)
    p.add_argument("--wandb_required", action="store_true",
                   help="Abort before loading a model if W&B cannot start. Pass this for "
                        "any Kaggle session: session 02i burned ~12 GPU-hours with no live "
                        "channel because a secrets-service outage was only a warning.")
    p.add_argument("--dump_samples_langs", nargs="*", default=[])
    p.add_argument("--dump_samples_n", type=int, default=8)
    run(p.parse_args())


if __name__ == "__main__":
    _cli()
