"""
evaluation/phoneme_adherence_eval.py
=====================================
Post-/mid-training evaluation for Notebook 02's length-constrained translation LLM.

WHY THIS EXISTS (read before running):
Cross-entropy (train/eval loss on W&B) is a *proxy* for the actual objective. It answers
"how probable is the reference translation token-by-token", NOT "does the model produce a
translation of the requested phoneme length" and NOT "is fidelity preserved". Once eval CE
plateaus (as it did around step ~3200 in this run), CE can no longer tell you whether the
model is still getting better at the thing you actually care about. This script measures
the thing you actually care about, and it does so PER LANGUAGE and ACROSS A CHECKPOINT
TRAJECTORY, so it directly answers the two open research questions:

  Q1. "CE has plateaued — is length conditioning still being learned?"
      -> Run --mode all across the checkpoint series. If mean |phoneme error| and/or the
         length-response SLOPE keep improving while eval CE is flat, conditioning is still
         being learned and it is worth spending more quota. If adherence has ALSO flattened,
         the plateau is real and early-stopping loses nothing. (See LENGTH-RESPONSE PROBE.)

  Q2. "Are some languages still improving while others overfit?"
      -> Every metric here is grouped by the `language` field of val.jsonl. Per-language
         eval CE across checkpoints shows which languages are still dropping vs. plateaued
         vs. rising (a rising per-language eval CE across late checkpoints = that language
         overfitting first). Per-language adherence shows the same for the real objective.

DESIGN NOTES (consistency with the training pipeline — do not "fix" these):
- Phoneme counting uses `translation.duration_predictor.phonemize_text`, the EXACT function
  `dataset_generator.build_example` used to write the [Target Phonemes: N] labels. Using any
  other phonemizer would make adherence numbers incomparable to the training signal.
- The requested N is read from the row's `n_phonemes` field (what the label was built from),
  not re-parsed from the prompt string, to avoid drift.
- Per-language CE replicates training's *completion-only* loss: prompt tokens are masked to
  -100 so only translation tokens contribute — matching TRL's conversational SFT masking, so
  these numbers are directly comparable to the W&B eval_loss (which is the aggregate of this).
- Model loading mirrors the notebook's Gate cell: unsloth FastLanguageModel.from_pretrained
  on the adapter directory (adapter dirs already contain the merged config unsloth needs).

USAGE
-----
Single checkpoint, full eval:
    python -m pipeline_v3.evaluation.phoneme_adherence_eval \
        --val_jsonl data/translation_dataset/val.jsonl \
        --checkpoints checkpoints/translation_llm/checkpoint-3200 \
        --output_dir eval_out --mode all

Checkpoint trajectory (answers Q1/Q2) + base-model baseline (the paper's headline delta):
    python -m pipeline_v3.evaluation.phoneme_adherence_eval \
        --val_jsonl data/translation_dataset/val.jsonl \
        --checkpoints_glob "checkpoints/translation_llm/checkpoint-*" \
        --base_baseline --output_dir eval_out --mode all

Outputs (in --output_dir):
    per_checkpoint_metrics.csv   # one row per (checkpoint, language, metric)
    trajectory_summary.csv       # aggregate + per-language, one row per checkpoint
    length_response.csv          # N-sweep probe results per checkpoint/language
    eval_report.md               # human-readable summary + the Q1/Q2 verdicts
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.languages import LANGUAGES, get_language  # noqa: E402
from translation.duration_predictor import phonemize_text  # noqa: E402  # SAME fn as label build

logger = logging.getLogger("phoneme_adherence_eval")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

PROMPT_RE = re.compile(r"\[Target Phonemes:\s*(\d+)\]")


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------

def read_val(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def group_by_language(rows: list[dict]) -> dict[str, list[dict]]:
    g: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        g[r.get("language", "unknown")].append(r)
    return g


def requested_n(row: dict) -> int:
    """Requested phoneme budget for a row. Prefer the stored label field; fall back to
    parsing the prompt so this still works on rows written by older generator versions."""
    if "n_phonemes" in row and row["n_phonemes"]:
        return int(row["n_phonemes"])
    m = PROMPT_RE.search(row.get("prompt", ""))
    return int(m.group(1)) if m else 0


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------

def load_model(adapter_path: Optional[str], base_model_id: str, max_seq_length: int = 512):
    """adapter_path=None loads the base model with no adapter (baseline). Otherwise loads
    the LoRA adapter directory (unsloth format, as saved by train_translation_llm.py)."""
    from unsloth import FastLanguageModel
    src = adapter_path if adapter_path else base_model_id
    model, tok = FastLanguageModel.from_pretrained(
        src, max_seq_length=max_seq_length, load_in_4bit=True, dtype=None,
    )
    FastLanguageModel.for_inference(model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


def generate(model, tok, prompt: str, max_new_tokens: int = 128, temperature: float = 0.3) -> str:
    import torch
    inputs = tok.apply_chat_template(
        [{"role": "user", "content": prompt}], return_tensors="pt", add_generation_prompt=True
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(
            inputs, max_new_tokens=max_new_tokens,
            do_sample=temperature > 0, temperature=max(temperature, 1e-4),
            top_p=0.9, pad_token_id=tok.pad_token_id,
        )
    text = tok.decode(out[0][inputs.shape[1]:], skip_special_tokens=True).strip()
    # mirror isochrony_translation_v3._clean_generated_text defensive stripping
    for prefix in ("Translation:", "Output:", "Target:"):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    if len(text) >= 2 and text[0] in "\"'" and text[-1] in "\"'":
        text = text[1:-1].strip()
    return text


def completion_ce(model, tok, prompt: str, completion: str, max_seq_length: int = 512) -> Optional[float]:
    """Teacher-forced, completion-only cross-entropy for one (prompt, completion) pair —
    the exact quantity TRL's SFTTrainer averages into eval_loss, computed here per row so it
    can be grouped by language. Returns mean CE over completion tokens (nats)."""
    import torch
    ids = tok.apply_chat_template(
        [{"role": "user", "content": prompt}, {"role": "assistant", "content": completion}],
        return_tensors="pt", add_generation_prompt=False,
    )
    prompt_ids = tok.apply_chat_template(
        [{"role": "user", "content": prompt}], return_tensors="pt", add_generation_prompt=True,
    )
    if ids.shape[1] > max_seq_length:
        return None  # skip rows that would be truncated (they'd bias the number)
    labels = ids.clone()
    labels[:, : prompt_ids.shape[1]] = -100  # mask prompt -> completion-only loss
    ids, labels = ids.to(model.device), labels.to(model.device)
    with torch.no_grad():
        loss = model(input_ids=ids, labels=labels).loss
    return float(loss.item())


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------

def chrf_pp(hypothesis: str, reference: str) -> Optional[float]:
    try:
        import sacrebleu
        return float(sacrebleu.sentence_chrf(hypothesis, [reference], word_order=2).score)
    except Exception:  # noqa: BLE001 - sacrebleu optional; fidelity metric degrades gracefully
        return None


def summarize(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0}
    xs_sorted = sorted(xs)
    n = len(xs)
    mean = sum(xs) / n
    median = xs_sorted[n // 2]
    var = sum((x - mean) ** 2 for x in xs) / n
    return {"n": n, "mean": mean, "median": median, "std": math.sqrt(var)}


def linfit_slope(xs: list[float], ys: list[float]) -> tuple[Optional[float], Optional[float]]:
    """OLS slope + R^2 of ys on xs. For the length-response probe: xs=requested N,
    ys=produced N. Slope→1.0 and R^2→1.0 mean the model obeys the requested length."""
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


# --------------------------------------------------------------------------------------
# Evaluation passes
# --------------------------------------------------------------------------------------

def eval_checkpoint(model, tok, rows_by_lang: dict[str, list[dict]], adherence_per_lang: int,
                    ce_per_lang: int, do_generation: bool, do_ce: bool) -> dict:
    """Returns {lang: {ce_mean, adherence_rel_mean, adherence_signed_mean, chrf_mean, ...}}."""
    results: dict[str, dict] = {}
    for lang, rows in rows_by_lang.items():
        lang_res: dict = {"language": lang}

        # --- per-language completion-only CE (fast; teacher-forced; matches W&B eval_loss) ---
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

        # --- adherence + fidelity (needs generation; slower) ---
        if do_generation:
            rel_errs, signed_errs, chrfs, req_ns, gen_ns = [], [], [], [], []
            for r in rows[:adherence_per_lang]:
                N = requested_n(r)
                if N <= 0:
                    continue
                gen = generate(model, tok, r["prompt"])
                if not gen:
                    continue
                n_gen = len(phonemize_text(gen, lang))
                rel_errs.append(abs(n_gen - N) / N)
                signed_errs.append((n_gen - N) / N)
                req_ns.append(float(N))
                gen_ns.append(float(n_gen))
                c = chrf_pp(gen, r.get("target") or r.get("completion") or "")
                if c is not None:
                    chrfs.append(c)
            ra, sa, ca = summarize(rel_errs), summarize(signed_errs), summarize(chrfs)
            lang_res["adherence_rel_mean"] = ra.get("mean")     # |Ngen-N|/N  (lower=better)
            lang_res["adherence_rel_median"] = ra.get("median")
            lang_res["adherence_signed_mean"] = sa.get("mean")  # sign shows over/undershoot bias
            lang_res["chrf_mean"] = ca.get("mean")              # fidelity (higher=better)
            lang_res["adherence_n"] = ra["n"]
            slope, r2 = linfit_slope(req_ns, gen_ns)
            lang_res["length_slope"] = slope   # produced-vs-requested slope on natural-N spread
            lang_res["length_r2"] = r2
        results[lang] = lang_res
    return results


LENGTH_SWEEP_FACTORS = [0.6, 0.8, 1.0, 1.2, 1.4]


def length_response_probe(model, tok, rows_by_lang: dict[str, list[dict]],
                          sentences_per_lang: int) -> dict:
    """The purest length-control signal, fully decoupled from translation CE: take a fixed
    set of English sentences, ask for the SAME sentence at N = f * natural_N for a sweep of
    f, and measure whether the produced phoneme count tracks the request. A model that truly
    learned length conditioning yields a monotone response with slope≈1; a model that ignores
    N yields a flat response (slope≈0). Rising slope across checkpoints while CE is flat is
    the decisive evidence that length conditioning is still being learned (Q1)."""
    out: dict[str, dict] = {}
    for lang, rows in rows_by_lang.items():
        req_all, gen_all = [], []
        for r in rows[:sentences_per_lang]:
            natural_N = requested_n(r)
            if natural_N <= 0:
                continue
            eng = r.get("english") or ""
            lang_name = get_language(lang).name if lang in LANGUAGES else lang
            for f in LENGTH_SWEEP_FACTORS:
                target = max(1, round(natural_N * f))
                prompt = f'[Translate to {lang_name}] [Target Phonemes: {target}] "{eng}"'
                gen = generate(model, tok, prompt)
                if not gen:
                    continue
                req_all.append(float(target))
                gen_all.append(float(len(phonemize_text(gen, lang))))
        slope, r2 = linfit_slope(req_all, gen_all)
        out[lang] = {"language": lang, "length_slope": slope, "length_r2": r2,
                     "n_points": len(req_all)}
    return out


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------

def checkpoint_step(path: str) -> int:
    m = re.search(r"checkpoint-(\d+)", os.path.basename(path.rstrip("/")))
    return int(m.group(1)) if m else -1


def run(args):
    rows = read_val(args.val_jsonl)
    rows_by_lang = group_by_language(rows)
    logger.info("Loaded %d val rows across %d languages", len(rows), len(rows_by_lang))

    targets: list[tuple[str, Optional[str]]] = []  # (label, adapter_path_or_None)
    if args.base_baseline:
        targets.append(("base_model", None))
    ckpts = list(args.checkpoints or [])
    if args.checkpoints_glob:
        ckpts += glob.glob(args.checkpoints_glob)
    ckpts = sorted(set(ckpts), key=checkpoint_step)
    for c in ckpts:
        targets.append((os.path.basename(c.rstrip("/")), c))
    if not targets:
        raise SystemExit("Nothing to evaluate: pass --checkpoints/--checkpoints_glob and/or --base_baseline")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    do_gen = args.mode in ("all", "adherence", "length")
    do_ce = args.mode in ("all", "ce")
    do_probe = args.mode in ("all", "length")

    per_ckpt_rows, traj_rows, lr_rows = [], [], []
    for label, adapter in targets:
        logger.info("=== Evaluating %s ===", label)
        model, tok = load_model(adapter, args.base_model_id, args.max_seq_length)
        res = eval_checkpoint(model, tok, rows_by_lang, args.adherence_samples_per_lang,
                              args.ce_samples_per_lang, do_gen, do_ce)
        probe = length_response_probe(model, tok, rows_by_lang, args.probe_sentences_per_lang) if do_probe else {}

        # aggregate across languages (sample-weighted where counts exist)
        def agg(key):
            vals = [v[key] for v in res.values() if v.get(key) is not None]
            return sum(vals) / len(vals) if vals else None
        step = checkpoint_step(adapter) if adapter else -1
        traj_rows.append({
            "checkpoint": label, "step": step,
            "ce_mean": agg("ce_mean"), "ce_perplexity": agg("ce_perplexity"),
            "adherence_rel_mean": agg("adherence_rel_mean"),
            "adherence_signed_mean": agg("adherence_signed_mean"),
            "chrf_mean": agg("chrf_mean"), "length_slope": agg("length_slope"),
        })
        for lang, v in res.items():
            per_ckpt_rows.append({"checkpoint": label, "step": step, **v})
        for lang, v in probe.items():
            lr_rows.append({"checkpoint": label, "step": step, **v})

        del model
        try:
            import torch, gc  # noqa
            gc.collect(); torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    _write_csv(out_dir / "per_checkpoint_metrics.csv", per_ckpt_rows)
    _write_csv(out_dir / "trajectory_summary.csv", traj_rows)
    if lr_rows:
        _write_csv(out_dir / "length_response.csv", lr_rows)
    _write_report(out_dir / "eval_report.md", traj_rows, per_ckpt_rows, lr_rows)
    logger.info("Wrote outputs to %s", out_dir)


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    cols = list({k for r in rows for k in r.keys()})
    order = ["checkpoint", "step", "language"]
    cols = [c for c in order if c in cols] + [c for c in cols if c not in order]
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join("" if r.get(c) is None else str(r.get(c)) for c in cols) + "\n")


def _write_report(path: Path, traj: list[dict], per_ckpt: list[dict], lr: list[dict]):
    lines = ["# Phoneme-Adherence Evaluation Report", ""]
    lines.append("## Aggregate trajectory (the Q1 answer is in whether adherence/slope keep")
    lines.append("moving after CE flattens)\n")
    lines.append("| Checkpoint | Step | CE | Perplexity | |ΔN|/N | signed | chrF++ | length_slope |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for t in sorted(traj, key=lambda x: x["step"]):
        def fmt(x, p=4):
            return "—" if x is None else f"{x:.{p}f}"
        lines.append(f"| {t['checkpoint']} | {t['step']} | {fmt(t['ce_mean'])} | "
                     f"{fmt(t['ce_perplexity'],3)} | {fmt(t['adherence_rel_mean'],3)} | "
                     f"{fmt(t['adherence_signed_mean'],3)} | {fmt(t['chrf_mean'],1)} | "
                     f"{fmt(t['length_slope'],3)} |")
    lines += ["", "## How to read this", "",
              "- **CE flat + |ΔN|/N still falling OR length_slope still rising toward 1.0** ⇒ "
              "length conditioning is STILL being learned; more training quota is justified.",
              "- **CE flat + adherence flat** ⇒ genuine plateau; early-stop at the best-eval "
              "checkpoint loses nothing measurable.",
              "- **signed mean** drifting negative ⇒ systematic UNDER-generation (too short); "
              "positive ⇒ over-generation. Guides whether the inference 65/85/100% budgets or "
              "an added >100% candidate are warranted.",
              "- **Per-language** rows in per_checkpoint_metrics.csv answer Q2: sort by step "
              "within a language and watch for a language whose CE rises across late "
              "checkpoints (overfitting onset for THAT language) while others keep falling.",
              "- **base_model row** is the untrained baseline; every trained checkpoint's "
              "delta vs. it is the paper's headline 'what fine-tuning bought' number."]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _cli():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--val_jsonl", required=True)
    p.add_argument("--checkpoints", nargs="*", default=[])
    p.add_argument("--checkpoints_glob", default=None)
    p.add_argument("--base_baseline", action="store_true", help="also eval the base model (no adapter)")
    p.add_argument("--base_model_id", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--mode", choices=["all", "ce", "adherence", "length"], default="all")
    p.add_argument("--max_seq_length", type=int, default=512)
    p.add_argument("--adherence_samples_per_lang", type=int, default=40)
    p.add_argument("--ce_samples_per_lang", type=int, default=150)
    p.add_argument("--probe_sentences_per_lang", type=int, default=10)
    run(p.parse_args())


if __name__ == "__main__":
    _cli()
