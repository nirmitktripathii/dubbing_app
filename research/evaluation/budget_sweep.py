"""
evaluation/budget_sweep.py
===========================
Step 1 of the Phase-02 repair: find out **why** five languages do not follow the phoneme
budget, and come away with usable training data either way.

THE QUESTION
------------
At checkpoint 3801, six languages follow the budget and five do not. "Slope 0.35" is
compatible with at least four different defects that need four different fixes (see
`evaluation/response_diagnosis.py`). You cannot tell them apart from a slope. You can tell
them apart instantly from the SHAPE of the response curve, which nobody has ever looked at
because only the fitted slope was ever saved.

So: hold the sentence fixed, sweep the budget across a deliberately wide range, and keep
every raw point.

WHY THE SWEEP IS WIDER THAN BEFORE
-----------------------------------
The old probe swept 0.6x to 1.4x. Two problems. First, saturation outside that band is
invisible — a model that tracks to 0.8x and then refuses looks identical to one that
follows everywhere. Second, and worse, the old estimator was not zero-referenced: pooling
raw lengths across sentences let sentence-length variance leak into the slope, so a model
that ignores the budget entirely scored 0.60-0.83 depending on the language rather than 0.
This module reports the within-sentence normalised slope, where 0 means ignored.

WHY IT ALSO HARVESTS
--------------------
A GPU session that produces only a diagnosis is a session that has to be followed by
another session before anything improves. This one generates thousands of translations at
off-natural budgets, and every one of them is exactly measurable — the phoneme counter is
exact and the semantic gate is the same one production uses. So any generation that
verifiably landed near its budget AND kept its meaning is a valid training row for the
corrective fine-tune, harvested for free from work that had to happen anyway.

That is the general rule this file exists to follow:

    Every GPU run must leave behind an artifact the next run consumes.
    A run whose only output is knowledge is a run you will have to pay for twice.

ORDER OF WORK, AND THE WALL CLOCK
----------------------------------
Decision-critical first, opportunistic second, hard stop always. The sweep runs first and
writes after every language, so an interrupted session still answers the question. The
harvest runs only with time left over. `--time_budget_s` is enforced between languages, so
a session cannot overrun its quota allocation.

USAGE
-----
    python -m evaluation.budget_sweep \
        --val_jsonl data/val.phonemes.jsonl \
        --checkpoint checkpoints/checkpoint-3801 \
        --output_dir sweep_out --time_budget_s 5400 --wandb --wandb_required
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.languages import LANGUAGES, get_language  # noqa: E402
from common.phonemes import assert_g2p_available, count_phonemes, ruler_id  # noqa: E402
from evaluation.response_diagnosis import diagnose, format_report  # noqa: E402

logger = logging.getLogger("budget_sweep")

# Wide on purpose. 0.4x and 2.0x are outside anything the product would ask for, and that
# is the point: the extremes are where saturation shows up, and a curve is only readable
# if it has been driven past its limits.
DEFAULT_SCALES = [0.4, 0.55, 0.7, 0.85, 1.0, 1.15, 1.3, 1.6, 2.0]

# What counts as a harvested training row. Both gates, or neither.
HARVEST_REL_TOL = 0.10          # landed within 10% of the requested budget
HARVEST_SEM_MIN = 0.80          # and did not get there by deleting the meaning
HARVEST_SCALES = [0.6, 0.75, 0.9]   # compression: what dubbing actually asks for
HARVEST_SAMPLES = 4
HARVEST_TEMPERATURE = 0.9       # higher than the sweep — we want spread to select from

PROMPT = '[Translate to {language}] [Target Phonemes: {n}] "{english}"'


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    cols, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                cols.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow(["" if r.get(c) is None else r.get(c) for c in cols])


def load_rows(path: str) -> dict[str, list[dict]]:
    by_lang: dict[str, list[dict]] = defaultdict(list)
    rulers = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("language") in LANGUAGES:
                by_lang[r["language"]].append(r)
            rulers.add(r.get("ruler", "MISSING"))
    if not any(str(x).startswith("phonemes:") for x in rulers):
        raise SystemExit(
            f"{path} is not phoneme-ruled (rulers={rulers}). Measuring a phoneme budget "
            f"against character labels is the defect this whole phase exists to undo.")
    logger.info("%d languages | rulers: %s", len(by_lang), rulers)
    return by_lang


# ========================================================================================

def sweep_language(model, tok, lang: str, rows: list[dict], scales: list[float],
                   n_sentences: int, generate_batch, semantic, batch_size: int,
                   point_sink: list, cand_sink: list) -> None:
    """One language, every sentence at every scale. Appends raw points and candidates."""
    lang_name = get_language(lang).name
    specs, naturals, metas = [], [], []
    for r in rows[:n_sentences]:
        eng = r.get("english") or ""
        ref = r.get("target") or r.get("completion") or ""
        if not eng or not ref:
            continue
        try:
            natural = count_phonemes(ref, lang)
        except Exception:  # noqa: BLE001
            continue
        if not natural:
            continue
        si = len(naturals)
        naturals.append(natural)
        metas.append({"english": eng, "reference": ref})
        for s in scales:
            want = max(1, round(natural * s))
            specs.append((si, s, want, PROMPT.format(language=lang_name, n=want, english=eng)))

    if not specs:
        logger.warning("%s: no usable sentences", lang)
        return

    gens = generate_batch(model, tok, [s[3] for s in specs],
                          temperature=0.3, batch_size=batch_size)

    # The semantic anchor is the model's own 1.0x generation, because at dub time there is
    # no reference to compare against — the shipped gate has to work without one.
    anchor: dict[int, str] = {}
    for (si, s, _, _), gen in zip(specs, gens):
        if s == 1.0 and gen:
            anchor[si] = gen

    for (si, s, want, _), gen in zip(specs, gens):
        if not gen:
            continue
        try:
            produced = count_phonemes(gen, lang)
        except Exception:  # noqa: BLE001
            continue
        if not produced:
            continue
        sem_ref = semantic.similarity(metas[si]["reference"], gen) if semantic else None
        sem_anchor = (semantic.similarity(anchor[si], gen)
                      if semantic and si in anchor else None)
        point_sink.append({
            "language": lang, "sentence_idx": si, "scale": s,
            "natural_n": naturals[si], "requested_n": want, "produced_n": produced,
            "semantic": sem_ref, "semantic_vs_own_1x": sem_anchor,
        })
        rel = abs(produced - want) / want
        cand_sink.append({
            "language": lang, "english": metas[si]["english"],
            "reference": metas[si]["reference"], "scale": s, "natural_n": naturals[si],
            "requested_n": want, "produced_n": produced, "rel_err": round(rel, 4),
            "semantic": sem_ref, "generated": gen, "source": "sweep",
            "harvestable": bool(rel <= HARVEST_REL_TOL
                                and (sem_ref is None or sem_ref >= HARVEST_SEM_MIN)),
        })


def harvest_language(model, tok, lang: str, rows: list[dict], n_sentences: int,
                     generate_batch, semantic, batch_size: int, cand_sink: list) -> int:
    """Best-of-n at compression budgets, kept only when both gates pass.

    This is rejection sampling with an EXACT verifier. There is no reward model and no
    judge: the phoneme counter says precisely whether the candidate hit its budget, and the
    semantic gate says whether it cheated to get there. That combination is rare enough to
    be worth exploiting — it turns "the model can sometimes do this" into supervised data
    for "the model should always do this".
    """
    lang_name = get_language(lang).name
    specs, meta = [], []
    for r in rows[:n_sentences]:
        eng = r.get("english") or ""
        ref = r.get("target") or r.get("completion") or ""
        if not eng or not ref:
            continue
        try:
            natural = count_phonemes(ref, lang)
        except Exception:  # noqa: BLE001
            continue
        if not natural:
            continue
        for s in HARVEST_SCALES:
            want = max(1, round(natural * s))
            for _ in range(HARVEST_SAMPLES):
                specs.append(PROMPT.format(language=lang_name, n=want, english=eng))
                meta.append((eng, ref, natural, s, want))

    if not specs:
        return 0
    gens = generate_batch(model, tok, specs, temperature=HARVEST_TEMPERATURE,
                          batch_size=batch_size)

    # Keep the single best candidate per (sentence, budget) rather than every passing one:
    # four near-duplicates of the same sentence would let one easy example dominate.
    best: dict[tuple, dict] = {}
    for (eng, ref, natural, s, want), gen in zip(meta, gens):
        if not gen:
            continue
        try:
            produced = count_phonemes(gen, lang)
        except Exception:  # noqa: BLE001
            continue
        if not produced:
            continue
        rel = abs(produced - want) / want
        if rel > HARVEST_REL_TOL:
            continue
        sem = semantic.similarity(ref, gen) if semantic else None
        if sem is not None and sem < HARVEST_SEM_MIN:
            continue
        key = (eng, s)
        prev = best.get(key)
        if prev is None or rel < prev["rel_err"]:
            best[key] = {
                "language": lang, "english": eng, "reference": ref, "scale": s,
                "natural_n": natural, "requested_n": want, "produced_n": produced,
                "rel_err": round(rel, 4), "semantic": sem, "generated": gen,
                "source": "harvest", "harvestable": True,
            }
    cand_sink.extend(best.values())
    return len(best)


# ========================================================================================

def run(args) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    t0 = time.time()
    g2p = assert_g2p_available()
    logger.info("G2P preflight PASSED — ruler=%s", ruler_id())

    # Imported here, not at module scope: this file must stay importable on a machine with
    # no torch so the diagnosis can be re-run offline against a saved sweep.
    from evaluation.phoneme_adherence_eval import (  # noqa: PLC0415
        SemanticScorer, generate_batch, load_model, wandb_init,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scales = [float(x) for x in args.scales] if args.scales else DEFAULT_SCALES
    by_lang = load_rows(args.val_jsonl)
    langs = [lg for lg in (args.languages or sorted(by_lang)) if lg in by_lang]

    wb = wandb_init(
        {"job": "budget_sweep", "ruler": g2p["ruler"], "scales": scales,
         "sentences_per_lang": args.sentences_per_lang, "checkpoint": args.checkpoint,
         "time_budget_s": args.time_budget_s},
        args.wandb_project, args.wandb_entity,
        args.wandb_run_name or "02j-budget-sweep",
        required=args.wandb_required) if args.wandb else None

    semantic = None if args.no_semantic else SemanticScorer()
    model, tok = load_model(args.checkpoint, args.base_model_id, args.max_seq_length)

    points: list[dict] = []
    cands: list[dict] = []
    done: list[str] = []

    def flush():
        _write_csv(out_dir / "sweep_points.csv", points)
        with open(out_dir / "candidates.jsonl", "w", encoding="utf-8") as f:
            for c in cands:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # ---- Phase 1: the diagnosis. Decision-critical, so it goes first. -----------------
    for lg in langs:
        if args.time_budget_s and time.time() - t0 > args.time_budget_s:
            logger.warning("time budget reached before %s — stopping the sweep with "
                           "%d/%d languages done", lg, len(done), len(langs))
            break
        t = time.time()
        sweep_language(model, tok, lg, by_lang[lg], scales, args.sentences_per_lang,
                       generate_batch, semantic, args.batch_size, points, cands)
        done.append(lg)
        flush()

        d = diagnose([p for p in points if p["language"] == lg])["per_language"].get(lg, {})
        logger.info("%s: %s  slope=%.3f r2=%.3f  (%.0fs)", lg,
                    d.get("diagnosis", "?"), d.get("slope") or float("nan"),
                    d.get("r2") or float("nan"), time.time() - t)
        if wb is not None:
            try:
                wb.log({f"{lg}/slope_normalized": d.get("slope"),
                        f"{lg}/r2_normalized": d.get("r2"),
                        f"{lg}/slope_compress": d.get("slope_compress"),
                        f"{lg}/slope_expand": d.get("slope_expand"),
                        f"{lg}/within_sentence_cv": d.get("within_sentence_cv"),
                        f"{lg}/semantic_far_compress": d.get("semantic_far_compress"),
                        "languages_done": len(done)})
            except Exception as e:  # noqa: BLE001
                logger.error("wandb log failed for %s: %s", lg, e)

    result = diagnose(points)
    (out_dir / "DIAGNOSIS.json").write_text(
        json.dumps({"generated_utc": datetime.now(timezone.utc).isoformat(),
                    "ruler": g2p["ruler"], "scales": scales,
                    "languages_swept": done, **result}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    print(format_report(result))

    # ---- Phase 2: harvest, only with time to spare. -----------------------------------
    harvested = {}
    if args.harvest:
        weak = [lg for lg in done
                if result["per_language"].get(lg, {}).get("diagnosis") != "OBEYS"]
        for lg in weak:
            left = args.time_budget_s - (time.time() - t0) if args.time_budget_s else 1e9
            if left < args.harvest_reserve_s:
                logger.warning("stopping harvest before %s — %.0fs left, need %.0fs",
                               lg, left, args.harvest_reserve_s)
                break
            n = harvest_language(model, tok, lg, by_lang[lg], args.harvest_sentences,
                                 generate_batch, semantic, args.batch_size, cands)
            harvested[lg] = n
            flush()
            logger.info("%s: harvested %d verified elastic rows", lg, n)
            if wb is not None:
                try:
                    wb.log({f"{lg}/harvested_rows": n})
                except Exception:  # noqa: BLE001
                    pass

    flush()
    n_ok = sum(1 for c in cands if c.get("harvestable"))
    summary = {
        "languages_swept": done, "n_points": len(points),
        "n_candidates": len(cands), "n_harvestable": n_ok,
        "harvested_per_lang": harvested,
        "elapsed_s": round(time.time() - t0),
        "diagnosis_counts": result["counts"],
        "dominant_failure": result["dominant_failure"],
    }
    (out_dir / "SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n" + json.dumps(summary, indent=2))

    if wb is not None:
        try:
            wb.summary.update(summary)
            wb.finish()
        except Exception as e:  # noqa: BLE001
            logger.error("wandb finalisation failed: %s", e)
    logger.info("Wrote outputs to %s", out_dir)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--val_jsonl", required=True)
    p.add_argument("--checkpoint", default=None, help="Adapter dir; omit for the base model.")
    p.add_argument("--base_model_id", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_seq_length", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=24)
    p.add_argument("--languages", nargs="*", default=None)
    p.add_argument("--scales", nargs="*", default=None)
    p.add_argument("--sentences_per_lang", type=int, default=40)
    p.add_argument("--no_semantic", action="store_true")
    p.add_argument("--time_budget_s", type=int, default=0,
                   help="Hard wall-clock stop, enforced between languages. 0 = no limit.")
    p.add_argument("--harvest", action="store_true",
                   help="After the sweep, best-of-n at compression budgets to build "
                        "verified elastic training rows.")
    p.add_argument("--harvest_sentences", type=int, default=60)
    p.add_argument("--harvest_reserve_s", type=int, default=420,
                   help="Do not start another language's harvest with less than this left.")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_required", action="store_true")
    p.add_argument("--wandb_project", default="indic-dubbing-v3")
    p.add_argument("--wandb_entity", default="nktthegreat-soccernet")
    p.add_argument("--wandb_run_name", default=None)
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
