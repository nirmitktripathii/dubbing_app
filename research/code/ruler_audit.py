"""
tools/ruler_audit.py
====================
Answers two questions about a training/validation corpus, and one decision that follows
from them.

QUESTION 1 — "Which ruler wrote these labels?"
----------------------------------------------
For every row, `n_phonemes` is compared against both candidate rulers computed from the
row's own target text: the non-space character count, and the true espeak-ng phoneme
count. Whichever matches is the ruler that wrote the label. A corpus where different rows
answer differently is **mixed**, which is worse than a corpus that is uniformly wrong: a
uniformly-wrong corpus teaches one consistent (if mislabelled) task, while a mixed corpus
teaches two contradictory tasks under the same prompt token and the model can only split
the difference between them.

This project's corpus is mixed. `training/dataset_generator.py` wrote character counts
(its espeak fallback fired for the whole run); `training/length_augmentation.py`, run in a
later session that did have espeak-ng, wrote real phoneme counts. Both used the prompt
token `[Target Phonemes: N]`.

QUESTION 2 — "Is chars -> phonemes a tight enough map to rescale instead of retrain?"
-------------------------------------------------------------------------------------
This is the load-bearing question and the reason this tool exists before any GPU is
booked. A model trained on character budgets learned a real, usable capability — it just
learned it in the wrong unit. If, *within a language*, phonemes are a near-constant
multiple of characters, then the existing checkpoint is salvageable with no retraining at
all: at dub time, convert the phoneme budget you want into the character budget the model
was actually taught, prompt with that, and the model lands where you meant.

If instead the ratio is noisy within a language, then the character label was only weakly
informative about duration, the conditioning signal the model received was correspondingly
noisy, and no amount of inference-time arithmetic recovers it — the corpus must be
relabelled and the model retrained.

THE DECISION RULE, PRE-REGISTERED
----------------------------------
Written down here, before the numbers are seen, so the answer cannot be rationalised after
the fact (Gate 0 of the training doctrine).

Per language, over the phonemes/chars ratio:

    CV = std(ratio) / mean(ratio)          # coefficient of variation, unitless

  - **CV <= 0.08 for all 11 languages** -> SALVAGE. A per-language constant rescale
    recovers the capability. Retraining buys accuracy, not correctness. Ship the rescale,
    relabel the corpus for the *next* run, do not spend a session on it now.

  - **CV > 0.15 for any language** -> RETRAIN that language's supervision. The label was
    too weakly coupled to duration to have taught budget obedience.

  - **In between** -> rescale, and measure the residual length error on the probe. The
    rescale is free; whether it is sufficient is an empirical question the length-response
    probe answers directly.

The 0.08 threshold is not arbitrary: the fine-tuned model's own reported length error is
~10%, so a rescale whose own noise floor is under ~8% is not the binding constraint. A
rescale noisier than the model's existing error would be adding more error than it removes.

USAGE
-----
    python -m tools.ruler_audit --jsonl data/translation_dataset/train.jsonl \\
                                --out research/ruler_audit_train.json

    # audit several at once, e.g. train + val + the augmented file
    python -m tools.ruler_audit --jsonl data/.../train.jsonl data/.../val.jsonl \\
                                --out research/ruler_audit.json

Exit code is 1 if any audited file is mixed-ruler or character-ruled, so this can gate a
notebook cell rather than merely inform one.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.languages import LANGUAGES  # noqa: E402
from common.phonemes import (  # noqa: E402
    RULER_CHARS, RULER_PHONEMES, RULER_UNKNOWN, assert_g2p_available, count_chars,
    phonemize_many, ruler_id, validate_inventory,
)

logger = logging.getLogger("ruler_audit")

# Pre-registered thresholds — see module docstring. Do not tune these to make a result
# come out the way you want; change them only with a written justification.
CV_SALVAGE_MAX = 0.08
CV_RETRAIN_MIN = 0.15


def read_jsonl(path: str) -> list[dict]:
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
        logger.warning("%s: %d unparseable lines skipped", path, bad)
    return rows


def _fit_through_origin(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """OLS slope for y = k*x with no intercept, plus R^2 about that model.

    Through the origin rather than with a free intercept because the relationship is
    physically proportional — a sentence with no characters has no phonemes — and a free
    intercept would let a bad fit hide behind an offset.
    """
    sxx = sum(x * x for x in xs)
    if sxx == 0:
        return float("nan"), float("nan")
    k = sum(x * y for x, y in zip(xs, ys)) / sxx
    ss_res = sum((y - k * x) ** 2 for x, y in zip(xs, ys))
    my = sum(ys) / len(ys)
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return k, r2


def audit_file(path: str, max_rows_per_lang: int | None = None) -> dict:
    rows = read_jsonl(path)
    by_lang: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_lang[r.get("language", "unknown")].append(r)

    per_lang: dict[str, dict] = {}
    for lang in sorted(by_lang):
        lang_rows = by_lang[lang]
        if max_rows_per_lang:
            lang_rows = lang_rows[:max_rows_per_lang]
        if lang not in LANGUAGES:
            logger.warning("skipping unknown language '%s' (%d rows)", lang, len(lang_rows))
            continue

        targets = [(r.get("target") or r.get("completion") or "") for r in lang_rows]
        labels = [int(r.get("n_phonemes") or 0) for r in lang_rows]

        # One batched espeak call per language rather than one per row.
        tokenized = phonemize_many(targets, lang)
        true_ph = [len(toks) for toks in tokenized]
        chars = [count_chars(t) for t in targets]

        # The phoneme inventory this language's G2P actually produced, over the whole
        # corpus. A count cannot tell you whether the things being counted are phonemes;
        # the inventory can. `leaks` names any source-script grapheme that rode through
        # untranslated — a hole in the converter, with the exact codepoint that fell in.
        inventory = Counter()
        for toks in tokenized:
            inventory.update(toks)
        leaks = validate_inventory(inventory, lang)

        keep = [i for i in range(len(labels)) if labels[i] > 0 and chars[i] > 0 and true_ph[i] > 0]
        if not keep:
            continue
        labels = [labels[i] for i in keep]
        true_ph = [true_ph[i] for i in keep]
        chars = [chars[i] for i in keep]
        aug_flags = [("augmentation" in lang_rows[i]) for i in keep]

        n = len(labels)
        match_chars = sum(1 for i in range(n) if labels[i] == chars[i])
        match_phon = sum(1 for i in range(n) if labels[i] == true_ph[i])

        ratios = [true_ph[i] / chars[i] for i in range(n)]
        mean_ratio = statistics.fmean(ratios)
        # Sample stdev (n-1). The old eval harness used the population divisor, which
        # understates spread on small samples — exactly where an audit must not be
        # optimistic.
        sd_ratio = statistics.stdev(ratios) if n > 1 else 0.0
        cv = sd_ratio / mean_ratio if mean_ratio else float("nan")
        k, r2 = _fit_through_origin([float(c) for c in chars], [float(p) for p in true_ph])

        # How wrong is the label, in the unit that matters?
        label_err = [abs(labels[i] - true_ph[i]) / true_ph[i] for i in range(n)]

        frac_chars = match_chars / n
        frac_phon = match_phon / n
        if frac_chars >= 0.95:
            ruler = RULER_CHARS
        elif frac_phon >= 0.95:
            ruler = RULER_PHONEMES
        elif frac_chars + frac_phon >= 0.95:
            ruler = "MIXED"
        else:
            ruler = RULER_UNKNOWN

        per_lang[lang] = {
            "n": n,
            "n_augmented": sum(aug_flags),
            "ruler": ruler,
            "inventory_size": len(inventory),
            "inventory_top20": inventory.most_common(20),
            "inventory_leaks": leaks,
            "frac_label_eq_chars": round(frac_chars, 4),
            "frac_label_eq_phonemes": round(frac_phon, 4),
            "phonemes_per_char_mean": round(mean_ratio, 4),
            "phonemes_per_char_sd": round(sd_ratio, 4),
            "phonemes_per_char_cv": round(cv, 4),
            "ols_k_through_origin": round(k, 4),
            "ols_r2": round(r2, 4),
            "label_rel_error_vs_true_phonemes_mean": round(statistics.fmean(label_err), 4),
            "verdict": (
                "SALVAGE" if cv <= CV_SALVAGE_MAX
                else "RETRAIN" if cv > CV_RETRAIN_MIN
                else "RESCALE_THEN_MEASURE"
            ),
        }

    rulers = {v["ruler"] for v in per_lang.values()}
    if rulers == {RULER_PHONEMES}:
        corpus_ruler = RULER_PHONEMES
    elif rulers == {RULER_CHARS}:
        corpus_ruler = RULER_CHARS
    else:
        corpus_ruler = "MIXED"

    verdicts = {v["verdict"] for v in per_lang.values()}
    if verdicts == {"SALVAGE"}:
        corpus_verdict = "SALVAGE"
    elif "RETRAIN" in verdicts:
        corpus_verdict = "RETRAIN"
    else:
        corpus_verdict = "RESCALE_THEN_MEASURE"

    return {
        "file": path,
        "n_rows": len(rows),
        "scoring_ruler": ruler_id(),
        "corpus_ruler": corpus_ruler,
        "corpus_verdict": corpus_verdict,
        "per_language": per_lang,
    }


def _print_table(result: dict) -> None:
    print(f"\n=== {result['file']} ===")
    print(f"rows={result['n_rows']}  corpus_ruler={result['corpus_ruler']}  "
          f"verdict={result['corpus_verdict']}")
    print(f"scored with: {result['scoring_ruler']}\n")
    hdr = (f"{'lang':<5}{'n':>6}{'aug':>6}  {'ruler':<16}{'=chars':>8}{'=phon':>8}"
           f"{'ph/char':>9}{'CV':>8}{'R2':>7}{'lblErr':>8}  verdict")
    print(hdr)
    print("-" * len(hdr))
    for lang, v in result["per_language"].items():
        print(f"{lang:<5}{v['n']:>6}{v['n_augmented']:>6}  {v['ruler']:<16}"
              f"{v['frac_label_eq_chars']:>8.2f}{v['frac_label_eq_phonemes']:>8.2f}"
              f"{v['phonemes_per_char_mean']:>9.3f}{v['phonemes_per_char_cv']:>8.3f}"
              f"{v['ols_r2']:>7.3f}{v['label_rel_error_vs_true_phonemes_mean']:>8.3f}"
              f"  {v['verdict']}")

    print("\nPhoneme inventories (validation, not a pipeline metric):")
    for lang, v in result["per_language"].items():
        top = " ".join(s for s, _ in v["inventory_top20"][:14])
        print(f"  {lang}: {v['inventory_size']:>3} distinct | {top}")
        for lk in v["inventory_leaks"]:
            print(f"      !! LEAK {lk}")
    if not any(v["inventory_leaks"] for v in result["per_language"].values()):
        print("  no untranslated source-script graphemes in any language's output.")


def _print_decision(results: list[dict]) -> None:
    print("\n" + "=" * 78)
    print("DECISION")
    print("=" * 78)
    verdicts = {r["corpus_verdict"] for r in results}
    if "RETRAIN" in verdicts:
        print("RETRAIN. At least one language's phoneme/char ratio varies more than 15%")
        print("within the language, so the character label was too weakly coupled to")
        print("duration to have taught budget obedience. Relabel the corpus with")
        print("tools/relabel_dataset.py and retrain. A rescale cannot recover this.")
    elif verdicts == {"SALVAGE"}:
        print("SALVAGE. Every language's phoneme/char ratio is stable to within 8%.")
        print("The existing checkpoint learned the right capability in the wrong unit.")
        print("Apply the per-language `ols_k_through_origin` as an inference-time budget")
        print("conversion (phoneme budget / k = the character budget the model was")
        print("taught) and re-run the length-response probe to confirm. Relabel the")
        print("corpus for the NEXT training run, not this one.")
    else:
        print("RESCALE, THEN MEASURE. The ratio is stable enough that the conversion is")
        print("worth applying and free, but not so stable that it is guaranteed")
        print("sufficient. Apply the rescale, run the length-response probe, and let the")
        print("residual slope decide whether a relabelled retrain is still needed.")
    print("\nPer-language conversion constants (phoneme_budget / k = char_budget to prompt):")
    for r in results:
        for lang, v in r["per_language"].items():
            print(f"  {lang}: k = {v['ols_k_through_origin']:.4f}  (R2 {v['ols_r2']:.3f}, "
                  f"CV {v['phonemes_per_char_cv']:.3f})")
        break  # constants are a property of the language, not the file


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--jsonl", nargs="+", required=True, help="One or more corpus files to audit.")
    p.add_argument("--out", default=None, help="Write the full JSON report here.")
    p.add_argument("--max_rows_per_lang", type=int, default=None,
                   help="Subsample for a fast pass; omit to audit every row.")
    args = p.parse_args()

    # Refuse to audit at all without a verified phonemizer — an audit that silently used
    # the character fallback would 'prove' the corpus was correctly labelled.
    assert_g2p_available()

    results = [audit_file(path, args.max_rows_per_lang) for path in args.jsonl]
    for r in results:
        _print_table(r)
    _print_decision(results)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
        print(f"\nWrote {args.out}")

    bad = [r for r in results if r["corpus_ruler"] != RULER_PHONEMES]
    if bad:
        print(f"\nFAIL: {len(bad)} file(s) are not phoneme-ruled: "
              f"{', '.join(r['file'] for r in bad)}")
        return 1
    print("\nPASS: every audited file is phoneme-ruled.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
