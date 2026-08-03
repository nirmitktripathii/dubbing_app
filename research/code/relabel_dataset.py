"""
tools/relabel_dataset.py
========================
Rewrites a corpus's `n_phonemes` labels and `[Target Phonemes: N]` prompts using the
canonical counter in `common/phonemes.py`, and stamps enough provenance into the output
that the mislabelling this repairs can never recur silently.

WHAT IT REPAIRS
---------------
1. **The label itself.** `n_phonemes` is recomputed from the row's own target text with
   espeak-ng. Every row gets `ruler` and `n_phonemes_legacy` fields, so the change is
   auditable and reversible rather than destructive.

2. **The prompt.** `[Target Phonemes: N]` is regenerated from the new N. Leaving the
   prompt stale while fixing the label would be strictly worse than doing nothing — the
   model reads the prompt, and the loss is computed against a completion whose length now
   disagrees with it.

3. **The augmentation direction gate, which was evaluated in mixed units.**
   `length_augmentation.py` admitted a paraphrase if
   `(variant_phonemes - source_phonemes) / source_phonemes` moved >= 10% in the intended
   direction. But `variant_phonemes` was a real phoneme count while `source_phonemes` was
   read from the base row's `n_phonemes`, which was a character count. That expression is
   not a length-change measurement; it is a comparison between two different units, and
   its sign and magnitude are both untrustworthy.

   So every augmented row is re-tested with both sides measured in real phonemes. Variants
   that no longer clear the threshold were admitted on a broken comparison and are dropped
   by default — a "compressed" example that did not actually compress teaches the model
   that the budget token means nothing, which is the precise opposite of the objective.
   Use `--keep_failed_augmentation` to retain and flag them instead of dropping.

WHAT IT DELIBERATELY DOES NOT DO
---------------------------------
It does not re-run the *semantic* gate. That gate compared the paraphrase against the
human reference with a multilingual embedder and its 0.80 threshold is unit-free, so it
was unaffected by the ruler bug and its verdicts still stand. Re-running it would need a
GPU and would change nothing.

USAGE
-----
    python -m tools.relabel_dataset \\
        --in  data/translation_dataset/train.jsonl \\
        --out data/translation_dataset/train.phonemes.jsonl

    # then confirm:
    python -m tools.ruler_audit --jsonl data/translation_dataset/train.phonemes.jsonl

A sidecar `<out>.manifest.json` records the espeak version, the ruler id, per-language
before/after label statistics, and the augmentation re-validation tally.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.languages import LANGUAGES  # noqa: E402
from common.phonemes import assert_g2p_available, count_chars, phonemize_many, ruler_id  # noqa: E402

logger = logging.getLogger("relabel_dataset")

PROMPT_TEMPLATE = '[Translate to {language}] [Target Phonemes: {n_phonemes}] "{english}"'

# Must match training/length_augmentation.py's default. Imported as a literal rather than
# from that module because that module pulls in torch-dependent backends.
MIN_LENGTH_CHANGE = 0.10


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
        logger.warning("%d unparseable lines skipped in %s", bad, path)
    return rows


def relabel(in_path: str, out_path: str, min_length_change: float = MIN_LENGTH_CHANGE,
            keep_failed_augmentation: bool = False) -> dict:
    g2p_manifest = assert_g2p_available()
    rows = read_jsonl(in_path)
    logger.info("Read %d rows from %s", len(rows), in_path)

    by_lang: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_lang[r.get("language", "unknown")].append(i)

    new_n: dict[int, int] = {}
    for lang, idxs in by_lang.items():
        if lang not in LANGUAGES:
            logger.warning("language '%s' is not in the supported table — %d rows left "
                           "untouched and flagged", lang, len(idxs))
            continue
        targets = [(rows[i].get("target") or rows[i].get("completion") or "") for i in idxs]
        counts = [len(toks) for toks in phonemize_many(targets, lang)]
        for i, c in zip(idxs, counts):
            new_n[i] = c
        logger.info("relabelled %-3s: %5d rows", lang, len(idxs))

    # Index base (non-augmented) rows so augmented variants can be re-tested against a
    # source measured in the same unit they now are.
    base_true: dict[tuple, int] = {}
    for i, r in enumerate(rows):
        if "augmentation" not in r and i in new_n:
            base_true[(r.get("language"), r.get("english"))] = new_n[i]

    stats: dict[str, dict] = defaultdict(lambda: {
        "n": 0, "n_augmented": 0, "dropped_augmentation": 0, "unresolved_source": 0,
        "label_before": [], "label_after": [],
    })
    out_rows: list[dict] = []
    dropped = 0

    for i, r in enumerate(rows):
        lang = r.get("language")
        if i not in new_n:
            out_rows.append({**r, "ruler": "unverified", "relabel_skipped": True})
            continue

        n = new_n[i]
        if n <= 0:
            dropped += 1
            continue

        s = stats[lang]
        s["n"] += 1
        s["label_before"].append(int(r.get("n_phonemes") or 0))
        s["label_after"].append(n)

        new_row = dict(r)
        new_row["n_phonemes_legacy"] = r.get("n_phonemes")
        new_row["n_chars"] = count_chars(r.get("target") or r.get("completion") or "")
        new_row["n_phonemes"] = n
        new_row["ruler"] = g2p_manifest["ruler"]
        new_row["prompt"] = PROMPT_TEMPLATE.format(
            language=LANGUAGES[lang].name, n_phonemes=n, english=r.get("english", ""),
        )

        aug = r.get("augmentation")
        if aug:
            s["n_augmented"] += 1
            src = base_true.get((lang, r.get("english")))
            if src is None or src <= 0:
                # No base row in this file to measure against. Keep it, but say so —
                # silently trusting the old mixed-unit number is what got us here.
                s["unresolved_source"] += 1
                new_row["augmentation"] = {
                    **aug,
                    "source_phonemes_legacy": aug.get("source_phonemes"),
                    "source_phonemes": None,
                    "length_change_revalidated": False,
                    "revalidation_note": "base row not present in this file",
                }
            else:
                change = (n - src) / src
                wanted = -1 if aug.get("direction") == "compress" else 1
                passed = (wanted * change) >= min_length_change
                new_row["augmentation"] = {
                    **aug,
                    "source_phonemes_legacy": aug.get("source_phonemes"),
                    "source_phonemes": src,
                    "relative_change": round(change, 4),
                    "length_change_revalidated": True,
                    "length_gate_passed": passed,
                }
                if not passed:
                    s["dropped_augmentation"] += 1
                    if not keep_failed_augmentation:
                        dropped += 1
                        continue

        out_rows.append(new_row)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    per_lang = {}
    for lang, s in sorted(stats.items()):
        before, after = s["label_before"], s["label_after"]
        changed = sum(1 for b, a in zip(before, after) if b != a)
        per_lang[lang] = {
            "n": s["n"],
            "n_augmented": s["n_augmented"],
            "labels_changed": changed,
            "labels_changed_frac": round(changed / max(s["n"], 1), 4),
            "mean_label_before": round(statistics.fmean(before), 2) if before else None,
            "mean_label_after": round(statistics.fmean(after), 2) if after else None,
            "mean_shift_pct": (
                round(100 * (statistics.fmean(after) / statistics.fmean(before) - 1), 2)
                if before and statistics.fmean(before) else None
            ),
            "augmentation_failed_revalidation": s["dropped_augmentation"],
            "augmentation_source_unresolved": s["unresolved_source"],
        }

    manifest = {
        "tool": "tools/relabel_dataset.py",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input": in_path,
        "output": out_path,
        "ruler": g2p_manifest["ruler"],
        "espeak_version": g2p_manifest["espeak_version"],
        "rows_in": len(rows),
        "rows_out": len(out_rows),
        "rows_dropped": dropped,
        "min_length_change": min_length_change,
        "keep_failed_augmentation": keep_failed_augmentation,
        "per_language": per_lang,
    }
    Path(str(out_path) + ".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="in_path", required=True)
    p.add_argument("--out", dest="out_path", required=True)
    p.add_argument("--min_length_change", type=float, default=MIN_LENGTH_CHANGE)
    p.add_argument("--keep_failed_augmentation", action="store_true",
                   help="Retain augmented rows that fail the re-validated length gate, "
                        "flagged rather than dropped.")
    args = p.parse_args()

    m = relabel(args.in_path, args.out_path, args.min_length_change,
                args.keep_failed_augmentation)

    print(f"\nruler: {m['ruler']}")
    print(f"rows: {m['rows_in']} in -> {m['rows_out']} out ({m['rows_dropped']} dropped)\n")
    hdr = (f"{'lang':<5}{'n':>7}{'aug':>6}{'changed':>9}{'before':>9}{'after':>9}"
           f"{'shift%':>9}{'augFail':>9}")
    print(hdr)
    print("-" * len(hdr))
    for lang, v in m["per_language"].items():
        print(f"{lang:<5}{v['n']:>7}{v['n_augmented']:>6}{v['labels_changed_frac']:>9.2f}"
              f"{v['mean_label_before'] or 0:>9.1f}{v['mean_label_after'] or 0:>9.1f}"
              f"{v['mean_shift_pct'] or 0:>9.1f}{v['augmentation_failed_revalidation']:>9}")
    print(f"\nWrote {m['output']} and {m['output']}.manifest.json")
    print("Next: python -m tools.ruler_audit --jsonl " + m["output"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
