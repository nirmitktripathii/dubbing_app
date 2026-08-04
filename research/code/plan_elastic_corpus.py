"""
tools/plan_elastic_corpus.py
=============================
Turns "the corpus is not elastic enough" into a number: exactly how many elastic groups
have to be manufactured, per language, to clear the gate.

WHY THIS IS A TOOL AND NOT A GUESS
-----------------------------------
The corpus that trained checkpoint 3801 has 0.2% of its rows in a group where one English
sentence appears with more than one budget, so "ignore the number and translate naturally"
is a correct answer almost everywhere and nothing penalises a model for not reading the
budget. Fixing that means generating new rows, and generation costs either GPU quota or API
calls. Both are worth sizing before spending.

THE ARITHMETIC
--------------
Let `B` be the existing base rows and `G` the number of elastic groups we build, each
contributing one base row plus `v` extra budget variants. Rows living in a multi-budget
group are then `G x (1 + v)`, out of a total of `B + G x v`, so the elastic row fraction is

    f = G(1+v) / (B + Gv)

Solving for G at a target f:

    G = f x B / ((1 + v) - f x v)

That is the whole model. It is written down here rather than done in someone's head because
the answer decides whether this is an afternoon of API calls or a second GPU session.

USAGE
-----
    python -m tools.plan_elastic_corpus --corpus data/train.phonemes.jsonl \\
        --have data/translation_dataset/train_augmented_subset.jsonl --target_frac 0.20
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.languages import LANGUAGES  # noqa: E402
from tools.preflight import (  # noqa: E402
    MIN_COMPRESS_SHARE, MIN_ELASTIC_GROUPS_PER_LANG, load_jsonl,
)


def elastic_groups(rows: list[dict]) -> collections.Counter:
    """(language, english) groups that are seen with more than one budget."""
    g = collections.defaultdict(set)
    for r in rows:
        try:
            g[(r.get("language"), r.get("english"))].add(int(r.get("n_phonemes", -1)))
        except (TypeError, ValueError):
            continue
    return collections.Counter(k[0] for k, v in g.items() if len(v) > 1)


def direction_split(rows: list[dict]) -> tuple[int, int]:
    by_group = collections.defaultdict(list)
    for r in rows:
        by_group[(r.get("language"), r.get("english"))].append(r)
    comp = exp = 0
    for grp in by_group.values():
        budgets = [int(x.get("n_phonemes", -1)) for x in grp]
        if len(set(budgets)) < 2:
            continue
        base_rows = [x for x in grp if not x.get("augmentation")]
        base = int(base_rows[0]["n_phonemes"]) if base_rows else sorted(budgets)[len(budgets) // 2]
        for x in grp:
            if base_rows and x in base_rows:
                continue
            n = int(x.get("n_phonemes", -1))
            if n < base * 0.98:
                comp += 1
            elif n > base * 1.02:
                exp += 1
    return comp, exp


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", required=True, help="The base training corpus.")
    p.add_argument("--have", nargs="*", default=[],
                   help="Files of already-built elastic rows to count against the target.")
    p.add_argument("--target_frac", type=float, default=0.20,
                   help="Target fraction of rows living in a multi-budget group.")
    p.add_argument("--variants", type=int, default=2,
                   help="Extra budgets per English sentence (e.g. one shorter, one longer).")
    p.add_argument("--compress_share", type=float, default=0.60,
                   help="Fraction of the new variants that should ask for LESS. Dubbing "
                        "compresses: the base model overshoots by 34%%.")
    p.add_argument("--json", default=None)
    args = p.parse_args()

    base = load_jsonl(Path(args.corpus))
    base_per_lang = collections.Counter(r.get("language") for r in base)
    have_rows: list[dict] = []
    for h in args.have:
        have_rows += load_jsonl(Path(h))
    have_groups = elastic_groups(have_rows)

    B = len(base)
    v = args.variants
    f = args.target_frac
    denom = (1 + v) - f * v
    G_total = math.ceil(f * B / denom) if denom > 0 else 0

    # Split the target across languages in proportion to their share of the base corpus,
    # then enforce the per-language floor. Deliberately NOT reweighted toward the weak
    # languages: the standing constraint is that the languages which already work are the
    # ones with the most to lose, and changing the mix risks catastrophic forgetting.
    langs = sorted(LANGUAGES)
    per_lang_target = {}
    for lg in langs:
        share = base_per_lang.get(lg, 0) / B if B else 0
        per_lang_target[lg] = max(MIN_ELASTIC_GROUPS_PER_LANG, round(G_total * share))

    print(f"\nBase corpus      : {args.corpus}")
    print(f"  rows           : {B}")
    print(f"  target         : {f:.0%} of rows in a multi-budget group, {v} variants each")
    print(f"  groups needed  : {G_total}  ->  {G_total * v} new rows")
    if args.have:
        print(f"  already built  : {sum(have_groups.values())} groups in "
              f"{len(have_groups)} languages ({', '.join(args.have)})")
        c, e = direction_split(have_rows)
        tot = c + e
        print(f"  their direction: {c} compress / {e} expand "
              f"({c / tot:.0%} compress, floor {MIN_COMPRESS_SHARE:.0%})"
              if tot else "  their direction: none")

    print(f"\n{'lang':<6}{'base rows':>11}{'target grp':>12}{'have':>7}{'BUILD':>8}"
          f"{'  of which compress':>20}")
    print("-" * 66)
    plan = {}
    total_build = 0
    for lg in langs:
        tgt = per_lang_target[lg]
        have = have_groups.get(lg, 0)
        build = max(0, tgt - have)
        total_build += build
        n_new_rows = build * v
        n_comp = round(n_new_rows * args.compress_share)
        plan[lg] = {"base_rows": base_per_lang.get(lg, 0), "target_groups": tgt,
                    "have_groups": have, "build_groups": build,
                    "new_rows": n_new_rows, "compress_rows": n_comp,
                    "expand_rows": n_new_rows - n_comp}
        flag = "  <- nothing yet" if have == 0 else ""
        print(f"{lg:<6}{base_per_lang.get(lg, 0):>11}{tgt:>12}{have:>7}{build:>8}"
              f"{n_comp:>20}{flag}")

    total_rows = total_build * v
    print("-" * 66)
    print(f"{'TOTAL':<6}{B:>11}{sum(per_lang_target.values()):>12}"
          f"{sum(have_groups.values()):>7}{total_build:>8}"
          f"{round(total_rows * args.compress_share):>20}")
    print(f"\n{total_rows} new rows to generate "
          f"({args.compress_share:.0%} compression / {1 - args.compress_share:.0%} expansion).")
    print("Each is a paraphrase of an existing reference translation at a different length,")
    print("kept only if it passes BOTH gates: within tolerance of the requested phoneme")
    print("count, and above the 0.80 semantic threshold. Expect to generate more than this")
    print("and discard the failures — the gates are the point.")

    if args.json:
        Path(args.json).write_text(json.dumps(
            {"base_rows": B, "target_frac": f, "variants": v,
             "compress_share": args.compress_share,
             "groups_needed": G_total, "total_build_groups": total_build,
             "total_new_rows": total_rows, "per_language": plan},
            indent=2), encoding="utf-8")
        print(f"\nplan -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
