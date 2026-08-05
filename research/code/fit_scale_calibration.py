"""
tools/fit_scale_calibration.py
===============================
Fits how far a generator misses the length it was asked for, so the next run can ask for
the number that lands where we want.

WHY THIS EXISTS
---------------
The bake-off rejected 106 candidates. 88% of them missed the requested length in the SAME
direction: the model under-compresses. Asked for 60% of the original, it returns above 69%.

A systematic miss and a random miss look identical in a yield number and call for opposite
responses. Random misses mean the tolerance is too tight. Systematic misses mean the
REQUEST is wrong, and widening the tolerance in that case just admits lengths we know are
wrong — the gate stops gating and the corpus quietly fills with rows that do not teach what
they claim to.

So: fit `achieved = a + b * asked` per language from every landing, kept and rejected
alike, and invert it. The rejects are most of the signal, which is why they are recorded.

Fitted on the same data the gate rejected — no new API calls.

USAGE
-----
    python -m tools.fit_scale_calibration --raw data/elastic/raw.jsonl \\
        --out research/scale_calibration.json
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MIN_POINTS = 12          # below this a per-language fit is noise; fall back to pooled
MIN_DISTINCT_SCALES = 2  # a line needs two points at different x


def fit(points: list[tuple[float, float]]) -> tuple[float, float, float]:
    """OLS achieved = a + b*asked, returning (a, b, r2)."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    mx, my = st.fmean(xs), st.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return my - mx, 1.0, 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    syy = sum((y - my) ** 2 for y in ys)
    b = sxy / sxx
    a = my - b * mx
    r2 = (sxy * sxy) / (sxx * syy) if syy > 0 else 0.0
    return a, b, r2


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", nargs="+", required=True,
                   help="raw_out JSONL from make_elastic_rows (one or more).")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    by_lang: dict[str, list[tuple[float, float]]] = collections.defaultdict(list)
    pooled: list[tuple[float, float]] = []
    for path in args.raw:
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            asked = r.get("asked_scale")
            achieved = r.get("achieved_scale")
            if asked is None or achieved is None:
                continue
            # Guard against runaway outliers: a "rewrite" five times the original length is
            # a different failure (the model ignored the task) and would drag the line.
            if not (0.1 <= achieved <= 4.0):
                continue
            by_lang[r["language"]].append((float(asked), float(achieved)))
            pooled.append((float(asked), float(achieved)))

    if not pooled:
        raise SystemExit("no usable landings in the raw file(s)")

    pa, pb, pr2 = fit(pooled)
    calib: dict = {"_all": {"a": round(pa, 4), "b": round(pb, 4), "r2": round(pr2, 4),
                            "n": len(pooled)}}

    print(f"\n{'lang':<6}{'n':>6}{'a':>9}{'b':>8}{'R2':>7}   "
          f"{'ask for 0.60':>13}{'ask for 0.75':>13}{'ask for 1.30':>13}")
    print("-" * 78)

    def show(name: str, a: float, b: float, r2: float, n: int) -> None:
        inv = lambda t: max(0.25, min(2.5, (t - a) / b)) if abs(b) > 1e-6 else t
        print(f"{name:<6}{n:>6}{a:>9.3f}{b:>8.3f}{r2:>7.3f}   "
              f"{inv(0.60):>13.2f}{inv(0.75):>13.2f}{inv(1.30):>13.2f}")

    for lg in sorted(by_lang):
        pts = by_lang[lg]
        if len(pts) < MIN_POINTS or len({round(x, 2) for x, _ in pts}) < MIN_DISTINCT_SCALES:
            print(f"{lg:<6}{len(pts):>6}   too few points — falls back to the pooled fit")
            continue
        a, b, r2 = fit(pts)
        # A negative or near-zero slope means asking for shorter did not produce shorter.
        # Inverting that would amplify noise into nonsense, so refuse it and pool instead.
        if b < 0.15:
            print(f"{lg:<6}{len(pts):>6}{a:>9.3f}{b:>8.3f}{r2:>7.3f}   slope too flat to "
                  f"invert — falls back to the pooled fit")
            continue
        calib[lg] = {"a": round(a, 4), "b": round(b, 4), "r2": round(r2, 4), "n": len(pts)}
        show(lg, a, b, r2, len(pts))
    print("-" * 78)
    show("ALL", pa, pb, pr2, len(pooled))

    bias = st.fmean([ach - ask for ask, ach in pooled])
    print(f"\nmean (achieved - asked) = {bias:+.3f}")
    print("Positive means the generator systematically returns MORE than it was asked for,")
    print("which is the bake-off's 88%-too-long finding expressed as one number.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(calib, indent=2), encoding="utf-8")
    print(f"\ncalibration -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
