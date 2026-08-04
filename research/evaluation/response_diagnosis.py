"""
evaluation/response_diagnosis.py
=================================
Turns a raw budget-response sweep into a **pre-registered** diagnosis and action.

WHY PRE-REGISTERED
------------------
"Slope 0.35" is compatible with at least four different defects that need four different
fixes. Deciding which one you are looking at *after* seeing the numbers, with a quota
half-spent and a deadline in view, is how a project talks itself into the cheap answer.
So the thresholds and the actions they imply are written here, before the run, and the
classifier is applied mechanically to whatever comes back.

THE FOUR THINGS A LOW SLOPE CAN MEAN
-------------------------------------
Obeying a phoneme budget is not one skill. It is three, and they fail differently:

  (a) NOTICING the number at all
  (b) ESTIMATING how long your own candidate output will be
  (c) REWRITING at a target length without wrecking the meaning

  FLAT        output length barely moves whatever you ask for      -> (a) fails
  NOISY       length moves, but not with the request               -> (b) fails
  SATURATING  tracks near the natural length, refuses to go far    -> (c) fails
  ASYMMETRIC  one direction works, the other does not              -> (c) fails, one way

Each has a different fix, and three of the four are NOT fixed by more training on the same
data. Distinguishing them is the entire purpose of the sweep.

NORMALISATION, AND WHY THE OLD PROBE SLOPE HAD A FLOOR
------------------------------------------------------
Points are normalised per sentence to `produced / natural` against `scale`, and fitted
**with an intercept**. Both choices are load-bearing, and getting them wrong is what made
the previous numbers hard to read.

The old probe regressed raw `produced` on raw `requested`, pooling points from sentences of
different lengths. But `requested = scale x natural`, so the between-sentence spread of
`natural` leaks straight into the fit. Consequence: a model that emits its natural
translation every time, completely ignoring the budget, does **not** score 0. Measured on
this project's actual validation sentences it scores **0.60 to 0.83**, per language:

    lang  as    bn    gu    hi    kn    ml    mr    or    pa    ta    te
    floor .788  .747  .603  .832  .707  .748  .787  .741  .831  .651  .766

So a raw probe slope is not interpretable without its floor, and the floor is different in
every language. Six of the eleven scored at or below their own floor at checkpoint 3801 —
that is the real meaning of "the budget is being ignored".

Fitting `ratio = a + k*scale` per sentence removes both problems. `natural` divides out, so
sentence length cannot leak in, and the intercept absorbs the constant part — leaving
k = 0 for a model that ignores the budget and k = 1 for one that follows it, in every
language, with no floor to subtract.
"""

from __future__ import annotations

import collections
import statistics as st
from typing import Iterable, Optional

# ---------------------------------------------------------------------------------------
# Pre-registered thresholds. Set before the run. Do not tune these to make a result
# come out a particular way — if they are wrong, change them and re-diagnose everything.
# ---------------------------------------------------------------------------------------
OBEYS_SLOPE = 0.70          # at or above this, with a decent fit, the budget is followed
OBEYS_R2 = 0.60
FLAT_SLOPE = 0.30           # below this, and barely varying, the number is not being read
FLAT_CV = 0.12              # coefficient of variation of produced/natural within a sentence
ASYMMETRY = 0.35            # gap between the compress-side and expand-side slopes
SATURATE_MID = 0.60         # responds near 1.0x ...
SATURATE_FAR = 0.30         # ... but not out at the extremes
NOISY_R2 = 0.35

MID_BAND = (0.85, 1.15)
FAR_LOW = 0.70              # scales at or below this are the "far compression" band
FAR_HIGH = 1.60             # scales at or above this are the "far expansion" band

# What each diagnosis implies. These are decisions, made in advance, in one place.
ACTIONS = {
    "OBEYS": (
        "No fix needed. Hold this language out of any corrective training so it cannot "
        "regress, and keep it in the eval set as a forgetting canary."),
    "FLAT": (
        "The budget is not being read. This is the degenerate-objective signature and it "
        "is fixed by DATA, not by more steps: build elastic rows (same English, several "
        "budgets) until ignoring the number stops being a correct answer. Expect a large "
        "gain; this is the cheapest failure mode to fix."),
    "NOISY": (
        "The model cannot estimate its own output length. More elastic rows will NOT fix "
        "this on their own — the supervision it needs is length feedback, not more "
        "examples. Options, in order of cost: (1) rejection-sample its own generations "
        "and train only on the ones that verifiably landed, which is length feedback in "
        "SFT clothing; (2) an auxiliary length-prediction head; (3) fall back to the "
        "deployed v2 generate-and-select path for this language, which already works."),
    "SATURATING": (
        "The budget is read and followed near the natural length, then refuses. Check "
        "whether the floor coincides with semantic collapse: if meaning is intact at the "
        "floor, this is a learned length prior and needs elastic rows AT the extremes, "
        "not more rows near 1.0x. If meaning is already breaking there, the floor is real "
        "and the honest fix is to cap what the pipeline asks of this language."),
    "ASYMMETRIC": (
        "One direction is trained and the other is not. Rebuild augmentation weighted "
        "toward the missing direction. Note the existing augmenter was 69% expand, and "
        "dubbing needs compression."),
    "WEAK": (
        "Partial response with no clean signature. Do not guess: widen the sweep and "
        "raise sentences-per-language for this one before committing training time."),
}


def _fit(xs: list[float], ys: list[float]) -> tuple[Optional[float], Optional[float]]:
    """OLS y = a + kx, returning (k, R^2).

    WITH an intercept, deliberately. Through the origin, a model that emits a constant
    ratio of 1.0 at every scale scores k = mean(s)/mean(s^2) ~ 0.82 on this sweep — it
    would look like it was following the budget. The intercept is what makes k = 0 mean
    "ignores the budget".
    """
    n = len(xs)
    if n < 3:
        return None, None
    mx, my = st.fmean(xs), st.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return None, None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    syy = sum((y - my) ** 2 for y in ys)
    k = sxy / sxx
    r2 = (sxy * sxy) / (sxx * syy) if syy > 0 else None
    return k, r2


def _slope(points: list[tuple[float, float]]) -> tuple[Optional[float], Optional[float]]:
    if len(points) < 3:
        return None, None
    return _fit([p[0] for p in points], [p[1] for p in points])


def diagnose_language(rows: Iterable[dict]) -> dict:
    """`rows` are raw sweep points for ONE language:
    {sentence_idx, scale, requested_n, produced_n, natural_n, semantic?}
    """
    pts: list[tuple[float, float]] = []          # (scale, produced/natural)
    by_sentence: dict[int, list[tuple[float, float]]] = collections.defaultdict(list)
    sem_by_band: dict[str, list[float]] = collections.defaultdict(list)

    for r in rows:
        nat = float(r.get("natural_n") or 0)
        prod = float(r.get("produced_n") or 0)
        scale = float(r.get("scale") or 0)
        if nat <= 0 or prod <= 0 or scale <= 0:
            continue
        ratio = prod / nat
        pts.append((scale, ratio))
        by_sentence[int(r.get("sentence_idx", -1))].append((scale, ratio))
        s = r.get("semantic")
        if s is not None:
            band = "far_low" if scale <= FAR_LOW else ("far_high" if scale >= FAR_HIGH else "mid")
            sem_by_band[band].append(float(s))

    if len(pts) < 6:
        return {"diagnosis": "WEAK", "n_points": len(pts),
                "note": "too few usable points to classify"}

    slope, r2 = _slope(pts)
    slope_comp, _ = _slope([p for p in pts if p[0] <= 1.0])
    slope_exp, _ = _slope([p for p in pts if p[0] >= 1.0])
    slope_mid, _ = _slope([p for p in pts if MID_BAND[0] <= p[0] <= MID_BAND[1]])
    slope_far_low, _ = _slope([p for p in pts if p[0] <= FAR_LOW])
    slope_far_high, _ = _slope([p for p in pts if p[0] >= FAR_HIGH])

    # Does the output length vary at all WITHIN a sentence as the budget moves? This is the
    # question a pooled slope cannot answer: a model emitting one fixed length per sentence
    # still gets a non-zero pooled slope because longer sentences have longer fixed lengths.
    cvs = []
    for _, sp in by_sentence.items():
        ratios = [r for _, r in sp]
        if len(ratios) >= 4 and st.fmean(ratios) > 0:
            cvs.append(st.pstdev(ratios) / st.fmean(ratios))
    cv = st.fmean(cvs) if cvs else None

    out = {
        "n_points": len(pts), "n_sentences": len(by_sentence),
        "slope": slope, "r2": r2,
        "slope_compress": slope_comp, "slope_expand": slope_exp,
        "slope_mid": slope_mid,
        "slope_far_compress": slope_far_low, "slope_far_expand": slope_far_high,
        "within_sentence_cv": cv,
        "semantic_mid": st.fmean(sem_by_band["mid"]) if sem_by_band.get("mid") else None,
        "semantic_far_compress": (st.fmean(sem_by_band["far_low"])
                                  if sem_by_band.get("far_low") else None),
    }

    asym = (abs(slope_comp - slope_exp)
            if slope_comp is not None and slope_exp is not None else None)
    out["asymmetry"] = asym

    # Priority order matters: OBEYS first so a good language is never pathologised, FLAT
    # second because it is the one signature that is unambiguous.
    if slope is not None and r2 is not None and slope >= OBEYS_SLOPE and r2 >= OBEYS_R2:
        d = "OBEYS"
    elif cv is not None and cv < FLAT_CV and (slope or 0) < FLAT_SLOPE:
        d = "FLAT"
    elif asym is not None and asym >= ASYMMETRY:
        d = "ASYMMETRIC"
    elif (slope_mid is not None and slope_mid >= SATURATE_MID
          and min([s for s in (slope_far_low, slope_far_high) if s is not None] or [1.0])
          < SATURATE_FAR):
        d = "SATURATING"
    elif r2 is not None and r2 < NOISY_R2:
        d = "NOISY"
    else:
        d = "WEAK"

    out["diagnosis"] = d
    out["action"] = ACTIONS[d]
    return out


def diagnose(points: Iterable[dict]) -> dict:
    """All languages. `points` is the flat sweep table."""
    by_lang: dict[str, list[dict]] = collections.defaultdict(list)
    for r in points:
        by_lang[r.get("language")].append(r)
    per_lang = {lg: diagnose_language(rs) for lg, rs in sorted(by_lang.items()) if lg}

    counts = collections.Counter(v["diagnosis"] for v in per_lang.values())
    # The corpus-level call is the majority failure mode among the languages that are not
    # already fine, because that is what the next training run has to be built for.
    broken = [d for d in (v["diagnosis"] for v in per_lang.values()) if d != "OBEYS"]
    dominant = collections.Counter(broken).most_common(1)[0][0] if broken else "OBEYS"
    return {
        "per_language": per_lang,
        "counts": dict(counts),
        "dominant_failure": dominant,
        "corpus_action": ACTIONS[dominant],
        "obeying": sorted(lg for lg, v in per_lang.items() if v["diagnosis"] == "OBEYS"),
    }


def format_report(result: dict) -> str:
    lines = ["", "Budget-response diagnosis", "=" * 108,
             f"{'lang':<6}{'diagnosis':<13}{'slope':>8}{'R2':>7}{'compress':>10}"
             f"{'expand':>9}{'mid':>8}{'farComp':>9}{'within-CV':>11}{'sem@0.7x':>11}"]
    lines.append("-" * 108)
    f = lambda v: f"{v:>8.3f}" if isinstance(v, (int, float)) else f"{'-':>8}"
    for lg, v in result["per_language"].items():
        lines.append(
            f"{lg:<6}{v['diagnosis']:<13}{f(v.get('slope'))}"
            f"{f(v.get('r2'))[1:]}{f(v.get('slope_compress'))[:10]:>10}"
            f"{f(v.get('slope_expand'))[:9]:>9}{f(v.get('slope_mid'))[:8]:>8}"
            f"{f(v.get('slope_far_compress'))[:9]:>9}"
            f"{(f'{v['within_sentence_cv']:.3f}' if v.get('within_sentence_cv') is not None else '-'):>11}"
            f"{(f'{v['semantic_far_compress']:.3f}' if v.get('semantic_far_compress') is not None else '-'):>11}")
    lines += ["", f"counts: {result['counts']}",
              f"already obeying: {result['obeying'] or 'none'}",
              f"dominant failure: {result['dominant_failure']}", "",
              "ACTION", "-" * 108]
    lines += ["  " + l for l in result["corpus_action"].split(". ")]
    return "\n".join(lines)
