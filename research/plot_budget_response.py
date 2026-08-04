"""
plot_budget_response.py
-----------------------
Renders the central finding of the corrected evaluation (Kaggle session `02i`):
**the eleven languages split into two populations, and the aggregate hides it.**

Six languages follow the phoneme budget (probe slope 0.69–0.94) and pay for it in
meaning. Five barely respond to the budget at all (0.35–0.42) and keep their meaning
intact. The mean, 0.637, describes none of them.

The left panel is the split itself. The right panel is why it matters: probe slope
plotted against semantic similarity, where the negative relationship is the trade-off
the fine-tune is actually making.

A note on the right panel: it is a scatter of eleven points, so no fit line is drawn.
Eleven points can support "there is a trade-off"; they cannot support a slope estimate,
and drawing a regression line would claim the second while only having evidence for the
first.

Input (produced by evaluation/phoneme_adherence_eval.py on Kaggle session 02i):
  evaluation/results/corrected_eval/corrected__per_checkpoint_metrics.csv

Usage:
    python plot_budget_response.py [--outdir figures]

Writes budget_response_light.png and budget_response_dark.png at 2x for print/Retina.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "evaluation" / "results" / "corrected_eval"
CHECKPOINT = "checkpoint-3801"

# Same palette as plot_decoupling.py — the two figures appear side by side in the
# write-up, so they must read as one family.
THEME = {
    "light": {
        "surface":   "#fcfcfb",
        "ink":       "#0b0b0b",
        "secondary": "#52514e",
        "muted":     "#898781",
        "grid":      "#e1e0d9",
        "axis":      "#c3c2b7",
        "follows":   "#eb6834",   # orange — obeys the budget
        "ignores":   "#2a78d6",   # blue   — ignores the budget
        "band":      "#f0efec",
    },
    "dark": {
        "surface":   "#1a1a19",
        "ink":       "#ffffff",
        "secondary": "#c3c2b7",
        "muted":     "#898781",
        "grid":      "#2c2c2a",
        "axis":      "#383835",
        "follows":   "#d95926",
        "ignores":   "#3987e5",
        "band":      "#383835",
    },
}

SANS = ["Segoe UI", "DejaVu Sans", "sans-serif"]
SPLIT = 0.55          # the empty band between the two populations
USABLE = 0.80         # what a production isochrony model needs


def load() -> pd.DataFrame:
    df = pd.read_csv(RESULTS / "corrected__per_checkpoint_metrics.csv")
    df = df[df["checkpoint"] == CHECKPOINT].copy()
    return df.sort_values("length_slope_probe", ascending=False).reset_index(drop=True)


def style_axis(ax, colors):
    ax.set_facecolor(colors["surface"])
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(colors["axis"])
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=colors["muted"], labelsize=9, length=0)


def render(mode: str, outdir: Path) -> Path:
    colors = THEME[mode]
    df = load()
    df["follows"] = df["length_slope_probe"] >= SPLIT
    bar_colors = [colors["follows"] if f else colors["ignores"] for f in df["follows"]]

    plt.rcParams["font.family"] = SANS
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12.6, 6.4),
        gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.26},
    )
    fig.patch.set_facecolor(colors["surface"])

    # ---- Panel 1: the split -------------------------------------------------
    style_axis(ax1, colors)
    ax1.grid(True, axis="x", color=colors["grid"], linewidth=0.8)
    y = range(len(df))
    ax1.barh(y, df["length_slope_probe"], color=bar_colors, height=0.68, zorder=3)
    ax1.set_yticks(list(y))
    ax1.set_yticklabels(df["language"], fontsize=10, color=colors["secondary"])
    ax1.set_xlim(0, 1.06)

    for i, v in enumerate(df["length_slope_probe"]):
        ax1.annotate(f"{v:.2f}", xy=(v, i), xytext=(5, 0), textcoords="offset points",
                     va="center", fontsize=9, color=colors["secondary"])

    # The two reference lines are annotated at opposite ends of the panel: stacking both
    # labels near the bottom bars ran them into each other.
    ax1.axvline(USABLE, color=colors["axis"], linewidth=1.0, linestyle=(0, (4, 3)), zorder=2)
    ax1.annotate("0.80 — usable for dubbing", xy=(USABLE, -0.55),
                 xytext=(6, 0), textcoords="offset points",
                 ha="left", va="center", fontsize=9, color=colors["muted"])

    mean_slope = df["length_slope_probe"].mean()
    ax1.axvline(mean_slope, color=colors["ink"], linewidth=1.0, alpha=0.45, zorder=2)
    ax1.annotate(f"mean {mean_slope:.2f} — describes no language",
                 xy=(mean_slope, len(df) - 0.25), xytext=(-7, 0), textcoords="offset points",
                 ha="right", va="center", fontsize=9, color=colors["secondary"])
    ax1.set_ylim(len(df) - 0.1, -0.9)

    ax1.set_title("Budget obedience splits the corpus in two",
                  color=colors["ink"], fontsize=12.5, fontweight="600", loc="left", pad=10)
    ax1.set_xlabel("probe slope  (1.0 = follows the budget, 0 = ignores it)",
                   color=colors["secondary"], fontsize=9.5)

    # ---- Panel 2: what obedience costs -------------------------------------
    style_axis(ax2, colors)
    ax2.grid(True, color=colors["grid"], linewidth=0.8)
    ax2.scatter(df["length_slope_probe"], df["semantic_mean"],
                s=170, c=bar_colors, edgecolors=colors["surface"],
                linewidths=2, zorder=3)
    for _, r in df.iterrows():
        ax2.annotate(r["language"],
                     xy=(r["length_slope_probe"], r["semantic_mean"]),
                     xytext=(0, -19), textcoords="offset points",
                     ha="center", fontsize=9.5, color=colors["secondary"])

    ax2.axhline(0.80, color=colors["axis"], linewidth=1.0, linestyle=(0, (4, 3)), zorder=2)
    ax2.annotate("0.80 — semantic gate threshold", xy=(0.50, 0.80),
                 xytext=(0, 6), textcoords="offset points",
                 ha="left", va="bottom", fontsize=9, color=colors["muted"])

    ax2.set_xlim(0.28, 1.02)
    ax2.set_ylim(0.70, 0.96)
    ax2.set_title("and every point of it is bought from meaning",
                  color=colors["ink"], fontsize=12.5, fontweight="600", loc="left", pad=10)
    ax2.set_xlabel("probe slope", color=colors["secondary"], fontsize=9.5)
    ax2.set_ylabel("semantic similarity to the reference",
                   color=colors["secondary"], fontsize=9.5)

    r = df["length_slope_probe"].corr(df["semantic_mean"])
    ax2.annotate(f"r = {r:+.2f}   (n = 11)", xy=(0.98, 0.94),
                 xycoords="axes fraction", ha="right", va="top",
                 fontsize=9.5, color=colors["secondary"])

    fig.suptitle("The average said 0.64. No language was at 0.64.",
                 color=colors["ink"], fontsize=15.5, fontweight="600",
                 x=0.042, ha="left", y=0.975)
    fig.text(0.042, 0.925,
             "Eleven Indic languages at checkpoint 3801, measured by holding the sentence "
             "fixed and sweeping only the phoneme budget.\nThe five languages that keep "
             "their meaning are the five that will not compress; the six that compress have "
             "already crossed the semantic gate.",
             color=colors["secondary"], fontsize=9.5, ha="left", va="top", linespacing=1.5)
    fig.text(0.042, 0.018,
             "Source: evaluation/results/corrected_eval/corrected__per_checkpoint_metrics.csv "
             "(Kaggle session 02i, ruler phonemes:espeak-ng-1.50)",
             color=colors["muted"], fontsize=8, ha="left")

    fig.subplots_adjust(top=0.815, bottom=0.115, left=0.075, right=0.975)

    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"budget_response_{mode}.png"
    fig.savefig(out, dpi=200, facecolor=colors["surface"])
    plt.close(fig)
    print(f"wrote {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=str(HERE / "figures"))
    args = ap.parse_args()
    for mode in ("light", "dark"):
        render(mode, Path(args.outdir))

    df = load()
    print("\nTable view (the figure's WCAG-clean twin), checkpoint-3801:\n")
    view = df[["language", "length_slope_probe", "length_r2_probe", "semantic_mean",
               "semantic_degraded_frac", "adherence_rel_mean", "chrf_mean"]]
    print(view.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nr(probe slope, semantic) = "
          f"{df['length_slope_probe'].corr(df['semantic_mean']):+.3f}   (n = 11)")


if __name__ == "__main__":
    main()
