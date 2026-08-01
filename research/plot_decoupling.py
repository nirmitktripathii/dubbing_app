"""
plot_decoupling.py
------------------
Renders the central finding of the length-constrained translation fine-tune:
**training loss went flat while length control kept improving.**

Two measures, two different units, one shared x-axis — so this is drawn as two
stacked panels (small multiples), NOT a dual-axis chart. Overlaying two y-scales
on one plot would let the arbitrary alignment of those scales invent a
relationship that isn't in the data; stacked panels sharing the x-axis show the
same divergence without that risk.

Inputs (both produced by evaluation/phoneme_adherence_eval.py):
  evaluation/results/eval_out_ce__trajectory_summary.csv   — CE at 23 checkpoints
  evaluation/results/eval_out_all__trajectory_summary.csv  — slope at 4 checkpoints

Usage:
    python plot_decoupling.py [--outdir figures]

Writes decoupling_light.png and decoupling_dark.png at 2x scale for print/Retina.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "evaluation" / "results"

# Palette roles. Light and dark are selected steps of the same hues, not a flip.
THEME = {
    "light": {
        "surface":   "#fcfcfb",
        "ink":       "#0b0b0b",
        "secondary": "#52514e",
        "muted":     "#898781",
        "grid":      "#e1e0d9",
        "axis":      "#c3c2b7",
        "series_1":  "#2a78d6",   # blue  — cross-entropy
        "series_2":  "#eb6834",   # orange — length slope
        "band":      "#f0efec",
    },
    "dark": {
        "surface":   "#1a1a19",
        "ink":       "#ffffff",
        "secondary": "#c3c2b7",
        "muted":     "#898781",
        "grid":      "#2c2c2a",
        "axis":      "#383835",
        "series_1":  "#3987e5",
        "series_2":  "#d95926",
        "band":      "#383835",
    },
}

PLATEAU_START = 3200          # global CE minimum
SANS = ["Segoe UI", "DejaVu Sans", "sans-serif"]


def load():
    ce_raw = pd.read_csv(RESULTS / "eval_out_ce__trajectory_summary.csv")
    all_raw = pd.read_csv(RESULTS / "eval_out_all__trajectory_summary.csv")
    # The base model sits at step -1. Plotting it would compress the x-axis, but
    # DROPPING it silently would be worse: with only the four fine-tuned points in
    # view, the y-axis auto-scales to a ~0.03 window and the 2558→3200 wobble fills
    # the panel, which reads as a collapse. So the base value is kept out of the
    # line and drawn as a labelled reference, which also fixes the scale.
    base_slope = float(all_raw.loc[all_raw["step"] < 0, "length_slope"].iloc[0])
    ce = ce_raw[ce_raw["step"] >= 0].sort_values("step")
    allm = all_raw[all_raw["step"] >= 0].sort_values("step")
    return ce, allm, base_slope


def style_axis(ax, colors, *, last=False):
    ax.set_facecolor(colors["surface"])
    ax.grid(True, color=colors["grid"], linewidth=0.8, alpha=1.0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(colors["axis"])
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=colors["muted"], labelsize=9, length=0)
    if not last:
        ax.tick_params(labelbottom=False)


def render(mode: str, outdir: Path):
    colors = THEME[mode]
    ce, allm, base_slope = load()

    plt.rcParams["font.family"] = SANS

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9, 7.4), sharex=True,
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.30},
    )
    fig.patch.set_facecolor(colors["surface"])

    xmax = max(ce["step"].max(), allm["step"].max())

    # ---- Panel 1: cross-entropy -----------------------------------------
    style_axis(ax1, colors)
    ax1.axvspan(PLATEAU_START, xmax * 1.02, color=colors["band"], zorder=0)
    ax1.plot(ce["step"], ce["ce_mean"], color=colors["series_1"],
             linewidth=2.0, solid_capstyle="round", zorder=3)

    lo = ce.loc[ce["ce_mean"].idxmin()]
    ax1.plot([lo["step"]], [lo["ce_mean"]], "o", markersize=9,
             color=colors["series_1"], markeredgecolor=colors["surface"],
             markeredgewidth=2, zorder=4)
    ax1.annotate(
        f"minimum  {lo['ce_mean']:.4f}\nstep {int(lo['step'])}",
        xy=(lo["step"], lo["ce_mean"]),
        xytext=(-14, 26), textcoords="offset points",
        ha="right", fontsize=9, color=colors["secondary"], linespacing=1.4,
    )
    ax1.set_title("Training loss (cross-entropy) flattens",
                  color=colors["ink"], fontsize=12.5, fontweight="600",
                  loc="left", pad=10)
    ax1.set_ylabel("cross-entropy", color=colors["secondary"], fontsize=9.5)

    # ---- Panel 2: length slope ------------------------------------------
    style_axis(ax2, colors, last=True)
    ax2.axvspan(PLATEAU_START, xmax * 1.02, color=colors["band"], zorder=0)

    # Base model as a labelled reference. This sets an honest y-scale: without it
    # the panel auto-zooms onto a 0.03 window and the mid-run wobble looks huge.
    ax2.axhline(base_slope, color=colors["axis"], linewidth=1.0, zorder=1)
    ax2.annotate(f"base model  {base_slope:.3f}",
                 xy=(60, base_slope), xytext=(0, 6), textcoords="offset points",
                 ha="left", va="bottom", fontsize=9, color=colors["muted"])

    ax2.plot(allm["step"], allm["length_slope"], color=colors["series_2"],
             linewidth=2.0, marker="o", markersize=9,
             markeredgecolor=colors["surface"], markeredgewidth=2,
             solid_capstyle="round", zorder=3)

    # Direct-label only the endpoints, not every point.
    first, last_pt = allm.iloc[0], allm.iloc[-1]
    ax2.annotate(f"{first['length_slope']:.3f}",
                 xy=(first["step"], first["length_slope"]),
                 xytext=(-6, 14), textcoords="offset points",
                 ha="right", fontsize=9, color=colors["secondary"])
    ax2.annotate(f"{last_pt['length_slope']:.3f}",
                 xy=(last_pt["step"], last_pt["length_slope"]),
                 xytext=(-4, 14), textcoords="offset points",
                 ha="right", fontsize=9, color=colors["secondary"],
                 fontweight="600")

    span = allm["length_slope"].max() - base_slope
    ax2.set_ylim(base_slope - span * 0.35, allm["length_slope"].max() + span * 0.45)
    ax2.set_title("Length control is still drifting upward",
                  color=colors["ink"], fontsize=12.5, fontweight="600",
                  loc="left", pad=10)
    ax2.set_ylabel("length slope  (1.0 = full obedience)",
                   color=colors["secondary"], fontsize=9.5)
    ax2.set_xlabel("training step", color=colors["secondary"], fontsize=9.5)
    ax2.set_xlim(0, xmax * 1.03)

    # Label the shaded region once, anchored inside the plot so it can't overflow.
    ax1.annotate(
        f"loss plateau, from step {PLATEAU_START}",
        xy=(xmax * 1.01, ax1.get_ylim()[1]),
        xytext=(-6, -10), textcoords="offset points",
        ha="right", va="top", fontsize=9, color=colors["muted"],
    )

    fig.suptitle(
        "The loss curve went quiet before the model stopped learning",
        color=colors["ink"], fontsize=15, fontweight="600", x=0.055, ha="left", y=0.982,
    )
    fig.text(
        0.055, 0.945,
        "Cross-entropy is measured by showing the model the right answer; length slope by "
        "making it generate and\ncounting the sounds. Independent measurements — which is "
        "why one can move while the other is flat.",
        color=colors["secondary"], fontsize=9.5, ha="left", va="top", linespacing=1.5,
    )
    fig.text(
        0.055, 0.020,
        "Source: evaluation/results/eval_out_ce__trajectory_summary.csv and "
        "eval_out_all__trajectory_summary.csv",
        color=colors["muted"], fontsize=8, ha="left",
    )

    fig.subplots_adjust(top=0.855, bottom=0.095, left=0.105, right=0.965)

    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"decoupling_{mode}.png"
    fig.savefig(out, dpi=200, facecolor=colors["surface"])
    plt.close(fig)
    print(f"wrote {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=str(HERE / "figures"))
    args = ap.parse_args()
    outdir = Path(args.outdir)
    for mode in ("light", "dark"):
        render(mode, outdir)

    ce, allm, base_slope = load()
    print(f"\nTable view (the figure's WCAG-clean twin).  base model slope = {base_slope:.4f}")
    print("\ncross-entropy by step:")
    print(ce[["step", "ce_mean"]].to_string(index=False))
    print("\nlength slope by step:")
    print(allm[["step", "length_slope", "adherence_rel_mean", "chrf_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
