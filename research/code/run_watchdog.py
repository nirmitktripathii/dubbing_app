"""
tools/run_watchdog.py
=====================
Watches a live W&B run, applies **pre-registered** rules to what it sees, and either raises
an alert or takes the decision itself.

WHY A WATCHDOG AND NOT JUST A DASHBOARD
----------------------------------------
Kaggle publishes a notebook's log only when the session ends, so W&B is the only channel
that reports while a multi-hour run is happening. But a channel nobody is reading is not
observability either. Two of this project's most expensive mistakes were visible in the
first twenty minutes of a run that then continued for five more hours.

So this polls, and it decides. The rules below are written before the run, when nothing is
at stake, precisely so that they still apply at hour four when stopping feels expensive and
"let it finish" is the tempting answer.

THE TWO MODES
-------------
`--alert`  (default) prints a verdict and stops polling on anything that needs a human.
`--decide` additionally applies the pre-registered default action for that verdict and
           records what it did and why. Use it when the run must not sit blocked waiting
           for someone to be awake.

Either way, everything is logged with its rationale. A decision taken by rule and written
down is reviewable; a decision taken by vibe at 3am is not.

WHAT IT WATCHES, AND WHY EACH RULE EXISTS
------------------------------------------
Every rule below is a scar.

  DEAD          no new metric for N minutes — the run has hung, and quota is still burning
  DIVERGED      loss is NaN or has blown up — nothing after this point is worth paying for
  SEMANTIC      a language crossed the meaning floor — length bought with damage is not a
                win, and this project has already measured r = -0.58 between the two
  REGRESSION    a language fell below its own base-model score — Odia did exactly this
                (probe slope 0.574 -> 0.345) and nobody saw it until the run was over
  PLATEAU       the objective stopped moving while CE was already flat — the pre-registered
                stopping rule, applied on the metric that matters rather than on loss
  CLIMBING      CE flat but the objective still rising — keep spending, this is the whole
                finding of Phase 02 and the reason not to early-stop on loss
  QUOTA         projected GPU spend is about to exceed what is left this week

USAGE
-----
    python -m tools.run_watchdog --run 02j-budget-sweep --watch
    python -m tools.run_watchdog --run 02k-corrective --watch --decide --training
    python -m tools.run_watchdog --run 02k-corrective --baseline research/.../3801.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT_ENTITY = "nktthegreat-soccernet"
DEFAULT_PROJECT = "indic-dubbing-v3"

# ---------------------------------------------------------------------------------------
# Pre-registered thresholds and the action each one implies.
# ---------------------------------------------------------------------------------------
STALE_MINUTES = 25              # no new step for this long => the run is hung
SEMANTIC_FLOOR = 0.70           # per language; below this the output is not usable
DEGRADED_CEILING = 0.75         # fraction past the 0.80 gate
REGRESSION_MARGIN = 0.08        # how far below its own base a language may fall
PLATEAU_SLOPE_MOVE = 0.010      # objective movement that still counts as progress
PLATEAU_CE_MOVE = 0.005         # CE movement below which CE is "flat"
QUOTA_HEADROOM = 0.90           # fraction of remaining quota we allow a run to project to

# The default action taken in --decide mode. STOP means "kill the session"; the watchdog
# does not have that power itself, so it reports the decision and the exit code carries it.
VERDICT_ACTION = {
    "DEAD":       ("STOP",     "A hung run burns quota and produces nothing. Kill it, "
                               "check the last logged step, and re-push."),
    "DIVERGED":   ("STOP",     "Nothing produced after divergence is worth paying for."),
    "SEMANTIC":   ("STOP",     "Length control bought with meaning is not progress. This "
                               "project measured r = -0.58 between slope and semantics; a "
                               "run that crosses the floor is optimising the wrong side of "
                               "that trade."),
    "REGRESSION": ("ALERT",    "One language is getting worse while others improve. Not "
                               "automatically fatal — it may be the sampling mix — but it "
                               "needs a human before the run is trusted."),
    "PLATEAU":    ("STOP",     "CE flat AND the objective flat is the pre-registered "
                               "stopping condition. Further steps cost quota and buy "
                               "nothing measurable."),
    "QUOTA":      ("ALERT",    "Projected spend exceeds what is left this week. Decide "
                               "whether to shorten the run or accept the overrun."),
    "CLIMBING":   ("CONTINUE", "CE is flat but the objective is still rising. This is the "
                               "Phase-02 finding: do not early-stop on loss."),
    "HEALTHY":    ("CONTINUE", "Nothing triggered."),
}


def _api():
    try:
        import wandb
    except ImportError:
        raise SystemExit("wandb not installed:  pip install wandb")
    try:
        return wandb.Api()
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"W&B authentication failed: {e}\n  Run once: wandb login")


def resolve(api, name: str, entity: str, project: str):
    runs = list(api.runs(f"{entity}/{project}", filters={"display_name": name},
                         order="-created_at"))
    if not runs:
        runs = [r for r in api.runs(f"{entity}/{project}", order="-created_at")
                if name in (r.name or "") or name == r.id]
    if not runs:
        raise SystemExit(f"no run matching {name!r} in {entity}/{project}")
    return runs[0]


def latest_metrics(run) -> tuple[dict, int]:
    """Most recent non-null value of every series, plus how many rows we saw."""
    hist = list(run.scan_history())
    latest: dict = {}
    for row in hist:
        for k, v in row.items():
            if v is not None and not (isinstance(v, float) and v != v):
                latest[k] = v
    return latest, len(hist)


def _per_lang(latest: dict, suffix: str) -> dict[str, float]:
    out = {}
    for k, v in latest.items():
        if k.endswith("/" + suffix) and isinstance(v, (int, float)):
            lang = k.split("/")[0]
            if len(lang) == 2:
                out[lang] = float(v)
    return out


def evaluate(run, latest: dict, history_len: int, baseline: Optional[dict],
             quota_left_h: Optional[float], started: Optional[float]) -> list[dict]:
    """Applies every rule. Returns the triggered ones, most severe first."""
    fired: list[dict] = []

    # DEAD — the run is still marked running but nothing has arrived.
    hb = run.summary.get("_timestamp") or latest.get("_timestamp")
    if run.state == "running" and hb:
        stale_min = (time.time() - float(hb)) / 60.0
        if stale_min > STALE_MINUTES:
            fired.append({"verdict": "DEAD",
                          "detail": f"no new metric for {stale_min:.0f} min "
                                    f"(limit {STALE_MINUTES})"})

    # DIVERGED
    for k in ("train/loss", "eval/loss", "eval_loss", "agg/ce_mean"):
        v = latest.get(k)
        if isinstance(v, (int, float)) and (v != v or v > 10.0):
            fired.append({"verdict": "DIVERGED", "detail": f"{k} = {v}"})
            break

    # SEMANTIC — per language, because the aggregate hides exactly this.
    sem = _per_lang(latest, "semantic_mean")
    bad = {lg: v for lg, v in sem.items() if v < SEMANTIC_FLOOR}
    deg = _per_lang(latest, "semantic_degraded_frac")
    bad_deg = {lg: v for lg, v in deg.items() if v > DEGRADED_CEILING}
    if bad or bad_deg:
        fired.append({"verdict": "SEMANTIC",
                      "detail": (f"below {SEMANTIC_FLOOR}: {bad} " if bad else "")
                                + (f"degraded above {DEGRADED_CEILING}: {bad_deg}"
                                   if bad_deg else "")})

    # REGRESSION — against a stored per-language baseline, not against the aggregate.
    if baseline:
        cur = _per_lang(latest, "slope_normalized") or _per_lang(latest, "length_slope_probe")
        worse = {lg: (v, baseline[lg]) for lg, v in cur.items()
                 if lg in baseline and v < baseline[lg] - REGRESSION_MARGIN}
        if worse:
            fired.append({"verdict": "REGRESSION",
                          "detail": "; ".join(f"{lg} {v:.3f} vs base {b:.3f}"
                                              for lg, (v, b) in sorted(worse.items()))})

    # PLATEAU / CLIMBING — needs at least two evaluation points.
    hist = list(run.scan_history())
    slope_key = next((k for k in ("agg/length_slope_normalized", "agg/length_slope_probe",
                                  "probe/length_slope") if k in latest), None)
    ce_key = next((k for k in ("agg/ce_mean", "eval/loss", "eval_loss") if k in latest), None)
    if slope_key and ce_key:
        pts = [(r.get("_step"), r.get(slope_key), r.get(ce_key)) for r in hist
               if r.get(slope_key) is not None and r.get(ce_key) is not None]
        if len(pts) >= 2:
            (_, s0, c0), (_, s1, c1) = pts[-2], pts[-1]
            ce_flat = abs(c1 - c0) < PLATEAU_CE_MOVE
            slope_move = s1 - s0
            if ce_flat and abs(slope_move) < PLATEAU_SLOPE_MOVE:
                fired.append({"verdict": "PLATEAU",
                              "detail": f"CE flat ({c1 - c0:+.4f}) and objective flat "
                                        f"({slope_move:+.4f})"})
            elif ce_flat and slope_move >= PLATEAU_SLOPE_MOVE:
                fired.append({"verdict": "CLIMBING",
                              "detail": f"CE flat ({c1 - c0:+.4f}) but objective "
                                        f"{slope_move:+.4f} — keep going"})

    # QUOTA — Kaggle's T4x2 bills both GPUs, so wall clock counts double.
    if quota_left_h and started:
        elapsed_h = (time.time() - started) / 3600.0
        projected = elapsed_h * 2.0
        if projected > quota_left_h * QUOTA_HEADROOM:
            fired.append({"verdict": "QUOTA",
                          "detail": f"{elapsed_h:.1f}h wall = {projected:.1f} GPU-h "
                                    f"against {quota_left_h:.1f}h left"})

    order = ["DEAD", "DIVERGED", "SEMANTIC", "PLATEAU", "REGRESSION", "QUOTA", "CLIMBING"]
    fired.sort(key=lambda f: order.index(f["verdict"]))
    return fired


def show(run, latest: dict, fired: list[dict]) -> None:
    print(f"\n=== {run.name} [{run.state}] {run.url}")
    langs = sorted({k.split("/")[0] for k in latest
                    if "/" in k and len(k.split("/")[0]) == 2})
    cols = [("slope_normalized", "slope(0=ignores)", 3),
            ("length_slope_probe", "slope(pooled)", 3),
            ("semantic_mean", "semantic", 3),
            ("semantic_degraded_frac", "degraded", 3),
            ("adherence_rel_mean", "relErr", 3),
            ("harvested_rows", "harvested", 0),
            ("ce_mean", "CE (diagnostic)", 4)]
    live = [c for c in cols if any(f"{lg}/{c[0]}" in latest for lg in langs)]
    if langs and live:
        hdr = f"  {'lang':<6}" + "".join(f"{lbl:>18}" for _, lbl, _ in live)
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for lg in langs:
            line = f"  {lg:<6}"
            for key, _, prec in live:
                v = latest.get(f"{lg}/{key}")
                line += f"{v:>18.{prec}f}" if isinstance(v, (int, float)) else f"{'-':>18}"
            print(line)
    for k in ("agg/length_slope_normalized", "agg/length_slope_probe", "agg/ce_mean",
              "train/loss", "languages_done"):
        if k in latest:
            note = "     <- the objective" if "slope" in k else (
                "     <- diagnostic, NOT the objective" if "ce" in k or "loss" in k else "")
            print(f"  {k:<32}{latest[k]}{note}")

    if not fired:
        print("\n  HEALTHY — nothing triggered.")
        return
    print()
    for f in fired:
        action, why = VERDICT_ACTION[f["verdict"]]
        print(f"  [{f['verdict']}] -> {action}")
        print(f"      {f['detail']}")
        print(f"      rationale: {why}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--entity", default=DEFAULT_ENTITY)
    p.add_argument("--project", default=DEFAULT_PROJECT)
    p.add_argument("--watch", action="store_true")
    p.add_argument("--interval", type=int, default=180)
    p.add_argument("--max_polls", type=int, default=200)
    p.add_argument("--decide", action="store_true",
                   help="Apply the pre-registered action instead of waiting for a human.")
    p.add_argument("--baseline", default=None,
                   help="JSON {lang: slope} to detect per-language regression against.")
    p.add_argument("--quota_left_h", type=float, default=None)
    p.add_argument("--log", default=None, help="Append every verdict here, with rationale.")
    args = p.parse_args()

    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8")) \
        if args.baseline else None

    api = _api()
    run = resolve(api, args.run, args.entity, args.project)
    started = None
    try:
        started = datetime.fromisoformat(str(run.created_at)).replace(
            tzinfo=timezone.utc).timestamp()
    except Exception:  # noqa: BLE001
        pass

    for i in range(args.max_polls if args.watch else 1):
        try:
            run.load(force=True)
            latest, n = latest_metrics(run)
            fired = evaluate(run, latest, n, baseline, args.quota_left_h, started)
            show(run, latest, fired)

            actions = {VERDICT_ACTION[f["verdict"]][0] for f in fired}
            record = {"utc": datetime.now(timezone.utc).isoformat(), "run": run.name,
                      "state": run.state, "fired": fired,
                      "actions": sorted(actions) or ["CONTINUE"]}
            if args.log:
                with open(args.log, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")

            if "STOP" in actions:
                print("\n  DECISION: STOP" if args.decide else
                      "\n  NEEDS A DECISION: the pre-registered action is STOP")
                return 3
            if "ALERT" in actions and not args.decide:
                print("\n  NEEDS A DECISION: see the rationale above")
                return 4
            if run.state not in ("running", "pending"):
                print(f"\n  run finished ({run.state}).")
                return 0
        except Exception as e:  # noqa: BLE001
            print(f"  poll error ({type(e).__name__}: {e}) — retrying")
        if not args.watch:
            break
        time.sleep(args.interval)

    print("\n  still running; re-run to keep watching")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
