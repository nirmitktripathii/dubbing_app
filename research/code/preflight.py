"""
tools/preflight.py
==================
The gate every GPU session must pass **before it is pushed**.

WHY THIS EXISTS
---------------
Phase 02 spent two weeks and most of a GPU quota on runs whose only product was the
discovery that an earlier run had been invalid. Look at what each discovery actually
required to find:

| discovery | how it was found | what it needed |
|---|---|---|
| corpus labelled in characters, not phonemes | recount the labels, compare | CPU, seconds |
| `train.jsonl` uniformly char-ruled, not mixed | count `frac_label_eq_chars` | CPU, seconds |
| 0.2% of rows carry an off-natural budget | group by (language, english) | CPU, seconds |
| augmentation missing `or`, `pa`, `te` entirely | group by language | CPU, seconds |
| augmentation 69% expand / 31% compress | group by direction | CPU, seconds |
| chrF++ silently returning `None` for 440 rows | import at module scope | CPU, instantly |
| the 11 languages split into two populations | generate and measure | **GPU, hours** |

Exactly one of those needed a GPU. Every other one was a property of a file on disk that
nobody had asked the file about — and each of them invalidated a GPU run already paid for.

So the rule this module enforces is not "test more". It is:

    A GPU session may only be spent on questions that CANNOT be answered on CPU.
    Everything else is a precondition, and preconditions are checked before you pay.

TWO STAGES, BECAUSE THE CHECKS HAVE DIFFERENT HOMES
----------------------------------------------------
`--stage local` runs on the dev machine before `--push`. It needs no espeak and no GPU:
every check is structural, answerable by grouping and counting.

`--stage kaggle` runs inside the notebook, in the first cell, before any model loads. It
adds the checks that need espeak (which is installed there and not here) and aborts the
session in seconds rather than discovering the problem after four hours of generation.

A CHECK THAT CANNOT RUN IS NOT A CHECK THAT PASSED
---------------------------------------------------
Results are three-state: PASS, FAIL, UNKNOWN. UNKNOWN blocks by default, because "I could
not verify this" and "this is fine" are the same outcome only to a process that is trying
to talk itself into launching. `--allow_unknown` exists for the case where you have
genuinely decided to proceed unverified, and it makes that decision explicit and logged.

USAGE
-----
    python -m tools.preflight --stage local  --corpus data/train.jsonl --val data/val.jsonl
    python -m tools.preflight --stage kaggle --corpus /kaggle/working/data/train.jsonl
    python -m tools.preflight --stage local  --corpus ... --json report.json

Exit 0 = safe to spend GPU. Non-zero = do not push.
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import os
import re
import statistics as st
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.languages import LANGUAGES  # noqa: E402

logger = logging.getLogger("preflight")

PROMPT_N = re.compile(r"\[Target Phonemes:\s*(\d+)\]")

# ---------------------------------------------------------------------------------------
# Pre-registered thresholds. They live in code so they are decided while nothing is at
# stake, rather than at 2am with a session already queued and a deadline in view.
# ---------------------------------------------------------------------------------------
MIN_ELASTIC_ROW_FRAC = 0.15     # rows sitting in a group that spans >1 budget
MIN_ELASTIC_SPREAD = 1.25       # median max/min budget within those groups
MIN_COMPRESS_SHARE = 0.40       # of off-base rows, how many ask for LESS
MIN_ROWS_PER_LANG = 200
MIN_ELASTIC_GROUPS_PER_LANG = 50

PASS, FAIL, UNKNOWN = "PASS", "FAIL", "UNKNOWN"


class Check:
    def __init__(self, name: str, state: str, detail: str,
                 data: Optional[dict] = None, blocking: bool = True):
        self.name, self.state, self.detail = name, state, detail
        self.data, self.blocking = data or {}, blocking

    @property
    def ok(self) -> bool:
        return self.state == PASS

    def line(self) -> str:
        mark = {PASS: "PASS", FAIL: "FAIL", UNKNOWN: "????"}[self.state]
        if not self.blocking and self.state != PASS:
            mark = "warn"
        return f"  [{mark}] {self.name:<12} {self.detail}"


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ========================================================================================
# Structural checks — no espeak, no GPU. These run on the dev machine.
# ========================================================================================

def _groups(rows: list[dict]) -> dict[tuple, list[int]]:
    """(language, english) -> the budgets that input is seen with."""
    g = collections.defaultdict(list)
    for r in rows:
        try:
            g[(r.get("language"), r.get("english"))].append(int(r.get("n_phonemes", -1)))
        except (TypeError, ValueError):
            continue
    return g


def check_elasticity(rows: list[dict], label: str) -> Check:
    """THE check Phase 02 did not have.

    A model learns to read the budget only if the budget is not deducible from the input.
    Samanantar gives ONE reference translation per English sentence, so the budget written
    into that row's prompt is that translation's own length — a deterministic function of
    the English. On such a row, "ignore the number, translate naturally" is a fully correct
    answer, and gradient descent has no reason to prefer a model that reads the number.

    The signal exists only where the SAME input appears with DIFFERENT budgets. That is
    what makes the number load-bearing.

    Measured on the corpus that trained checkpoint-3801: 0.2% of rows. Malayalam's probe
    slope then moved 0.419 -> 0.418 across 3,801 steps. That is not slow learning; that is
    a language that was never shown the task.
    """
    g = _groups(rows)
    if not g:
        return Check("elasticity", UNKNOWN, f"{label}: no (language, english) groups found")
    multi = {k: v for k, v in g.items() if len(set(v)) > 1}
    n_rows = sum(len(v) for v in g.values())
    rows_in_multi = sum(len(v) for v in multi.values())
    frac = rows_in_multi / n_rows if n_rows else 0.0

    spread = 0.0
    if multi:
        spreads = sorted(max(v) / min(v) for v in multi.values() if min(v) > 0)
        spread = st.median(spreads) if spreads else 0.0

    ok = frac >= MIN_ELASTIC_ROW_FRAC and spread >= MIN_ELASTIC_SPREAD
    detail = (f"{label}: {frac:.1%} of rows sit in a group spanning >1 budget "
              f"(floor {MIN_ELASTIC_ROW_FRAC:.0%}), median spread {spread:.2f}x "
              f"(floor {MIN_ELASTIC_SPREAD:.2f}x)")
    if not ok:
        detail += ("\n               <-- the budget is deducible from the English on "
                   "almost every row, so ignoring it is never penalised")
    return Check("elasticity", PASS if ok else FAIL, detail,
                 {"elastic_row_frac": frac, "median_spread": spread,
                  "n_multi_groups": len(multi), "n_groups": len(g)})


def check_direction(rows: list[dict], label: str) -> Check:
    """Dubbing mostly needs COMPRESSION. The base model overshoots by 34% (signed +0.344),
    so at dub time the operation asked for is almost always "say this in fewer sounds".
    The augmentation that existed was 737 expand / 327 compress: it trained the easy
    direction and under-trained the one the product needs.

    Within each elastic group the baseline is the SMALLEST budget's sibling... no — the
    baseline is the un-augmented row where one exists, else the group median. Every other
    row is then classified against it.
    """
    by_group = collections.defaultdict(list)
    for r in rows:
        by_group[(r.get("language"), r.get("english"))].append(r)

    comp = exp = 0
    for _, grp in by_group.items():
        if len(grp) < 2:
            continue
        budgets = [int(x.get("n_phonemes", -1)) for x in grp]
        if len(set(budgets)) < 2:
            continue
        base_rows = [x for x in grp if not x.get("augmentation")]
        base = (int(base_rows[0].get("n_phonemes", -1)) if base_rows
                else st.median(budgets))
        for x in grp:
            if base_rows and x in base_rows:
                continue
            n = int(x.get("n_phonemes", -1))
            if n < base * 0.98:
                comp += 1
            elif n > base * 1.02:
                exp += 1

    total = comp + exp
    if not total:
        return Check("direction", FAIL,
                     f"{label}: no off-baseline rows at all — nothing teaches either "
                     f"direction")
    share = comp / total
    ok = share >= MIN_COMPRESS_SHARE
    return Check("direction", PASS if ok else FAIL,
                 f"{label}: {share:.0%} of off-baseline rows ask for LESS "
                 f"({comp} compress / {exp} expand, floor {MIN_COMPRESS_SHARE:.0%})",
                 {"compress_share": share, "n_compress": comp, "n_expand": exp})


def check_coverage(rows: list[dict], label: str) -> Check:
    """All 11, and — for a training corpus — all 11 with elastic groups. The augmentation
    that existed covered 7 languages and omitted `or`, `pa` and `te`: three of the five
    later found stuck. The languages that most needed the lesson were the ones left out.

    A validation set is measured, never trained on, so it needs presence and enough rows to
    estimate from, not elasticity. Applying the training thresholds to it would be the same
    category error as reading `adherence_rel_mean` as the objective.
    """
    is_val = label == "val"
    min_rows = 100 if is_val else MIN_ROWS_PER_LANG

    per_lang = collections.Counter(r.get("language") for r in rows)
    g = _groups(rows)
    elastic_per_lang = collections.Counter(k[0] for k, v in g.items() if len(set(v)) > 1)

    problems = []
    missing = sorted(set(LANGUAGES) - set(per_lang))
    if missing:
        problems.append(f"absent: {missing}")
    thin = sorted(lg for lg in LANGUAGES if 0 < per_lang.get(lg, 0) < min_rows)
    if thin:
        problems.append(f"under {min_rows} rows: {thin}")
    if not is_val:
        no_elastic = sorted(lg for lg in LANGUAGES
                            if elastic_per_lang.get(lg, 0) < MIN_ELASTIC_GROUPS_PER_LANG)
        if no_elastic:
            problems.append(
                f"under {MIN_ELASTIC_GROUPS_PER_LANG} elastic groups: {no_elastic}")

    ok_msg = (f"{label}: all {len(LANGUAGES)} languages present, >={min_rows} rows each"
              + ("" if is_val else " and elastic"))
    return Check("coverage", PASS if not problems else FAIL,
                 ok_msg if not problems else f"{label}: " + "; ".join(problems),
                 {"per_lang": dict(per_lang), "elastic_per_lang": dict(elastic_per_lang)})


def check_prompt(rows: list[dict], label: str) -> Check:
    """The model trains on the prompt; the metric reads the field. A corpus where those
    disagree is inconsistent in a way no metric will ever report."""
    bad = checked = 0
    for r in rows:
        p = r.get("prompt")
        if not p:
            continue
        checked += 1
        m = PROMPT_N.search(p)
        if not m or int(m.group(1)) != int(r.get("n_phonemes", -1)):
            bad += 1
    if not checked:
        return Check("prompt", UNKNOWN, f"{label}: no prompts present")
    return Check("prompt", PASS if bad == 0 else FAIL,
                 f"{label}: {checked - bad}/{checked} prompts agree with n_phonemes",
                 {"checked": checked, "bad": bad})


def check_ruler(rows: list[dict], label: str) -> Check:
    from common.phonemes import RULER_PHONEMES
    rulers = collections.Counter(r.get("ruler", "MISSING") for r in rows)
    if len(rulers) != 1:
        return Check("ruler", FAIL,
                     f"{label}: {len(rulers)} distinct rulers {dict(rulers)} — a mixed "
                     f"corpus teaches two contradictory tasks under one prompt token",
                     {"rulers": dict(rulers)})
    only = next(iter(rulers))
    if not str(only).startswith(RULER_PHONEMES):
        return Check("ruler", FAIL,
                     f"{label}: ruler is {only!r}, not a phoneme ruler — run "
                     f"tools/relabel_dataset.py first", {"ruler": only})

    # Version skew is the original bug wearing a different hat. espeak-ng 1.50 and 1.52
    # do not have to agree on phoneme counts, so labels written by one and measured by the
    # other are mismatched units all over again — and the prefix check above passes
    # happily. Kaggle's apt gives 1.50; a winget install on Windows gives 1.52.0.
    try:
        from common.phonemes import ruler_id
        here = ruler_id()
    except Exception:  # noqa: BLE001
        here = None
    if here and here != only:
        # Non-blocking, and deliberately so. Version skew is a *proxy* for "the counts might
        # disagree", and `labels` tests that directly by recounting every sampled row. A
        # proxy that fires while the direct test passes should inform, not block — the
        # alternative is a red light nobody can clear, which is how gates get switched off.
        # Measured 2026-08-05: espeak-ng 1.50 and 1.52.0 agree on 550/550 rows across all
        # eleven languages, so this skew is cosmetic. Do not assume that for a new version;
        # let the labels check say so.
        return Check("ruler", PASS,
                     f"{label}: labelled with {only!r}, this machine counts with {here!r} "
                     f"— version skew, settled by the `labels` check below rather than "
                     f"assumed either way",
                     {"corpus_ruler": only, "local_ruler": here, "version_skew": True},
                     blocking=False)
    return Check("ruler", PASS, f"{label}: uniform {only!r} across {len(rows)} rows"
                 + (f", matching this machine" if here else ""),
                 {"ruler": only, "local_ruler": here})


# ========================================================================================
# Checks that need espeak — these run inside the Kaggle notebook, before any model loads.
# ========================================================================================

def check_g2p(_rows=None, _label=None) -> Check:
    """Not "is a phonemizer importable" — that passed in the session that mislabelled the
    whole corpus. This asserts the output symbols are demonstrably NOT the input's own
    characters."""
    try:
        from common.phonemes import assert_g2p_available
        info = assert_g2p_available()
        return Check("g2p", PASS, f"ruler={info['ruler']} — output is not the input", info)
    except ImportError as e:
        return Check("g2p", UNKNOWN, f"cannot import the counter: {e}")
    except Exception as e:  # noqa: BLE001
        msg = str(e).splitlines()[0]
        state = UNKNOWN if "not importable" in str(e) else FAIL
        return Check("g2p", state, f"{type(e).__name__}: {msg}")


def check_labels(rows: list[dict], label: str, sample: int = 3000) -> Check:
    """The stated count must reproduce from the row's own completion. This is the check
    that would have caught the character ruler on day one: it does not ask whether the
    phonemizer ran, it asks whether the number in the file is the number the counter
    produces."""
    try:
        from common.phonemes import count_phonemes
    except ImportError as e:
        return Check("labels", UNKNOWN, f"counter unavailable: {e}")

    step = max(1, len(rows) // sample)
    checked = bad = 0
    worst = []
    for r in rows[::step]:
        comp = r.get("completion") or r.get("target")
        lang = r.get("language")
        if not comp or lang not in LANGUAGES:
            continue
        try:
            true_n = count_phonemes(comp, lang)
        except Exception as e:  # noqa: BLE001
            return Check("labels", UNKNOWN,
                         f"{label}: cannot recount ({type(e).__name__}: "
                         f"{str(e).splitlines()[0]})")
        checked += 1
        if true_n != int(r.get("n_phonemes", -1)):
            bad += 1
            if len(worst) < 3:
                worst.append(f"{lang}: says {r.get('n_phonemes')}, counts {true_n}")
    if not checked:
        return Check("labels", UNKNOWN, f"{label}: nothing checkable")
    return Check("labels", PASS if bad == 0 else FAIL,
                 f"{label}: {checked - bad}/{checked} labels reproduce exactly"
                 + (f" — e.g. {'; '.join(worst)}" if worst else ""),
                 {"checked": checked, "bad": bad})


# ========================================================================================
# Environment
# ========================================================================================

def check_wandb(entity: str = "nktthegreat-soccernet") -> Check:
    """W&B is the only channel that reports while a Kaggle session runs, so a session
    launched without it is unobservable until it ends. 02i burned ~12 GPU-hours that way."""
    # `_netrc` on Windows, `.netrc` elsewhere — wandb picks by platform.
    if not (os.environ.get("WANDB_API_KEY") or
            any((Path.home() / n).exists() for n in (".netrc", "_netrc"))):
        return Check("wandb", FAIL,
                     "no WANDB_API_KEY and no ~/.netrc or ~/_netrc — the run would be blind")
    try:
        import wandb
        list(wandb.Api().projects(entity))
        return Check("wandb", PASS, f"{entity} reachable")
    except ImportError:
        return Check("wandb", UNKNOWN, "wandb not installed (pip install wandb)")
    except Exception as e:  # noqa: BLE001
        return Check("wandb", FAIL, f"cannot reach {entity}: {str(e).splitlines()[0]}")


def check_checkpoints(paths: list[str]) -> Check:
    """A checkpoint with weights but no `scheduler.pt` resumes at the wrong learning rate
    and looks perfectly healthy doing it."""
    need = ["adapter_model.safetensors", "adapter_config.json"]
    resume = ["scheduler.pt", "trainer_state.json"]
    problems = []
    for p in paths:
        d = Path(p)
        if not d.is_dir():
            problems.append(f"{d.name}: not a directory")
            continue
        for f in need:
            if not (d / f).exists():
                problems.append(f"{d.name}: missing {f}")
        gone = [f for f in resume if not (d / f).exists()]
        if gone:
            problems.append(f"{d.name}: cannot resume — missing {gone}")
    return Check("checkpoints", PASS if not problems else FAIL,
                 f"{len(paths)} checkpoint(s) complete" if not problems
                 else "; ".join(problems))


# ========================================================================================

STRUCTURAL = {"elasticity": check_elasticity, "direction": check_direction,
              "coverage": check_coverage, "prompt": check_prompt, "ruler": check_ruler}
G2P_DEPENDENT = {"labels": check_labels}

STAGES = {
    # Before pushing. No espeak, no GPU — every one of these is grouping and counting.
    "local": ["elasticity", "direction", "coverage", "prompt"],
    # Inside the notebook, first cell, before a model is loaded.
    "kaggle": ["ruler", "labels", "elasticity", "coverage", "prompt"],
    "all": ["ruler", "labels", "elasticity", "direction", "coverage", "prompt"],
}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=sorted(STAGES), default="local")
    p.add_argument("--corpus", default=None, help="Training corpus JSONL.")
    p.add_argument("--val", default=None, help="Validation corpus JSONL.")
    p.add_argument("--checkpoints", nargs="*", default=[])
    p.add_argument("--require", nargs="*", default=None,
                   help="Override the stage's check list.")
    p.add_argument("--warn_only", nargs="*", default=[],
                   help="Run these but do not block on them.")
    p.add_argument("--allow_unknown", action="store_true",
                   help="Let UNKNOWN pass. Makes 'I could not verify this' an explicit, "
                        "logged decision instead of a silent one.")
    p.add_argument("--no_wandb_check", action="store_true")
    p.add_argument("--json", default=None)
    args = p.parse_args()

    checks: list[Check] = []
    wanted = args.require or STAGES[args.stage]

    print(f"\nPreflight — stage {args.stage!r}")
    print("Environment")
    if args.stage == "kaggle" or "labels" in wanted:
        checks.append(check_g2p())
    if not args.no_wandb_check:
        checks.append(check_wandb())
    if args.checkpoints:
        checks.append(check_checkpoints(args.checkpoints))
    for c in checks:
        print(c.line())

    for label, path in (("train", args.corpus), ("val", args.val)):
        if not path:
            continue
        rows = load_jsonl(Path(path))
        print(f"\n{label}: {path}  ({len(rows)} rows)")
        for name in wanted:
            fn = STRUCTURAL.get(name) or G2P_DEPENDENT.get(name)
            if fn is None:
                print(f"  [....] unknown check {name!r}")
                continue
            c = fn(rows, label)
            # A validation set is measured, never trained on: elasticity and direction are
            # properties the TRAINING corpus must have, not this one.
            if label == "val" and name in ("elasticity", "direction"):
                c.blocking = False
            if name in args.warn_only:
                c.blocking = False
            checks.append(c)
            print(c.line())

    blocking = [c for c in checks if c.blocking and c.state == FAIL]
    unknown = [c for c in checks if c.blocking and c.state == UNKNOWN]
    warned = [c for c in checks if not c.blocking and c.state != PASS]

    print()
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"stage": args.stage, "ok": not blocking and not (unknown and not args.allow_unknown),
             "checks": [{"name": c.name, "state": c.state, "blocking": c.blocking,
                         "detail": c.detail, "data": c.data} for c in checks]},
            indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"report -> {args.json}")

    if warned:
        print(f"{len(warned)} non-blocking:")
        for c in warned:
            print(f"    {c.name}: {c.detail}")

    if unknown and not args.allow_unknown:
        print(f"\nPREFLIGHT BLOCKED — {len(unknown)} check(s) could not run:")
        for c in unknown:
            print(f"    {c.name}: {c.detail}")
        print("\nA check that cannot run is not a check that passed. Fix the tooling, or\n"
              "pass --allow_unknown to record that you decided to proceed unverified.")
        return 2

    if blocking:
        print(f"PREFLIGHT FAILED — {len(blocking)} blocking issue(s):")
        for c in blocking:
            print(f"    {c.name}: {c.detail}")
        print("\nDo NOT push. Every one of these is answerable here, on CPU, for free.\n"
              "A GPU session is for questions that cannot be answered any other way.")
        return 1

    print("PREFLIGHT PASSED — safe to spend GPU.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
