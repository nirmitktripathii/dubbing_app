# Operating rules for this repository

These are standing constraints, not suggestions. Each one exists because ignoring it
already cost this project time or GPU quota. If a rule seems to be in the way, that is
usually the moment it is doing its job.

---

## 1. A GPU session is only for questions that cannot be answered on CPU

Phase 02 spent two weeks and most of a quota on runs whose only product was the discovery
that an earlier run had been invalid. Of the seven defects found in that period, **one**
needed a GPU. The rest were properties of files on disk that nobody had asked the files
about.

**Before any push:**

```bash
python -m tools.preflight --stage local --corpus <train.jsonl> --val <val.jsonl>
```

Exit code 0 or do not push. The Kaggle-side twin (`--stage kaggle`) runs inside the
notebook's first cells, before any model loads, so a bad corpus costs seconds and not
hours.

A check that *cannot run* is not a check that passed. `preflight` returns a distinct exit
code for UNKNOWN. Use `--allow_unknown` only to record an explicit decision to proceed
unverified — never to make the red go away.

## 2. Every GPU run must leave behind an artifact the next run consumes

A run whose only output is knowledge is a run that has to be followed by another run before
anything improves — which is how "two weeks on Step 02" happens. Design each session so
that even a partial or disappointing result yields data the next step can use. Write
results incrementally, never only at the end, so an interrupted session is still worth what
it cost.

## 3. W&B is mandatory, live, per-language, and watched

Kaggle publishes a notebook's log **only when the session ends**. During a multi-hour run
W&B is the only channel that reports anything at all.

- Entity `nktthegreat-soccernet`, project `indic-dubbing-v3`. The personal namespace
  `nktthegreat` holds zero projects; logging there sends metrics where nobody looks.
- Pass `--wandb --wandb_required` on every Kaggle run. A missing credential must abort
  **before** a model loads. Session 02i burned ~12 GPU-hours with no live channel because a
  secrets-service outage was only a warning.
- Log **per language**, not just aggregates. Eleven languages across two families do not
  plateau together, and every aggregate in this project so far has hidden something.
- Lead with the objective. Cross-entropy and perplexity are diagnostics and must be
  labelled as such.

**And it must be watched, not merely written.** While any run is live:

```bash
python -m tools.run_watchdog --run <name> --watch --baseline <per-lang baseline.json>
```

The watchdog applies pre-registered rules (hung run, divergence, semantic collapse,
per-language regression, plateau, quota overrun) and returns a verdict with its rationale.
Add `--decide` to apply the pre-registered action when nobody is available to be asked.
Raise anything needing a human immediately; do not wait for the run to finish.

## 4. Quota arithmetic: Kaggle T4x2 bills two GPU-hours per wall-clock hour

Measured 2026-08-04: 5h45m wall clock consumed ~12.1 of 30 weekly GPU-hours. The 8B model
in 4-bit fits on one T4 and Unsloth uses `cuda:0`, so the second GPU is idle and billed
anyway. There is no single-T4 shape; P100 is sm_60 and cannot run this torch build.

**Plan every session against `remaining quota / 2`.** Give every long job a hard
`--time_budget_s`. Check with `python -m tools.kaggle_session --quota` before launching —
quota is billed on completion, so an in-flight run does not appear until it ends.

## 5. Measure the property, never the machinery

Every silent failure in this project passed a check that tested whether code *ran* instead
of whether the output had the property it needed:

| the check that passed | what was actually true |
|---|---|
| "the phonemizer works" — it returned output | it had fallen back to returning the input's own characters |
| "the labels look sane" — letters ≈ sounds on average | per-row error 5.7–11.3% |
| "sacrebleu scored it" — no exception | the import was inside the `try`; 440 silent `None`s |
| "the slope is rising" — a slope was computed | it was the confounded estimator |
| "slope 0.35 means partial obedience" | 0 was not the floor; the floor was 0.60–0.83 |

Assert the property. Not "did `phonemize()` return something" but "are these symbols
demonstrably not the input's own letters". Not "is there a checkpoint" but "did the step
counter advance".

## 6. Know which estimator you are reading

- `length_slope_normalized` — within sentence, `ratio = a + k·scale`. **Zero-referenced:
  0 = ignores the budget, 1 = follows it.** This is the objective.
- `length_slope_probe` — the older pooled fit on raw lengths. Sentence-length variance
  leaks in, so its floor is 0.60–0.83 per language, not 0. Never read against zero; the
  harness now reports `length_slope_probe_floor` beside it.
- `length_slope_population` — regressed across *different* sentences. Reported for
  comparison only. Never select on it.
- `adherence_rel_mean` — a fluency check, not the objective. In the adherence test the
  requested budget is the reference's own length, so a model that ignores the budget still
  scores well.

## 7. Do not trade meaning for length control silently

r(probe slope, semantic) = **−0.58** across the eleven languages. A global push on length
buys slope out of meaning, and the languages that already obey are at or past the 0.80
semantic gate. Any change that raises length adherence must report the semantic axis beside
it, per language.

## 8. Do not change the sampling ratios

Standing instruction from the project owner: the languages that are overfitting need *less*
fine-tuning. Reweighting toward the struggling languages risks catastrophic forgetting in
the ones that work.

## 9. Pre-register decisions

Thresholds and the actions they imply are written in code before the run —
`tools/preflight.py`, `evaluation/response_diagnosis.py`, `tools/run_watchdog.py`, and
`stopping_verdict()`. This is so the rule still applies at hour four, when stopping feels
expensive and "let it finish" is the tempting answer. Changing a threshold after seeing a
result is allowed; doing it silently is not.

## 10. Notebooks are built, not edited

`files_v3/kaggle_runs/build_*.py` reads modules from `pipeline_v3/` at build time. Hand-
editing a notebook cell is how the version that ran and the version in the repo diverge —
which has already happened once, leaving an unterminated f-string that would have killed a
session on its first cell. Every code cell must parse before the file is written.

---

## Key paths

| | |
|---|---|
| canonical phoneme counter | `pipeline_v3/common/phonemes.py` |
| the CPU gate | `pipeline_v3/tools/preflight.py` |
| budget-response sweep | `pipeline_v3/evaluation/budget_sweep.py` |
| pre-registered diagnosis + actions | `pipeline_v3/evaluation/response_diagnosis.py` |
| live run monitor | `pipeline_v3/tools/run_watchdog.py` |
| Kaggle push/poll/pull/quota | `pipeline_v3/tools/kaggle_session.py` |
| offline guards | `pipeline_v3/tests/test_phoneme_ruler.py` |
| phase plan and contingencies | `PHASE02_RUNBOOK.md` |
| published findings | `research/` |
