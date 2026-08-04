# Phase 02 Runbook — every branch decided in advance

The objective, the constraint the dataset imposes on it, and what happens next under every
outcome Step 1 can produce. Written before Step 1 runs, so that no branch gets chosen under
deadline pressure.

---

## 0. The objective, restated

Dubbed speech must fit the original speaker's time slot. Three points in the pipeline can
make that happen:

1. **Stretch the audio afterwards.** Cheap, and it sounds like it.
2. **Condition the TTS on a duration.** Better, but the TTS is being asked to say a fixed
   amount of content in the wrong amount of time.
3. **Constrain the text.** Ask for a translation that is *already* the right length in
   sounds. Nothing downstream has to compensate.

This project is (3). The deliverable is a model that, given an English sentence and a
phoneme budget, produces a faithful translation of approximately that many phonemes — for
eleven Indic languages, with meaning intact.

"Approximately that many phonemes" is measured by
`length_slope_normalized`: hold the sentence fixed, sweep the budget, regress
`produced/natural` on `scale`. **0 means the budget was ignored. 1 means it was followed.**

**Ship criterion:** slope ≥ 0.80 with semantic ≥ 0.80, per language. Not on average.

---

## 1. The constraint Samanantar imposes, and why it is structural

Samanantar is mined parallel text: for each English sentence, **one** Indic translation.
The phoneme budget written into a training prompt is therefore that translation's own
length — a deterministic function of the English input.

The consequence is not a bug in the pipeline. It is a property of every parallel corpus:

> On a row built this way, "ignore the number and translate naturally" is a completely
> correct answer. Gradient descent has no reason to prefer a model that reads the budget,
> because not reading it is never penalised.

Measured on the 53,349 rows that trained checkpoint 3801: **0.2%** of rows sit in a group
where the same English sentence appears with more than one budget, and those 59 groups are
incidental duplicates rather than designed. Malayalam's slope then moved 0.419 → 0.418
across 3,801 steps.

**No amount of Samanantar fixes this.** The lesson is not in any parallel corpus and never
will be — isochrony supervision has to be *manufactured*. That reframes the whole phase:
the corrective work is dataset engineering, not hyperparameter search.

### What we have that makes manufacturing it cheap

| asset | why it matters |
|---|---|
| `common/phonemes.py` | an **exact** verifier. Given any candidate, we know precisely how many phonemes it is. No reward model, no judge. |
| `translation/semantic_gate.py` | tells us whether a short candidate cheated by deleting content, using the same 0.80 threshold production uses |
| `training/length_augmentation.py` | already-built paraphrase augmenter with both gates wired |
| checkpoint 3801 | a model that already does this well in six languages — so its own outputs are a source of correct examples |

An exactly-checkable objective is rare. It means the missing data can be generated and
*verified* rather than trusted, which is what `evaluation/budget_sweep.py --harvest` does.

---

## 2. Where the phase actually stands

| | |
|---|---|
| corpus ruler | fixed — `phonemes:espeak-ng-1.50`, uniform, labels reproduce |
| measurement | fixed — chrF++ loud, probe/population separated, slope now zero-referenced |
| units salvage | **closed by measurement** — rescaling by phonemes-per-char moved the slope 0.001 |
| the model | six languages follow the budget; five are at or below the ignore-it floor |
| the corpus | **0.2% elastic** — this is the live defect |
| quota | ~17h weekly, refreshing Saturday; T4x2 bills 2x, so ~8.5h of wall clock |

---

## 3. Step 1 — the sweep (running now)

`02j_budget_sweep.ipynb` → `evaluation/budget_sweep.py`

11 languages × 40 sentences × 9 scales (0.4x–2.0x), every raw point saved, then a
best-of-4 harvest at compression budgets for whichever languages come back weak.
Hard stop at 5400s (~3 GPU-hours). Writes after every language.

Two outputs: `DIAGNOSIS.json` (which of four defects, per language) and
`candidates.jsonl` (verified elastic rows, usable as training data whatever the diagnosis
says).

---

## 4. The branches

`evaluation/response_diagnosis.py` classifies each language mechanically. The action for
each class was fixed before the run.

### 4a. FLAT — output length barely moves whatever you ask
**Meaning:** the budget is not being read. The degenerate-objective signature.
**Confidence this is the case:** high. It is what the corpus predicts.
**Action:** rebuild the corpus with elastic rows until ignoring the number stops being a
correct answer. Target ≥ 20% of rows in multi-budget groups, ≥ 40% of off-baseline rows
asking for *less*, all 11 languages, ≥ 50 elastic groups each. `tools/preflight --stage
local` enforces exactly these numbers.
**Sources, in cost order:** (1) harvested rows from Step 1 — free, already verified;
(2) `length_augmentation.py` re-run across all 11 languages with compression weighting;
(3) frontier-model paraphrase for languages where the base model's own output is too weak
to harvest from.
**Cost:** ~1h wall to build (CPU) + ~4h wall to fine-tune = ~8 GPU-h. Fits.
**Expected gain:** large. This is the cheapest failure mode to fix.

### 4b. ASYMMETRIC — one direction works, the other does not
**Meaning:** partially trained. Very plausible: the existing augmenter was 69% expand.
**Action:** as 4a, but weight the augmentation hard toward the missing direction. If the
missing direction is compression — the likely case, and the one dubbing needs — set the
compression share to 0.6 rather than the 0.4 floor.
**Cost:** same as 4a.

### 4c. SATURATING — tracks near natural, refuses to go far
**Meaning:** the budget is read, but there is a floor.
**The branch inside the branch:** compare `semantic_far_compress` against the 0.80 gate.
- *Meaning intact at the floor* → learned length prior. Fix with elastic rows **at the
  extremes** (0.5x, 0.6x), not more rows near 1.0x. Cost as 4a.
- *Meaning already breaking at the floor* → the floor is real: that language genuinely
  cannot say that content in that many sounds. **Do not train through it.** The honest fix
  is a per-language compression cap fed back into `duration_predictor.py`, so the pipeline
  stops asking for something impossible and lets stage (1) or (2) absorb the remainder.
  This is a *product* decision, and it is a legitimate outcome, not a failure.

### 4d. NOISY — length moves, but not with the request
**Meaning:** the model cannot estimate its own output length. The hardest case, because
more examples do not teach estimation.
**Action, in cost order:**
1. Rejection-sampled SFT — train only on generations that verifiably landed. This *is*
   length feedback, delivered as ordinary SFT rows. Step 1's harvest is the first slice.
2. Auxiliary length-prediction head, if (1) plateaus.
3. **Fall back to the deployed v2 path for that language.** `pipeline/isochrony_translation.py`
   already does generate-3-and-select with an exact counter, in production, today. A
   language routed there still ships.
**Note:** (3) is why this phase is an optimisation and not a blocker. There is a working
system; the fine-tune is here to make it cheaper and better, not to make it possible.

### 4e. OBEYS — already fine
**Action:** hold out of corrective training so it cannot regress, and keep in the eval set
as a forgetting canary. The standing constraint applies: **do not change the sampling
ratios**; the languages that work are the ones with the most to lose.

### 4f. WEAK / mixed — no clean signature
**Action:** do **not** guess and do not start a training run. Re-sweep that language alone
with more sentences and a wider range (~30 min). A wrong branch here costs 8 GPU-hours; a
second sweep costs 1.

---

## 5. Contingencies for the run itself

| if | then |
|---|---|
| smoke test fails | stop. It costs 90 seconds and has caught two sessions already. |
| W&B will not start | the run aborts by design (`--wandb_required`). Fix the credential, re-push. A blind session is not cheaper than no session. |
| session hits the 6h cap mid-sweep | fine. Results are written after every language; `DIAGNOSIS.json` covers whatever completed. Re-push with `--languages` for the rest. |
| harvest yields near zero for a language | that is itself a finding: the model cannot produce a correct short output even by chance, which rules out rejection sampling for that language and points at 4d(2) or 4d(3). |
| watchdog fires SEMANTIC | stop. Length bought with meaning is not progress. |
| watchdog fires REGRESSION | alert; check whether it is the sampling mix before doing anything. |
| quota runs out before Step 2 | Step 2's corpus build is **CPU-only**. Build and gate it while waiting for Saturday; the GPU is only needed for the fine-tune itself. |

---

## 6. Step 2 preconditions — all CPU, all enforced

Before any corrective fine-tune is pushed:

```bash
python -m tools.preflight --stage local --corpus data/train.elastic.jsonl --val data/val.phonemes.jsonl
```

must return 0, which requires: ruler uniform and phoneme-based; prompts agreeing with
labels; ≥15% of rows in multi-budget groups with median spread ≥1.25x; ≥40% of off-baseline
rows asking for less; all 11 languages with ≥200 rows and ≥50 elastic groups.

Then, and only then, the fine-tune — resumed from 3801, with the per-language baseline from
`research/evaluation/results/corrected_eval/` handed to the watchdog so a regression like
Odia's is caught in the first hour rather than the last.

---

## 7. Definition of done for Phase 02

Not "the loss went down". Not "the average improved".

> `length_slope_normalized ≥ 0.80` **and** `semantic_mean ≥ 0.80`, **per language**, on the
> held-out set, measured with the same counter that wrote the labels —
> **or** an explicit, recorded decision to route that language to the v2 generate-and-select
> path with a documented compression cap.

Eleven languages, eleven verdicts, no aggregates.
