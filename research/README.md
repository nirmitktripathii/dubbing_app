# Research Artifacts — Length-Constrained Translation (V3)

This directory holds the evaluation harness, results, and training telemetry behind the
length-constrained ("isochrony-aware") translation work in the Indic dubbing pipeline.

It exists so that the numbers quoted in the write-ups are **checkable**. Every figure in
the articles traces to a file here.

---

## ⚠ Read this before quoting any number below

Two audits in August 2026 invalidated the original measurements. Both are documented, and
the superseded numbers are kept rather than deleted so the correction is auditable.

| Document | What it establishes |
|---|---|
| [`RULER_AUDIT_FINDINGS.md`](RULER_AUDIT_FINDINGS.md) | The corpus was labelled with **character** counts while the prompts said "phonemes" — a silent `except Exception` fallback in the phonemizer, firing for the whole corpus. `frac_label_eq_chars = 1.00` in all 11 languages. |
| [`CORRECTED_EVAL_FINDINGS.md`](CORRECTED_EVAL_FINDINGS.md) | Re-evaluation under the corrected ruler. The aggregate hides a **two-population split**: 6 languages follow the phoneme budget (probe slope 0.69–0.94), 5 effectively ignore it (0.35–0.42). |

Concretely, in the tables that follow:

- **Any length/adherence figure sourced from `eval_out_*` is measured in characters, not
  phonemes.** Superseded by `evaluation/results/corrected_eval/`.
- **Any "length slope" quoted as a single number is the *population* estimator**, which
  regresses across different sentences and is confounded by sentence length. A model that
  ignores the budget entirely still scores well on it. The estimator that answers the
  question is `length_slope_probe` — same sentence, budget swept 0.6×–1.4×.
- The decoupling result (loss flat, length control still improving) **reproduces** under the
  corrected ruler and stands.

---

## Project status

The dubbing pipeline is **in development — stage 3 of 7**. What's here is a complete record
of the length-constrained translation fine-tune (stage 2, notebook `02_llm_finetune`) and
its evaluation. It is not a finished system, and some of the results below are explicitly
open questions rather than conclusions.

Two different systems reach isochrony in this repo, and it's worth not confusing them:

| | Where | How |
|---|---|---|
| **v2 — deployed** | `pipeline/isochrony_translation.py` | Prompts a frontier model with a phoneme budget derived from a per-language expansion-ratio table, generates 3 candidates, scores them with `pipeline/phoneme_counter.py` |
| **v3 — research track** | *(pipeline not yet published)* | Fine-tunes Llama-3.1-8B so the phoneme budget is learned rather than prompted |

**Everything in this `research/` directory is about v3.** The v3 pipeline code itself is
still moving and isn't published yet; these are its measurements.

---

## Where each number comes from

| Claim | File |
|---|---|
| **Corrected, phoneme-ruled** — relErr 0.565 → 0.117; signed +0.344 → −0.013; chrF++ 21.3 → 30.3; **probe** slope 0.481 → 0.637, split two ways across languages | `evaluation/results/corrected_eval/corrected__eval_report.md` |
| **Corrected** per-language probe slope and R² | `evaluation/results/corrected_eval/corrected__length_response.csv` |
| **Corrected** — the budget-rescale falsification test (salvage path closed) | `evaluation/results/corrected_eval/rescaled__eval_report.md` |
| ~~Length error 0.495 → 0.103; chrF++ 22.0 → 30.1; signed error +0.290 → −0.043; length slope 0.593 → 0.687~~ **superseded — character ruler, population slope** | `evaluation/results/eval_out_all__eval_report.md` |
| Full per-checkpoint trajectory for those four metrics | `evaluation/results/eval_out_all__trajectory_summary.csv` |
| Cross-entropy across all 23 checkpoints; minimum at step 3200 | `evaluation/results/eval_out_ce__eval_report.md` |
| Per-language decomposition (the aggregate hides that languages plateau at different steps) | `evaluation/results/eval_out_ce__per_checkpoint_metrics.csv` |
| Length-response probe — output length vs requested length at 0.6/0.8/1.0/1.2/1.4× | `evaluation/results/eval_out_all__length_response.csv` |
| Training config: QLoRA r=16, 53,350/1,650 rows, 11 languages, effective batch 16, 2 epochs = 6,670 steps | `logs/EXPERIMENT.md`, `logs/ANALYSIS_02_llm_finetune.md` |
| "No resume transient" — LR resumed mid-schedule at 1.60e-4, losses continued seamlessly | `logs/sessions/day2_2026-07-17_v12.md` |
| Session overhead dropping 66 min → ~8 min once checkpoints were attached | `logs/sessions/day2_2026-07-17_v12.md` |
| The silent resume bug (a one-level glob matching nothing) | `logs/sessions/day2_2026-07-17_v12.md`, incidents 1–3 |
| Cross-session loss/perplexity time series | `logs/metrics/day{1,2,3}_*.csv` |
| Combined multi-day analysis | `logs/reports/combined_report_days1-3_2026-07-18.md`, `logs/reports/final_eval_report_2026-07-19.md` |

---

## The evaluation harness

`evaluation/phoneme_adherence_eval.py` produces everything in `evaluation/results/`.
`evaluation/HARNESS.md` explains the design in full — in particular *why* cross-entropy
alone can't answer whether length conditioning is still being learned.

Three modes:

```bash
# Fast, teacher-forced, per-language cross-entropy only
python -m evaluation.phoneme_adherence_eval \
    --val_jsonl data/translation_dataset/val.jsonl \
    --checkpoints_glob "checkpoints/translation_llm/checkpoint-*" \
    --base_baseline --output_dir eval_out --mode ce

# Length-control probe only
--mode length

# Everything, including generation-based metrics (slow — budget accordingly)
--mode all
```

The four metrics, and why each exists:

- **`adherence_rel_mean`** — `|N_generated − N_requested| / N_requested`. How far off the
  requested phoneme count the output lands.
- **`adherence_signed_mean`** — the same miss with its sign, so you can see whether the
  model runs systematically long or short. The base model sits at +0.290; that bias is why
  naive dubbing pipelines always need a compressor.
- **`length_slope_probe`** — regress produced length on requested length across five budgets
  **for the same sentence**. Slope 1.0 = full obedience, slope 0 = the budget was ignored.
  This is the only metric here that is mathematically independent of cross-entropy, which is
  what makes it useful. It is *the* acceptance metric, read per language.
- **`length_slope_population`** — the same regression run across *different* sentences. It is
  reported for comparison and must never be selected on: it is confounded by sentence length,
  so a model that ignores the budget entirely still scores well. The harness names the two
  separately for exactly this reason; an earlier version reported only this one under the
  bare name `length_slope`.
- **`semantic_mean` / `semantic_degraded_frac`** — cosine similarity against the reference
  under `paraphrase-multilingual-MiniLM-L12-v2`, and the fraction falling below the 0.80
  gate. Budget obedience and meaning trade against each other (r = −0.58 across the eleven
  languages), so a slope number without a semantic number beside it is not interpretable.
- **`chrf_mean`** — translation quality, as a guard against hitting the budget by deleting
  content.

**Important:** phoneme counts must be produced by the same grapheme-to-phoneme function that
wrote the training labels. Different G2P tools make different judgement calls and introduce a
systematic offset large enough to swamp the signal.

---

## The finding worth reading

Around step 3200 cross-entropy went flat (0.5056, then 0.5079, 0.5079, 0.5075 — global
minimum at 3200, then noise). The textbook move is to early-stop.

But between steps 3200 and 3801, while loss was flat:

- length error kept falling, 0.104 → 0.103
- length slope kept climbing, 0.656 → **0.687**

> Those two figures are the superseded character-ruled, population-slope versions. Under the
> corrected phoneme ruler the same decoupling holds — CE moved +0.0001 between steps 3400
> and 3801 while the **probe** slope rose 0.623 → 0.637 — so the finding survives its own
> re-measurement. See [`CORRECTED_EVAL_FINDINGS.md`](CORRECTED_EVAL_FINDINGS.md) §5.

The slope is computed from generated text, not from teacher-forced likelihood of a
reference, so it is not a restatement of the loss. A rising slope against a flat loss is
direct evidence that length conditioning was still being learned after the loss curve went
quiet — and early-stopping on loss would have discarded 600 steps of improvement on the
actual objective.

Decision rule, encoded in the harness:

> CE-flat **+** adherence/slope still moving ⇒ keep training.
> CE-flat **+** adherence-flat ⇒ genuine plateau; early-stop loses nothing measurable.

A corollary that's still open: the best-loss checkpoint (3200) and the best-adherence
checkpoint (3801) are **not the same checkpoint**. Current plan is to average the adapter
weights across the plateau (steps 2800–3400) rather than pick either.

---

## What is deliberately not here

- **The v3 pipeline code** — still in development, published when it stabilises.
- **Model checkpoints** — the adapters are hundreds of MB; Kaggle notebook outputs remain
  the archive of record.
- **The training dataset** — derived from [Samanantar](https://ai4bharat.iitm.ac.in/samanantar);
  use the source rather than a copy.

External systems of record, cross-referenced in the logs: the W&B project
`indic-dubbing-v3` and the Kaggle notebook version history for `02-llm-finetune`.

---

## Reproducing

The harness runs against any checkpoint directory produced by the fine-tune, plus a
`val.jsonl` with `english`, `target`, `language`, and `n_phonemes` fields. Start with
`--mode ce` (cheap, teacher-forced) to get the per-language trajectory, then run `--mode all`
on just the interesting checkpoints — best-eval, final, and two or three around the plateau —
rather than every one. Generation-based metrics are the expensive part.
