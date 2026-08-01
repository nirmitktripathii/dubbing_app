# Research Analysis — Notebook 02: Length-Constrained Translation LLM Fine-Tune
### Indic Dubbing Pipeline V3 · Analysis version 1.0 · 2026-07-17 (updated live during Day 2 / Kaggle V12)

This is the canonical in-depth analysis of the currently-running fine-tuning stage,
connecting (i) the dataset and its isochrony-specific construction, (ii) the pipeline
architecture the model will serve, and (iii) the observed training/evaluation telemetry —
ending with inferences for downstream stages and the publication record.

---

## 1. The task, stated precisely

Given a prompt `[Translate to {Language}] [Target Phonemes: N] "{english_sentence}"`,
produce a translation in the target language that (a) preserves meaning and (b) has a
phoneme count close to N. N encodes *time*: dubbing requires the translated speech to fit
the source utterance's physical time window (isochrony). N is derived upstream from the
segment duration via per-language speaking-rate heuristics (`common/languages.py`), and
downstream a trained DurationPredictor validates what the model actually produced —
the LLM is the *proposer*, not the *enforcer* (see §3).

## 2. Dataset: provenance, construction, and known biases

**Source corpus.** AI4Bharat **Samanantar** (~49.7M En↔Indic pairs, 11 languages).
Streaming ingestion (no full download). License nuance (recorded for the paper's ethics
section): AI4Bharat's site and the IndicTrans2 paper describe Samanantar as CC0, but the
HF dataset card carries a `cc-by-nc-4.0` tag — the two disagree. **BPCC** (Samanantar's
superset successor) publishes an explicit license table with all mined corpora CC0, and
`dataset_generator.py` supports `--dataset bpcc` (subset `samanantar_v2`) as the clean
commercial-license path. The present training run was built from the Samanantar path —
fine for the research phase; re-generation from BPCC is the flagged path to a
commercially unencumbered release model.

**Sampling.** 5,000 pairs per language × 11 languages = 55,000 rows → shuffled, 3% held
out: **53,350 train / 1,650 val**. Filters: empty pairs and >400-char sentences dropped
(padding-cost control), rows whose reference phonemizes to 0 phonemes dropped.
(The generator's default of 20k/lang was deliberately not used this round — the
notebook's own budget note prescribes 5k/lang for the first full pass on the T4 quota,
20k/lang as a possible second pass.)

**Isochrony-specific modification — the self-labeling trick.** For each pair, the
*reference* Indic translation is phonemized with espeak-ng (same `phonemize_text`
function used at inference by the DurationPredictor — deliberate representational
consistency), its phoneme count N is measured, and N is written INTO the prompt. Labels
are therefore perfectly self-consistent by construction: "asked for N" is always paired
with "answer that has exactly N." No annotation was required, no phoneme-count noise
exists in the training signal, and the conditioning is learned *implicitly* through
ordinary next-token cross-entropy — there is **no separate phoneme-count loss term**
(a differentiable one would require backprop through G2P, which is non-differentiable;
the project deliberately avoids RL-style machinery at this stage).

**Known bias (single-reference limitation).** Samanantar provides ONE reference per
English sentence at whatever length the human translator produced. The model therefore
learns "translate naturally, and the requested N will roughly match what naturally comes
out" — a correlation between prompt-N and natural length — NOT true elasticity
(compressing/expanding the SAME meaning to arbitrary different N). Consequences:
- Expect good adherence when the requested N is near the natural translation length
  (the common case, since production N comes from the real segment duration).
- Expect degraded adherence for aggressive compression targets (the 65%-budget
  "minimal" candidates in the inference loop) — the model has rarely seen exemplars of
  deliberate compression.
- **Remedy is already scheduled**: Notebook 03 (`length_augmentation.py`) bootstraps
  0.72×/1.28× self-paraphrases of each reference, double-gated by semantic similarity
  (MiniLM cosine ≥ 0.80) and actual phoneme movement (≥10% in the intended direction),
  expanding each sentence to up to 3 lengths. The 02-model being trained now is the
  generator for that augmentation — meaning 02's quality lower-bounds 03's data quality.

**Completion-only loss detail (matters for interpreting loss values).** Rows are
converted to TRL's conversational format, which auto-enables completion-only loss:
cross-entropy is computed ONLY over the Indic translation tokens, not the English prompt.
Observed losses are therefore pure translation-generation uncertainty, not diluted by
easy prompt-echo tokens — a lower-variance, more honest signal than full-sequence loss.

## 3. Architectural context: where notebook 02 sits

Pipeline (one Kaggle notebook per stage):
- **01_wiring_and_dataset** — dependency wiring; generated the 53,350/1,650 dataset (done).
- **02_llm_finetune** — THIS STAGE. QLoRA on Llama-3.1-8B-Instruct (unsloth 4-bit).
- **03_length_augmentation** — self-paraphrase elasticity data (uses 02's adapter).
- **04_tts_data_prep** — TTS manifests (Rasa/Kathbath) + WSOLA prosody augmentation.
- **05_indicf5_finetune** — IndicF5 (330M flow-matching TTS) fine-tune w/ coverage loss.
- **06_adapter_and_validation** — FiLM speaker adapter; forced-alignment validation.
- **07_multispeaker_dubbing** — end-to-end orchestration (diarization → translation →
  duration validation → TTS → assembly).

**Two-layer isochrony enforcement** (the design insight that de-risks this training run):
at inference (`isochrony_translation_v3.py`), the fine-tuned model generates 3 candidates
at 100/85/65% of the phoneme budget; each is scored by the trained DurationPredictor in
*milliseconds*; any candidate predicted to overflow the segment window (>8% tolerance) is
rejected outright; up to 3 shorter-budget refinement rounds follow if nothing fits.
Implication for interpreting 02's metrics: the LLM does not need to be perfect at length
control — it needs to put at least one of three shots inside a window that a separate
model referees. This materially lowers the eval-loss bar required for downstream success
and shifts the correct evaluation target from "exact phoneme match" to "candidate-set
hit rate" (defined in §6).

**Training config** (run_config.json, both days): LoRA r=16, α=16, dropout 0, on
q/k/v/o/gate/up/down_proj (~0.5% of params trainable); 4-bit NF4 base; seq 512;
effective batch 16 (2×8); lr 2e-4, 3% warmup, decay; 2 epochs = 6,670 steps; save/eval
every 200; seed 3407; `max_train_seconds=14400` (4h/day time-slicing with clean
checkpoint + auto-resume — the quota-bounded training protocol under study).

## 4. Observed telemetry and what it says

### 4.1 The evaluation series (primary signal; 1,650 held-out rows, all 11 languages)

| Step | Epoch | eval_loss | Perplexity e^loss | Δ vs prev eval |
|---|---|---|---|---|
| 400 | 0.120 | 0.4997 | 1.648 | — |
| 600 | 0.180 | 0.4891 | 1.631 | −0.0106 |
| 1200 | 0.360 | 0.4775 | 1.612 | −0.0116 (over 600 steps) |
| 1400 | 0.420 | 0.4680 | 1.597 | −0.0095 |
| 1600 | 0.480 | 0.4626 | 1.588 | −0.0054 |
| 1800 | 0.540 | 0.4588 | 1.582 | −0.0038 |
| 2000 | 0.600 | 0.4599 | 1.584 | +0.0011 (first uptick — noise-level) |
| 2200 | 0.660 | 0.4525 | 1.572 | −0.0074 |
| 2400 | 0.720 | 0.4497 | 1.568 | −0.0028 |
| 2600 | 0.780 | 0.4470 | 1.564 | −0.0027 |
| 2800 | 0.840 | 0.4403 | 1.553 | −0.0067 |
| 3000 | 0.900 | 0.4418 | 1.556 | +0.0015 (noise) |
| **3200** | **0.960** | **0.4368** | **1.548** | **−0.0050 — GLOBAL MIN** |
| 3400 | 1.020 | 0.4405 | 1.554 | +0.0037 (post-epoch-1 plateau begins) |
| 3600 | 1.080 | 0.4400 | 1.553 | −0.0005 |
| 3800 | 1.140 | 0.4393 | 1.552 | −0.0007 |

**Day-3 addendum (2026-07-18): plateau + overfitting onset.** Eval min occurred at step
3200 (before the epoch-1 boundary at ~3335); the four epoch-2 evals are flat (0.437–0.441)
while epoch-2 train loss dropped to 0.33–0.36 — train-eval gap widened ~0.015 → ~0.09
(memorization signature on data repetition). F2 revised accordingly; deployment must use
checkpoint-argmin(eval), not checkpoint-final. Full analysis:
reports/combined_report_days1-3_2026-07-18.md.

Session boundary (step 1357, Day1→Day2) sits between the 1200 and 1400 evals: the series
crosses it without any discontinuity — first quantitative evidence for the time-slicing
equivalence claim (F1, §7).

**Level interpretation.** Loss 0.46 ⇒ the model assigns ≈ e^−0.4626 ≈ 63% mean
probability to the correct next token of the reference translation. For an 8B
instruction model that could already translate En→Indic before fine-tuning, the work
being done here is (a) format compliance, (b) distribution shift toward Samanantar's
register, and (c) absorbing the [Target Phonemes] conditioning — which is why loss
started near ~0.52 (not ~2-3 as it would from scratch) and why the marginal gains are
concentrated and slow. This must be stated in the paper to preempt "your model barely
improved" reviews: the delta that matters is not 0.52→0.46 CE but the phoneme-adherence
delta vs. the base model, which CE only proxies (see §6 gap list).

**Shape interpretation.** Per-200-step improvement is decaying roughly geometrically
(−0.0106, −0.0039avg, −0.0095, −0.0054 — noisy but decelerating). Crude power-law
extrapolation of the eval series projects **eval_loss ≈ 0.43–0.45 (ppl ≈ 1.54–1.57) at
step 6,670** if trends hold — i.e., most of the readily-available gain will be captured
by mid-epoch-2, supporting an early-stop option at ~step 5,000 if quota pressure rises
(decision point flagged for Day 4).

### 4.2 Train-loss behavior

Minibatch (16-row) losses oscillate 0.43–0.53 throughout — variance from batch
composition (languages × sentence lengths × rarity), not instability. Day-2 window
(steps 1360–1770): 0.4378–0.4662, centered ~0.45 (ppl ≈ 1.57). Train–eval gap ≈ 0.01–0.02
in loss — essentially zero generalization gap at r=16 capacity on 53k rows; no
overfitting signal anywhere in the series. Grad-norm stayed in [0.36, 0.79] across ~1,770
steps and a session boundary; no spikes/NaN — the fp16 + 4-bit + LoRA numerical setup is
stable at this lr.

### 4.3 Resume mechanics (the methodology result)

- LR at Day-1 end: 1.651e-4 (step ~1330); first Day-2 readings continue at 1.60e-4 —
  scheduler state restored, no re-warmup. Optimizer moments likewise (no loss spike:
  first Day-2 losses 0.44–0.47 continue Day-1's ending level).
- Overhead economics: Day 1 paid 66 min setup (model download + full smoke test) for 240
  min training (27.5% overhead). Day 2 paid ~8 min (smoke test self-skipped via its own
  restored checkpoint) — ~3% overhead. Fixed cost amortizes: the time-sliced regime's
  per-session tax is front-loaded, not recurring.
- Two failure classes found and neutralized (F4/F5, §7): the `%%writefile` cell-magic
  authoring failure (V8, fail-fast, 24s) and the silent single-level-glob resume failure
  (V11, caught at 10 min) — the latter is the dangerous class: it produces *plausible
  training* that silently restarts from step 0 every session. Countermeasure now in the
  notebook: explicit post-restore print of the checkpoint list, and the daily protocol
  requires verifying it before letting a session proceed.

### 4.4 Throughput and cost accounting

~9.4 s/step (Day 1) vs ~10.3–10.8 s/step (Day 2) on the same nominal hardware (T4) —
~10% inter-session throughput variance (shared-tenancy noise); relevant to the paper's
budget model. Eval cost: 285 s × ~7 evals/session ≈ 33 min ≈ 14% of the 4h budget —
a deliberate trade for eval-series resolution across the session boundary (the
phenomenon under study). Projected completion: step ~2,600 today (Day 2), ~6,670 after
**~4 more daily sessions** (≈ Day 5–6), total ≈ 17–18 GPU-hours training + ~1.5h
overhead against the 30h/week quota.

## 5. Inferences for downstream stages (03–07)

**HAND-OFF RULES (binding; updated 2026-07-19 after the interim gating eval).**
- **Checkpoint selection — REVISED, soup BUILT & VERIFIED 2026-07-20.** The interim
  `phoneme_adherence_eval` pass (57%, `reports/final_eval_report_2026-07-19.md`) showed
  checkpoints 2558/3200/3400/3801 statistically tied on CE, adherence, and chrF++ (no
  single winner). The checkpoint-soup (uniform weight-average of all 4 LoRA adapters,
  448 tensors) was built on Kaggle notebook **`notebookf7c7e78db7`** ("02d — Build
  checkpoint-soup adapter"), CPU-only, 90.9s runtime, no GPU quota consumed. Sanity-check
  tensor norms are monotone and plausible (ckpt-2558=4.8024 → ckpt-3801=5.0416, soup=
  4.8543 — sits within the source range as expected of a uniform average, not an
  extrapolation). Output: `checkpoint_soup/adapter_model.safetensors` +
  `adapter_config.json` + tokenizer files + `SOUP_PROVENANCE.json` (records source
  steps + rationale), persisted as that notebook's Kaggle Output. **Notebooks 03, 06, 07
  should attach `notebookf7c7e78db7` as an input and load `checkpoint_soup/`** as the
  default translator adapter, pending confirmation from the full 11-language re-eval
  (Task #37) that the soup performs at least as well as the best single checkpoint.
  Fallback if that input isn't attachable: single argmin-eval checkpoint-3200
  (eval 0.4368). NEVER the `checkpoints/translation_llm/` root (= last step, overfitting
  regime, §4.2).
- **Gating eval — interim pass delivered 2026-07-19, confirmatory pass delivered
  2026-07-20** (retroactive, at 57%/step 3801, run early per explicit request rather
  than waiting for the ≥6,600-step completion gate). Full results, per-language
  breakdown, continue-vs-stop recommendation, and the n=45 confirmatory addendum:
  `reports/final_eval_report_2026-07-19.md` (§6). A further full 11-language gating
  eval still runs at true 02 completion per the daily protocol (Task #37). Method:
  `files_v3/evaluation/README.md`.

1. **For 03 (length augmentation)**: the adapter is the paraphrase generator; its
   near-zero train-eval gap suggests it will paraphrase in-distribution Samanantar
   register well, but the single-reference bias means the 0.72×/1.28× rewrite targets are
   *out-of-distribution* requests — expect elevated Gate-2 (length-movement) rejection
   rates. **Log gate-level rejection statistics per language in 03** — that acceptance
   rate is itself a publishable measurement of the base fine-tune's latent elasticity.
2. **For 07 (inference loop)**: eval CE cannot predict candidate-set hit rate directly.
   Before 03 begins, run the §6 post-training eval on checkpoint-final vs. the BASE
   model (no adapter) to quantify what fine-tuning bought — this base-vs-tuned contrast
   is the paper's headline table.
3. **Compression asymmetry prediction** (testable, pre-registered here): phoneme
   adherence error will be larger for prompts requesting N below ~85% of natural length
   than above it, due to the single-reference bias. If confirmed by the §6 eval, it
   directly motivates 03; if refuted, 03's cost can be down-scoped.
4. **For 05/06 (TTS)**: the DurationPredictor's ms-validation quality bounds the whole
   isochrony chain; its error distribution (trained in 01/validated later) should be
   reported jointly with 02's adherence numbers, since window-rejection correctness
   depends on both.
5. **Per-language reporting requirement**: aggregate eval_loss hides an 11-language mix
   (Dravidian agglutinative vs. Indo-Aryan morphology ⇒ different tokens-per-phoneme and
   difficulty). The W&B run logs only the aggregate; per-language CE + adherence must be
   computed in the §6 eval from val.jsonl's `language` field.

## 6. Metrics gap register — what is NOT yet measured (action items)

| Missing metric | Why it matters | Where it gets added |
|---|---|---|
| Phoneme-adherence: mean/median |N_actual−N_requested|/N_requested | The actual task objective; CE only proxies it | Post-training eval script (planned `evaluation/phoneme_adherence_eval.py`), run in 02 or 03 notebook on ~500 val rows: generate → espeak-ng G2P → compare |
| Candidate-set hit rate: P(≥1 of 3 candidates fits window) | The quantity 07 actually consumes | Same script, simulating the 100/85/65% loop with DurationPredictor |
| Translation quality: chrF++ (BLEU secondary) vs references | Guards against length-adherence degrading fidelity | Same script (sacrebleu) |
| Per-language breakdowns of all the above | 11-language fairness + difficulty analysis | Same script, group by `language` |
| Base-model (no adapter) baseline on all the above | The paper's headline delta | Same script, run twice |

## 7. Scientific findings register (running; evidence-linked)

- **F1 — Time-sliced QLoRA ≈ continuous training**: no loss/LR/grad transient across the
  1357 boundary; eval series monotone through it. (Strengthens with each boundary; 4 more coming.)
- **F2 — Zero generalization gap at r=16/53k rows**: train≈eval throughout; LoRA capacity
  is the effective regularizer. Relevant to adapter-size guidance for low-resource practitioners.
- **F3 — Overhead amortization**: 27.5% (cold) → ~3% (warm) session overhead; time-sliced
  training's fixed costs are first-session-only. Quota budget models should not multiply cold-start cost by session count.
- **F4 — Fail-fast vs. silent failure taxonomy in notebook-mediated training**: authoring
  errors (V8) self-announce; environment-topology errors (V11 glob) are silent and
  compounding. Countermeasure: mandatory provenance printing (restored-checkpoint list +
  resume step) as a protocol requirement.
- **F5 — Free-tier throughput variance**: ~10% step-time drift across sessions on nominally
  identical hardware; error bars required on any GPU-hour claims.
- **F6 — Deceleration of eval improvement**: consistent with power-law; projects
  0.43–0.45 terminal CE; supports possible early stop at ~5k steps under quota pressure.
- **F7 — Implicit phoneme conditioning learnability, DELIVERED 2026-07-19 (interim,
  57%)**: fine-tuning cuts phoneme-adherence relative error by ~79% vs. base (0.495→0.103)
  and improves chrF++ fidelity by ~37% (22.0→30.3), with no single checkpoint dominating
  among 2558/3200/3400/3801 (motivates checkpoint-soup deployment, see §5). Compression-
  asymmetry sub-claim not yet directly tested (needs a length-sweep below 85% specifically
  isolated) — still open. Full detail: `reports/final_eval_report_2026-07-19.md`.
- **F8 (2026-07-19, CONFIRMED/REVISED 2026-07-20 at n=45)** — Per-language plateau is
  non-uniform but small in CE terms. Of 11 languages, only Assamese and Odia show
  unambiguous continued CE improvement past step 3200 (both also gain chrF over the same
  span); Gujarati and Telugu show the earliest mild overfitting-style CE uptick. The
  "still improving" set (as, or, ta) and the "early overfit" set (gu, te) are disjoint
  from each other — exactly the condition under which per-language checkpoint routing
  (Option B) would pay off if pursued later.
  **Confirmatory re-run (n=45, `reports/final_eval_report_2026-07-19.md` §6) resolved
  the three n=15 caution flags:** (a) the Bengali chrF regression was **refuted** — chrF
  actually rose +1.22 points ckpt-2558→3801 at 3× sample size; drop from the register.
  (b) the Gujarati chrF regression was **confirmed but at roughly half the original
  magnitude** (−1.79 points, not −3.5) and reads as the same early-overfitting phenomenon
  already flagged on CE, not a separate failure mode. (c) Malayalam's length-conditioning
  weakness was **reproduced and localized**: length_slope stays low (~0.45–0.46) and
  checkpoint-independent, with the raw-generation dump showing the undershoot concentrated
  specifically on long requested lengths (e.g. 122-phoneme requests generating only ~72),
  short/medium requests track fine — likely a Malayalam-specific phoneme/token-count
  mismatch (agglutinative script), not a checkpoint artifact. Malayalam now joins the
  targeted-continuation candidate list (oversample long-N Malayalam examples), alongside
  Assamese/Odia (still improving) and Gujarati (early overfit, chrF-confirmed).

## 8. Ongoing observability protocol (applies to notebooks 03–07 as well)

Canonical local archive: **`E:\Dubbing app\files_v3\research_logs\`**
- `ANALYSIS_<notebook>.md` — one living analysis doc per pipeline stage (this file for 02).
- `sessions/dayN_<date>_v<kaggleversion>.md` — per-session factual log: config hash,
  steps covered, stop reason, checkpoint list, incidents, key metrics.
- `metrics/dayN_v<version>_metrics.csv` — schema:
  `session,kaggle_version,wandb_run,step_approx,epoch,metric,value,perplexity,grad_norm,learning_rate,notes`
- `checkpoints/` — final/best adapters pulled from Kaggle at stage completion (not daily;
  Kaggle version outputs are the daily checkpoint archive of record).
- W&B project `indic-dubbing-v3` = full-resolution time series of record; run IDs are
  cross-referenced in every session log (Day 1: vg1fhqxx feasible-rain-4; Day 2: dvnoyzv5).
- The daily scheduled task (6:08 AM) archives the previous session's metrics into this
  folder and verifies resume provenance before allowing a new session to run.
- Per-stage obligations, 03–07: same session/metrics discipline, PLUS stage-specific
  registers — 03: gate acceptance rates per language; 04: augmentation QC (duration
  accuracy, pitch preservation); 05: flow loss + coverage aux loss + periodic CTC
  alignment scores; 06: adapter identity-init verification + alignment eval history;
  07: end-to-end isochrony compliance (candidate hit rates, refinement-round counts,
  overflow rejections) + human/LLM fidelity spot checks.

## 9. Publication mapping (working skeleton)

Methods ← §2 (dataset construction, self-labeling), §3 (two-layer enforcement), §4.3/8
(quota-bounded training protocol). Results ← §4 tables + §6 eval outputs + F-register.
Discussion ← §5 inferences, single-reference bias and 03's remedy, license path (BPCC).
Reproducibility ← configs, seeds (3407), W&B runs, session logs, per-version Kaggle
outputs. Target framing: "Isochrony-constrained translation for Indic dubbing on
free-tier compute: implicit phoneme conditioning + validate-and-reject inference,
with a quota-bounded training methodology."
