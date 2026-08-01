# Combined Research Report — Days 1–3 (Kaggle V10, V12, V13)
### Notebook 02: Length-Constrained Translation LLM Fine-Tune · Indic Dubbing Pipeline V3
### One-time catch-up report covering all sessions to date · Generated 2026-07-18
### (Daily reports begin with the next scheduled run; this document consolidates the backlog.)

---

## 1. Experiment recap (context for standalone reading)

QLoRA fine-tune (r=16, α=16, 7 projection targets, 4-bit NF4 base) of
meta-llama/Llama-3.1-8B-Instruct on 53,350 length-conditioned En→11-Indic translation
pairs (Samanantar-derived; phoneme targets self-labeled via espeak-ng G2P of the
reference; completion-only cross-entropy). Plan: 2 epochs = 6,670 steps, effective batch
16, lr 2e-4 → decay, seed 3407. Regime under study: quota-bounded time-slicing —
4h wall-clock per daily session (TimeLimitCallback), clean checkpoint at cap,
auto-resume next session. Hardware: Kaggle free T4. Observability: W&B project
indic-dubbing-v3 + Kaggle logs + this archive.

## 2. Session ledger

| Day | Kaggle ver | W&B run (full) | Steps covered | Epoch reach | Session outcome |
|---|---|---|---|---|---|
| 1 (Jul 16) | V10 | n/a→vg1fhqxx* | 0 → 1,357 | 0.407 | Success; 4h cap hit exactly; ckpt-1357 |
| 2 (Jul 17) | V12 | n3q8p5le | 1,357 → 2,558 | 0.767 | Success; cap hit; ckpt-2558; 4.76 GB output |
| 3 (Jul 18) | V13 | 6zpo7lih | 2,558 → 3,801 | 1.140 | Success; cap hit; ckpt-3801; 394 files |

*Day 1's W&B naming: feasible-rain-4 (vg1fhqxx). Day 2 correction on record: dvnoyzv5
was the smoke-phase init; n3q8p5le is the full run. Day 3: gohveo9u = smoke, 6zpo7lih = full.
Aborted/incident versions (not training sessions): V6 uncapped (cancelled), V8 %%writefile
fail (24s), V11 silent-resume glob bug (cancelled at 10 min).

**Cumulative: 3,801 / 6,670 steps = 57.0% of plan; ~12.7 GPU-hours training consumed.**

## 3. The complete evaluation series (1,650 held-out rows; ppl = e^loss)

| Step | Epoch | eval_loss | Perplexity | Δ vs prev | Session |
|---|---|---|---|---|---|
| 400 | 0.12 | 0.4997 | 1.648 | — | 1 |
| 600 | 0.18 | 0.4891 | 1.631 | −0.0106 | 1 |
| 1200 | 0.36 | 0.4775 | 1.612 | −0.0116/600 | 1 |
| 1400 | 0.42 | 0.4680 | 1.597 | −0.0095 | 2 |
| 1600 | 0.48 | 0.4626 | 1.588 | −0.0054 | 2 |
| 1800 | 0.54 | 0.4588 | 1.582 | −0.0038 | 2 |
| 2000 | 0.60 | 0.4599 | 1.584 | +0.0011 | 2 |
| 2200 | 0.66 | 0.4525 | 1.572 | −0.0074 | 2 |
| 2400 | 0.72 | 0.4497 | 1.568 | −0.0028 | 2 |
| 2600 | 0.78 | 0.4470 | 1.564 | −0.0027 | 3 |
| 2800 | 0.84 | 0.4403 | 1.553 | −0.0067 | 3 |
| 3000 | 0.90 | 0.4418 | 1.556 | +0.0015 | 3 |
| **3200** | **0.96** | **0.4368** | **1.548** | **−0.0050 · GLOBAL MIN** | 3 |
| 3400 | 1.02 | 0.4405 | 1.554 | +0.0037 (post-epoch-1) | 3 |
| 3600 | 1.08 | 0.4400 | 1.553 | −0.0005 | 3 |
| 3800 | 1.14 | 0.4393 | 1.552 | −0.0007 | 3 |

Total improvement: 0.4997 → 0.4368 best (−0.063 CE; perplexity 1.648 → 1.548, −6.1%).

## 4. Principal analyses

### 4.1 Convergence has entered a plateau, and the epoch boundary is the hinge
Through epoch 1 (steps ≤3,335) the eval series is near-monotone with decelerating gains
(single upticks at 2000 and 3000 are ≤0.0015, noise-level). The global minimum 0.4368
occurred at step 3,200 — *before* the epoch-1 boundary. The four evals since (3400–3800:
0.4405, 0.4400, 0.4393) straddle 0.437–0.441: no meaningful improvement across 600
steps of epoch 2. The Day-2 power-law projection (terminal 0.43–0.45) is confirmed
early: the model has effectively reached its eval floor at 57% of the training plan.

### 4.2 Overfitting onset detected at the second epoch — Finding F2 REVISED
Epoch-1 train-loss center: ~0.45–0.46 with train-eval gap ≈ 0.015. First Day-3 epoch-2
readings (steps 3770–3790): train 0.3323–0.3590 (ppl 1.39–1.43) against eval ~0.44 —
**gap ≈ 0.08–0.10, a 5–6× widening**. Mechanism: second pass over identical data; the
model recognizes repeated examples (memorization credit) while held-out uncertainty is
unchanged. F2 now reads: *LoRA r=16 fully regularizes single-epoch training on 53k rows,
but does NOT prevent memorization signatures on data repetition; eval-based checkpoint
selection is mandatory in epoch 2.* This is a useful, publishable boundary condition on
the common "LoRA doesn't overfit" folklore.

### 4.3 Recommendation (decision point, Day 4)
Two defensible options:
- **(a) Early stop now / after Day 4 partial:** adopt checkpoint-3200 (best eval) or
  whichever epoch-2 checkpoint first beats 0.4368; saves ~9 GPU-hours (~30% of weekly
  quota) for notebooks 03–05. Risk: length-conditioning behavior may still be refining
  even at flat CE (CE is a proxy — §6 of ANALYSIS_02).
- **(b) Complete the plan (2 epochs)** for protocol cleanliness and the F2 curve, then
  select best-eval checkpoint for downstream use regardless.
Given the paper angle values the full overfitting trajectory, (b) with best-eval
selection is the default unless quota pressure demands (a). Either way: **the deployed
adapter should be checkpoint-argmin(eval), not checkpoint-final.** The final adapter dir
saved at output root currently corresponds to the *last* step — downstream notebooks
(03, 06, 07) must be pointed at the selected checkpoint explicitly.

### 4.4 Time-sliced training equivalence — F1 now supported by two boundaries
Resume boundaries at 1357 and 2558 both show: LR continues mid-schedule (1.651→1.605e-4;
1.274e-4→continuation into Day 3, reaching 8.88e-5 by step 3790), no loss transient, no
grad-norm excursion attributable to resume, eval series smooth across both. Optimizer/
scheduler/RNG state restoration via HF checkpointing is behaviorally lossless at this
scale. One benign anomaly on record: a single grad_norm excursion to 1.091 mid-Day-2,
no train-loss consequence.

### 4.5 Operational economics (free-tier methodology data)
- Session overhead: 66 min (Day 1 cold) → ~8 min (Day 2) → ~9 min (Day 3): fixed costs
  are first-session-only; steady-state overhead ≈ 3-4%.
- Eval tax: 6-7 evals/session × ~280 s ≈ 28-33 min ≈ 12-14% of each 4h budget — the
  price of boundary-resolution on the eval series (deliberate, retained).
- Throughput variance: ~9.4 (D1) vs ~10.5 (D2) vs ~10.1 (D3, implied) s/step — ±5-10%
  shared-tenancy noise; error bars required on GPU-hour claims.
- Incident cost to date: V6 1h32m (pre-protocol) + V8 24s + V11 ~10 min ≈ 1.7 GPU-hours
  lost across all failures; the two in-protocol failures cost <15 min combined (fail-fast
  + early detection). Quota remains comfortable (~13 of 30 h used this week incl. losses).

### 4.6 Findings register status
F1 supported (2 boundaries) · F2 revised (epoch-2 memorization onset — see 4.2) ·
F3 confirmed (overhead amortization curve now 3 points) · F4 unchanged (failure
taxonomy; no new incidents Days 3) · F5 confirmed (±5-10%) · F6 confirmed early
(plateau reached at 0.437-0.44) · F7 still pending the phoneme-adherence eval — now
MORE urgent: with CE flat, adherence is the only metric that can justify (or refute)
continuing to spend quota on epoch 2.

## 5. Downstream implications snapshot — status: IMPLEMENTED / SCHEDULED

**Point 1 (notebook 03 uses best-eval checkpoint) — SCHEDULED.** Recorded as a hard
hand-off rule: notebook 03 (and 06/07) must load the argmin-eval checkpoint explicitly,
never `checkpoints/translation_llm/` root (= last step). Current best is **ckpt-3200**
(eval 0.4368); this will be re-confirmed against every later checkpoint by the eval script
and re-published in each daily report before 03 begins. Encoded in the daily scheduled
task's completion step and in ANALYSIS_02 §5.

**Point 2 (post-training adherence eval) — IMPLEMENTED as a runnable script + SCHEDULED
to run at 02 completion.** `files_v3/evaluation/phoneme_adherence_eval.py` computes G2P
phoneme adherence, chrF++ fidelity, and a length-response slope, all per-language and
across the checkpoint trajectory, plus a base-model baseline (the headline "what
fine-tuning bought" delta). The daily scheduled task now runs it once training reaches
≥6,600 steps. (Candidate-set hit rate needs the trained DurationPredictor from nb 01/05 to
supply real ms windows; the script's signed-error output is the proxy until that model is
available, at which point the hit-rate pass is added.)

**Point 3 (per-language breakdown) — IMPLEMENTED.** Every metric in the eval script is
grouped by `val.jsonl`'s `language` field → `per_checkpoint_metrics.csv`. This directly
resolves the aggregate-plateau blind spot: it exposes whether individual languages are
still improving or have begun overfitting. See §5a for the two open diagnostic questions
this was built to answer.

### 5a. Diagnostic methods now available (answers to the two open questions)

**"Is length conditioning still being learned even though CE plateaued?"** CE cannot tell
you — it scores reference-token probability, not length control, and the two decouple. The
eval script gives three CE-independent signals, read across the checkpoint series:
(i) **adherence error** |N_gen−N_req|/N_req still falling after step 3200 ⇒ still learning;
(ii) **length-response slope** — regress produced-N on requested-N over an N∈{0.6…1.4}×
natural sweep of the SAME sentence; slope→1.0 means true obedience to the length
instruction, mathematically independent of CE (unit-tested: obedient≈0.9/R²0.98,
N-ignoring≈0.03); a slope still climbing across late checkpoints is decisive proof
conditioning is still improving; (iii) **signed error** direction (under/over-shoot).
Decision rule: CE-flat + adherence/slope-still-moving ⇒ keep training; CE-flat +
adherence-flat ⇒ genuine plateau, early-stop at best-eval loses nothing.

**"Are some languages still improving while others overfit?"** Sort `per_checkpoint_metrics.csv`
by step within each language: a language whose per-language eval CE keeps falling at late
checkpoints is under-fit; one whose CE starts rising is overfitting first. Because Dravidian
(agglutinative) and Indo-Aryan languages differ in tokens-per-phoneme and difficulty they
won't plateau together — aggregate CE hides this entirely; the per-language CE trajectory
(a faithful decomposition of the W&B eval_loss, same masking) exposes it, as does the
per-language adherence/slope trajectory. Full method: `files_v3/evaluation/README.md`.

## 6. Reproducibility pointers
Configs/seed in run_config.json (each Kaggle version output); session logs
`sessions/day{1,2,3}_*.md`; metrics CSVs `metrics/day{1,2,3}_*.csv`; W&B runs
vg1fhqxx / n3q8p5le / 6zpo7lih; Kaggle scriptVersionIds 335699281 / 335911104 / (V13 id
in its session log); checkpoints archived per-version on Kaggle (14 dirs at V12-end;
V13 adds 2600–3801 series).
