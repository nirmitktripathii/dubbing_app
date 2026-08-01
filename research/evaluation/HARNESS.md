# Evaluation — Length-Conditioning & Per-Language Diagnostics

`phoneme_adherence_eval.py` is the measurement layer that CE (train/eval loss) cannot
provide. It exists to answer two questions the plateaued eval loss left open, plus to
produce the paper's headline "what fine-tuning bought" table.

## The two research questions and exactly how this answers them

### Q1 — CE has plateaued (~step 3200). Is length conditioning still being learned?

CE measures *token probability of the reference*, not *length control*. These can diverge:
the model can keep getting better at hitting a requested phoneme count while its
next-token CE is already saturated (because CE is dominated by lexical/semantic choice,
of which length control is only a small part). Three independent, CE-free signals here
settle it — run `--mode all` across the checkpoint series (`--checkpoints_glob
"checkpoints/translation_llm/checkpoint-*"`), then read `trajectory_summary.csv`:

1. **Adherence error `|N_gen − N_req| / N_req`** (`adherence_rel_mean`). If this keeps
   falling after step 3200 while eval CE is flat → conditioning still improving.
2. **Length-response slope** (`length_slope`, and the dedicated `length_response.csv`).
   This is the cleanest probe: take the SAME English sentence, ask for it at
   N ∈ {0.6, 0.8, 1.0, 1.2, 1.4} × natural length, and regress produced-N on requested-N.
   Slope → 1.0 means the model truly obeys the length instruction; slope ≈ 0 means it
   ignores N and just translates naturally. A slope that keeps climbing toward 1.0 across
   late checkpoints is *direct* proof length conditioning is still being learned, and it
   is mathematically independent of CE. (Verified in unit tests: an obedient response
   gives slope≈0.9/R²≈0.98; an N-ignoring response gives slope≈0.03.)
3. **Signed error** (`adherence_signed_mean`). Tells you the *direction* of any residual
   miss (negative = systematically too short, positive = too long) — feeds the inference
   candidate-budget design (e.g. whether to add a >100% "expanded" candidate).

**Decision rule** (encoded in the generated `eval_report.md`): CE-flat + adherence/slope
still moving ⇒ keep spending quota; CE-flat + adherence-flat ⇒ genuine plateau, early-stop
at best-eval checkpoint loses nothing measurable.

### Q2 — Are some languages still improving while others overfit?

Every metric is computed **per language** (from `val.jsonl`'s `language` field) at every
checkpoint, written to `per_checkpoint_metrics.csv`. Two ways to read it:

1. **Per-language eval CE across steps.** Sort rows for one language by `step`. A language
   whose `ce_mean` is still falling at late checkpoints is under-fit (wants more training);
   one whose `ce_mean` starts *rising* across late checkpoints is the first to overfit —
   and because Dravidian (Tamil/Telugu/Kannada/Malayalam, agglutinative) vs. Indo-Aryan
   (Hindi/Bengali/Marathi/…, more analytic) differ in tokens-per-phoneme and data
   difficulty, they will not plateau simultaneously. Aggregate CE hides this entirely.
2. **Per-language adherence + slope across steps.** Same trajectory read for the real
   objective, so you can see, e.g., "Tamil length control still tightening at step 3800
   while Hindi adherence plateaued at 3000."

This is how you decide whether to keep training globally, or (future work) up-sample the
still-improving languages in a second data pass.

## Why the CE here is comparable to W&B eval_loss

`completion_ce()` reproduces TRL's completion-only masking exactly (prompt tokens set to
-100), teacher-forced, so a per-language `ce_mean` here averaged over all languages equals
the W&B `eval_loss` up to sampling. That means the per-language split is a faithful
decomposition of the number you already trust — not a different metric with its own scale.

## Why phoneme counting must use the pipeline's own phonemizer

`phonemize_text` (from `translation.duration_predictor`) is the SAME function
`dataset_generator.build_example` used to write the `[Target Phonemes: N]` labels. Adherence
is only meaningful if generated text is counted with the identical G2P; any other
phonemizer would introduce a systematic offset that swamps the signal.

## Running it (on Kaggle, after a training session)

To slot into the pipeline package, copy this file to `pipeline_v3/evaluation/` (create the
package dir + empty `__init__.py`) alongside the notebook's restored checkpoints, then:

```bash
# Trajectory across all checkpoints + base baseline (the full Q1/Q2 + headline table run)
python -m pipeline_v3.evaluation.phoneme_adherence_eval \
    --val_jsonl data/translation_dataset/val.jsonl \
    --checkpoints_glob "checkpoints/translation_llm/checkpoint-*" \
    --base_baseline --output_dir eval_out --mode all

# Cheap CE-only per-language decomposition (no generation; minutes): --mode ce
# Fast length-control probe only: --mode length
```

Cost note: `--mode all` generates text, so budget ~ (adherence_samples_per_lang +
5×probe_sentences_per_lang) generations × 11 languages × N_checkpoints. Start with
`--mode ce` (fast, teacher-forced) to get the per-language CE trajectory cheaply, then run
`--mode all` on just the interesting checkpoints (best-eval, final, and 2–3 around the
plateau) rather than every one.

## Outputs → downstream

- The **argmin(aggregate eval CE)** checkpoint AND the **argmin(adherence)** checkpoint are
  both reported. If they disagree, prefer the adherence-best for deployment (it is closer to
  the true objective) but record both — the disagreement is itself a finding (F7).
- Notebook 03 (paraphrase augmentation) and Notebook 07 (inference) must load the selected
  checkpoint explicitly, NOT `checkpoints/translation_llm/` root (which holds the LAST step).

### On the two-stage / "MoE composite" idea (best-CE model translates → best-adherence model resizes)

Considered and **not adopted as first choice** — with the reasons, because the underlying
instinct is right and re-surfaces in better forms below.

Why the literal two-checkpoint cascade is the wrong tool here:
1. The candidates are not different *experts*; they are the SAME LoRA adapter at two points
   ~600 steps apart on ONE trajectory (same base, same target modules). Their "best-CE" and
   "best-adherence" behaviours are near-identical, so splitting buys little.
2. Stage 2 ("here is a translation, rewrite it to N phonemes") is a capability the model was
   never trained for — it only ever saw "English + N → translation". The length-elastic
   *rewrite* skill is exactly what **Notebook 03** (`length_augmentation.py`) is meant to
   create; the cascade presupposes the thing 03 hasn't built yet.
3. Two forward passes = 2× inference cost on a quota-constrained pipeline.

Better routes, in preference order (all decided by the per-language eval trajectory this
script produces):

- **(A) Checkpoint soup (weight averaging).** Because every checkpoint is the same adapter,
  you can literally average the adapter tensors of the plateau checkpoints (e.g. 2800–3400)
  into ONE adapter — Model Soups / SWA. Frequently generalises better than any single
  checkpoint at ZERO extra inference cost, and is the cleanest way to capture "best-CE AND
  best-adherence" at once. **Default recommendation; try first.**
- **(B) Per-language checkpoint (or per-language soup) selection — the legitimate "MoE-lite".**
  Your "best per-language eval CE" instinct is the sound part: languages plateau/overfit at
  different steps (the whole Q2 question). The target language is known at inference (it is
  in the prompt), so route on it — pick, per language, the best per-language checkpoint. Real
  mixture keyed on a discrete known variable, ~no extra cost (2–3 adapters resident). Worth it
  ONLY if the eval shows languages peak at different steps; if they all peak together, skip it.
- **(C) Best-adherence subject to a fidelity floor.** Pick argmin(adherence) s.t.
  chrF++ ≥ base_chrF++ − ε. Single model, single pass; guards the cascade's real worry (that
  chasing length silently hurts meaning) without the 2× cost.
- **(D) RL/GRPO (roadmap, later).** The principled fusion of fidelity + length into ONE policy
  via a combined reward — this is the "correct" version of what the cascade approximates, and
  is already the pipeline README's flagged next research step. Multi-week; not now.

Note the pipeline ALREADY realises a two-stage structure, and a stronger one than two LLM
snapshots: the LLM *proposes* (3 candidates at 100/85/65% budgets) and the trained
DurationPredictor *referees* in real milliseconds (reject-overflow + refine-shorter loop in
`isochrony_translation_v3.py`). That referee measures the true objective (time), whereas a
second LLM snapshot would just be another imperfect generator. So the "two specialists"
intuition is best served by generator→referee (already built) + option (A)/(B) for the
generator itself, not by a two-checkpoint LLM cascade.

**Decision recorded:** default deploy = plateau checkpoint-soup (A); fall back to
argmin(adherence)-with-fidelity-floor (C) if the soup underperforms; add per-language routing
(B) only if the eval shows divergent per-language peaks. Cascade shelved; RL is the long-term
fusion.
