# Notebook 02 Interim Gating Evaluation — Per-Language Isochrony Learning & Phoneme-Adherence Report
### Kaggle notebook `02b_llm_eval` (notebook384d15794b, scriptVersionId 336371597) · Generated 2026-07-19
### Training status at evaluation time: 3,801 / 6,670 planned steps = 57.0% complete (checkpoint-3801, epoch 1.14)

---

## 0. What this report is

This is the retroactive/interim run of the phoneme-adherence gating evaluation (STEP 6 of the daily
protocol), triggered early at the user's request rather than waiting for the ≥6,600-step completion
gate. It runs `phoneme_adherence_eval.py` in two passes plus a sanity check, producing three output
folders on Kaggle, all now archived to `files_v3/evaluation/results/`:

- **`eval_out_ce`** (`--mode ce`, cheap, teacher-forced, no generation): per-language completion CE
  across **all 22 checkpoints** (200 → 3801) plus base. n≈120–150 held-out examples per language per
  checkpoint. This is the highest-confidence dataset here — large sample, directly comparable to W&B
  `eval_loss`.
- **`eval_out_all`** (`--mode all`, generation-heavy): full metrics — CE, phoneme adherence, chrF++
  fidelity, length-response slope — but only on **5 checkpoints** (base, 2558, 3200, 3400, 3801), with
  n=15 samples/language for adherence/chrF and n=20 points/language for the length-slope regression.
  This is the only source for adherence/chrF/slope, but the small n means per-language deltas of a few
  points should be read as directional, not exact.
- **`eval_sanity`**: a 5-sample/language smoke test on checkpoint-3801 only (CE 0.4890) — a pipeline
  health check, not an analysis input (excluded below; noted only for completeness).

### 0.1 What these four metrics mean, in plain terms

**CE (cross-entropy / "completion CE").** At each point in a translation, the model outputs a
probability for every possible next word. CE measures how "surprised" the model is by the word that
actually comes next in the correct answer, averaged over the whole sentence. If the model is confident
and right — it puts 90% probability on the correct next word — it's barely surprised, and CE is low. If
it puts only 10% probability on the correct word, it's very surprised, and CE is high. Concretely, a CE
of 0.51 means the model's average probability on the correct token was about e^-0.51 ≈ 60%. The
"perplexity" column (e^CE) restates this as "the model is about as unsure as if it were guessing evenly
among this many options" — perplexity 1.7 means something like "torn between roughly 1–2 plausible next
words on average," which is a good, fluent model; perplexity 50 would mean genuinely guessing. CE is a
proxy for translation quality, not a direct measure of the thing we actually care about for dubbing
(getting the *length* right) — that's why the other three metrics exist.

**Phoneme adherence.** Dubbing has a hard timing constraint: the translated sentence, when spoken aloud,
has to take about the same number of seconds as the original clip. Phonemes are the basic sound units of
speech — "cat" is roughly 3 phonemes (k-ae-t) — and counting them is a reasonable stand-in for how long a
sentence takes to say out loud (more phonemes ≈ takes longer to speak). So the model is given a target
phoneme count (e.g. "produce a Hindi translation with about 40 phonemes") alongside the English sentence,
and phoneme adherence measures how close the model's actual output comes to that target. If it's asked
for 40 phonemes and produces something that phonemizes to 44, the relative error is |44−40|/40 = 10%.
Lower is better — it means the model is actually respecting the timing budget it's been given, not just
translating the sentence however long that happens to come out.

**chrF++ (fidelity).** This checks a different thing entirely: not length, but whether the translation
actually *means* the same thing as a human reference translation. chrF++ compares the model's output and
the human reference character-by-character (technically, overlapping character chunks called n-grams)
rather than whole-word-by-whole-word, which makes it more forgiving of minor spelling or suffix
differences — important for languages like Tamil or Kannada where a single correct word can take several
slightly different valid forms depending on grammar. A higher chrF++ score means the translation is
closer in wording and meaning to what a professional human translator produced; a low score (like the
base model's 22.0, or Odia's earlier 2.7) signals the output was barely a real translation at all. This
metric exists specifically to catch a failure mode the other metrics can't see: a model that hits the
phoneme count perfectly by padding with filler words, or by dropping words to shorten the sentence, would
still score badly here even if its length control looked great.

**Length-response slope.** This is the cleverest of the four, because it isolates length-control from
everything else. Take one single English sentence and ask the model to translate it at five different
requested lengths — say 60%, 80%, 100%, 120%, and 140% of its "natural" length — and measure how much the
*actual output* length moves in response each time. Plot requested length on one axis and produced length
on the other, and fit a line through the five points: the slope of that line is the number. A slope near
1.0 means "every time I ask for a longer or shorter translation, the model's output actually gets longer
or shorter by roughly the right amount" — true obedience to the length instruction. A slope near 0 means
the model ignores the instruction completely and just produces whatever its "default" translation is
regardless of what length was requested — like a thermostat dial that's been disconnected from the
heater: you can turn it wherever you like, the room temperature doesn't change. This is the one metric
that can catch "still learning length control" even after CE has gone flat, because CE mostly reflects
word-choice quality, and a model can get very good at word choice while still not fully obeying the
length knob.

Two questions this was built to answer (from `evaluation/README.md`):

1. **Q1**: aggregate eval CE has plateaued since ~step 3200 — is phoneme-length conditioning still
   being learned regardless?
2. **Q2**: is the plateau uniform, or are some of the 11 languages still improving while others have
   begun to overfit?

---

## 1. Aggregate recap (context)

The full 22-checkpoint CE-only pass (`eval_out_ce`) confirms the plateau already flagged in the Day-3
combined report: CE falls from 0.8195 (base) to a global minimum of **0.5056 at checkpoint-3200**, then
sits flat within noise through checkpoint-3801 (0.5079 → 0.5079 → 0.5075 → 0.5071 across steps
3400/3600/3800/3801 — a span of 0.0023 nats, i.e. noise-level). This matches and sharpens the earlier
finding: **the aggregate plateau is real and has now held for 600+ steps (three additional evals) past
where it was first flagged.**

The `eval_out_all` 5-point subsample reproduces the same shape (0.5167 → 0.5077 → 0.5109 → 0.5100 at
2558/3200/3400/3801) despite a different, smaller eval slice — good cross-validation that the plateau
is not a sampling artifact.

---

## 2. Per-language isochrony / length-conditioning learning report (Q2 + Q1)

### 2.1 Per-language CE trajectory — is the plateau uniform?

Using the full 22-checkpoint `eval_out_ce` series per language, the diagnostic is: take the average CE
over the last four checkpoints (3400/3600/3800/3801) and compare it to the CE at checkpoint-3200 (the
global aggregate minimum). A negative or ~zero delta means the language kept improving (or held its
gain) past the point where the aggregate looks flat; a positive delta is a mild uptick — the per-language
signature of early overfitting.

| Language | CE @ 3200 | avg CE (3400–3801) | Δ vs 3200 | Min CE reached at step | Read |
|---|---|---|---|---|---|
| **as** (Assamese) | 0.4822 | 0.4766 | **−0.0056** | 3600 | Still improving |
| **or** (Odia) | 0.2703 | 0.2693 | **−0.0010** | 3800 | Still improving |
| **kn** (Kannada) | 0.3671 | 0.3676 | +0.0006 | 3801 | Flat / negligible drift |
| **hi** (Hindi) | 0.8859 | 0.8868 | +0.0009 | 3000 | Flat / negligible drift |
| **ml** (Malayalam) | 0.3614 | 0.3632 | +0.0018 | 3200 | Flat / negligible drift |
| **ta** (Tamil) | 0.4166 | 0.4192 | +0.0026 | 3800 | Mild plateau |
| **mr** (Marathi) | 1.0149 | 1.0180 | +0.0031 | 3200 | Mild plateau |
| **bn** (Bengali) | 0.5519 | 0.5557 | +0.0038 | 2800 | Mild plateau |
| **pa** (Punjabi) | 0.4034 | 0.4074 | +0.0040 | 3200 | Mild plateau |
| **te** (Telugu) | 0.3603 | 0.3652 | +0.0049 | 3200 | Mild uptick |
| **gu** (Gujarati) | 0.4482 | 0.4545 | **+0.0063** | 3200 | Clearest uptick |

**Reading this**: the plateau is genuinely non-uniform, but the effect size is small everywhere — the
largest drift (Gujarati, +0.0063 nats ≈ 0.6% perplexity) is an order of magnitude smaller than the
train/eval gap widening already on record for the epoch-2 boundary (that finding compared train loss to
eval loss and saw a 5–6× gap open up; this is an eval-only, checkpoint-to-checkpoint comparison, a much
finer-grained and noisier signal). With that caveat: **Assamese and Odia are the two languages with
clear, unambiguous room left to improve** — both are agglutinative/lower-resource in this dataset and
both had by far the worst base-model starting points (Odia's base CE, chrF≈2.7, was barely above random).
**Gujarati and Telugu show the earliest, if mild, overfitting-style uptick.** The remaining seven
languages sit in a genuine flat plateau with no clear signal either way.

### 2.2 Is phoneme-length conditioning still being learned where CE is flat? (Q1)

The length-response slope (regressing produced phoneme count on requested count across a 0.6–1.4×
sweep of the same sentence; slope→1.0 means true obedience to the length instruction) is the
CE-independent probe built for exactly this question. Aggregate `length_slope` across the fine-tuned
checkpoints: **0.673 (step 2558) → 0.656 (3200) → 0.684 (3400) → 0.687 (3801)** — a small net increase
of +0.014 (≈2%) since 2558, most of it landing in the last 600 steps. This is modest but directionally
real, and — critically — it is a **different trend shape than CE**, which was already flat by 2558. So
the honest answer to Q1 is: **yes, a small amount of length-conditioning improvement is still happening
after CE looks flat, but it is a minor effect, not a strong signal.** It does not by itself justify
continued full-scale training; see §4.

Per-language, the picture is much more mixed than the aggregate suggests, and this is the most
important novel finding of this pass:

| Language | Base slope | ft slopes (2558→3200→3400→3801) | Pattern |
|---|---|---|---|
| or | 0.056 | 0.256 / 0.317 / 0.246 / 0.295 | Learned essentially from zero; still noisy, not saturated |
| pa | 0.349 | 0.706 / 0.693 / 0.680 / 0.709 | Large gain over base, flat since 2558 |
| ta | 0.480 | 0.898 / 0.902 / 0.869 / 0.927 | Large gain, best-in-class obedience, still edging up |
| as | 0.127 | 0.585 / 0.610 / 0.580 / 0.565 | Large gain, flat/mild decline late |
| kn | 0.365 | 0.347 / 0.369 / 0.340 / 0.358 | **No real gain over base at any checkpoint** |
| te | 0.357 | 0.333 / 0.349 / 0.352 / 0.363 | **No real gain over base at any checkpoint** |
| **ml** | **0.456** | **0.366 / 0.385 / 0.308 / 0.330** | **Regresses vs. base at every ft checkpoint** |
| hi | 1.012 (overshoot) | 0.856 / 0.863 / 1.022 / 0.952 | Base already ≈1; ft oscillates around it |
| gu | 1.159 (overshoot) | 1.096 / 0.855 / 1.027 / 1.032 | ft mostly *improves* calibration toward 1.0 |
| mr | 1.252 (overshoot) | 0.958 / 0.839 / 1.112 / 0.972 | ft mostly *improves* calibration toward 1.0, noisy |
| bn | 0.907 | 1.004 / 1.032 / 0.992 / 1.058 | Already good at base; ft pushes to mild overshoot |

Two things stand out. First, **for Gujarati, Marathi, and Hindi the base Llama-3.1 model already
over-responds to length instructions (slope > 1)** — for these, fine-tuning mostly *pulls the slope back
toward the ideal of 1.0* rather than "teaching" length control from nothing; that this shows up as
"ft < base" for Gujarati/Marathi in a raw before/after comparison is a correction, not a regression
(distance-from-1.0 shrinks for both across most checkpoints). Second, and genuinely concerning:
**Malayalam is the one language where fine-tuning makes length-conditioning measurably worse** — its
base slope (0.456) is already closer to the ideal of 1.0 than any fine-tuned checkpoint (0.31–0.39), and
this holds at all four sampled checkpoints, not a one-off. Kannada and Telugu show essentially flat,
weak length-responsiveness that fine-tuning never moved much either direction. These three
(ml/kn/te — all Dravidian) are worth a closer look before Notebook 03/07 treat length control as
uniformly solved; a single change of phonemizer behavior or tokenization quirk for these languages could
explain it and is worth a quick manual spot-check of a few generated outputs.

---

## 3. Phoneme-adherence report

### 3.1 Headline: what fine-tuning bought, aggregate

| Metric | Base model | Best fine-tuned checkpoint | Change |
|---|---|---|---|
| CE / perplexity | 0.819 / 2.41 | 0.506 / 1.70 (ckpt-3200) | −38% CE, −29% perplexity |
| Adherence relErr (\|N_gen−N_req\|/N_req) | 0.495 | 0.103 (ckpt-3801, ckpt-3200 tied at 0.104) | **−79% relative error** |
| chrF++ (fidelity) | 22.0 | 30.3 (ckpt-2558, ckpt-3400/3801 tied ~30.0–30.1) | +37% relative |
| length_slope (obedience, →1.0 ideal) | 0.593 | 0.687 (ckpt-3801) | +16% relative toward ideal |

Fine-tuning delivers a large, unambiguous win on every axis versus the untrained base model — the
headline the paper needs. The phoneme-count adherence error alone dropping by ~79% is the single
strongest number in this evaluation.

### 3.2 Adherence trend across fine-tuned checkpoints — has it plateaued too?

Aggregate `adherence_rel_mean`: 0.124 (2558) → 0.104 (3200) → 0.115 (3400) → 0.103 (3801). This is
non-monotonic and stays inside a 0.103–0.124 band across 1,243 steps (2558→3801) — at n=15 samples/
language this is consistent with **adherence having already plateaued by step 2558**, i.e. earlier than
CE's own plateau at 3200. There is no clean further improvement visible; the 3400 reading (0.115) is
likely just sampling noise given the small n, not a real regression, since 3801 returns to 0.103.

### 3.3 Fidelity (chrF++) trend — a caution flag for two languages

Aggregate chrF++ is flat (30.3→29.8→30.1→30.1 across 2558/3200/3400/3801), consistent with "converged."
But per-language chrF from 2558→3801 shows real divergence:

- **Bengali: 28.3 → 24.4 (−3.9 points)** and **Gujarati: 28.9 → 25.4 (−3.5 points)** — both real fidelity
  regressions over the same span in which Bengali's length_slope crept into overshoot (1.00→1.06) and
  Gujarati's adherence error fell to its best value (0.049). This pattern — length metrics improving or
  holding while chrF drops — is exactly the failure mode the "MoE composite" idea worried about
  (over-optimizing for length at the expense of translation quality), and it is showing up empirically in
  exactly two languages.
- Conversely, **Tamil (+4.8), Hindi (+3.0), Assamese (+2.3)** show real fidelity gains over the same
  span — these three are unambiguously still benefiting from continued training on every axis measured
  (CE, chrF, and for Assamese, still-falling CE per §2.1).
- Marathi (−1.1), Telugu (−1.6), Kannada (−1.4), Punjabi (−1.0), Malayalam (−0.5) show small declines,
  within plausible n=15 sampling noise but worth re-checking with a larger sample before treating as
  settled.

### 3.4 Caveat on statistical confidence

The CE trajectory (§2.1) is the load-bearing evidence here — n≈120–150/language/checkpoint, and it is
the same masking used for W&B `eval_loss`, so it is trustworthy to the third decimal. The
adherence/chrF/length-slope numbers (§2.2, §3.2, §3.3) use only 15–20 samples per language per
checkpoint; deltas under roughly 2–3 chrF points or 0.02–0.03 relErr should be read as directional
signals, not settled facts. The Bengali/Gujarati chrF regression and the Malayalam length-slope
regression are flagged here because they are large enough, and consistent enough across all four
checkpoints, to very likely be real rather than noise — but a targeted re-run with n=40–50 on just these
three languages would firm this up cheaply before it drives any pipeline decision.

---

## 4. Recommendation: continue training beyond 57%, or stop?

**Recommendation: do not resume undifferentiated full-scale training. Treat Notebook 02 as functionally
converged for 8 of 11 languages, and adopt the checkpoint-soup / per-language-routing strategy already
recorded in `evaluation/README.md` rather than spending more quota chasing an aggregate metric that has
been flat for 600+ steps.**

Reasoning, combining every signal above:

- **Aggregate CE has been flat for three consecutive evals (3400/3600/3800/3801), a full 600 steps past
  its minimum at 3200.** That alone would normally be enough to stop.
- **Per-language, only two languages (Assamese, Odia) show an unambiguous, still-improving CE signal**,
  and they are also the two with the most room to gain (they had the worst base-model performance and
  the biggest fine-tuning uplift so far). Tamil shows a smaller version of the same pattern via its
  chrF trend even though its CE is only "mildly plateaued."
- **Adherence plateaued even earlier than CE (~step 2558)**, and the small residual length_slope drift
  (+0.014 aggregate) is real but too small to be a strong argument for continuing broad training — it is
  outweighed by the opposite evidence (Bengali/Gujarati chrF regression, Malayalam length-slope
  regression) that more training is not uniformly helpful and is measurably hurting a couple of
  languages on the axes that matter.
- **Best-checkpoint selection is now solidly triangulated**: best aggregate CE is checkpoint-3200
  (0.5056); best aggregate adherence is checkpoint-3801/3200 (tied ~0.103–0.104); best aggregate chrF is
  checkpoint-2558 (30.3, effectively tied with 3400/3801 at ~30.0–30.1). All four candidates (2558, 3200,
  3400, 3801) are within noise of each other on every metric — this is the strongest evidence yet for
  the README's Option A (checkpoint soup): average the adapters from these four checkpoints rather than
  picking one, since none of them dominates the others.

**Concrete next steps, in order:**

1. **Adopt the checkpoint soup of {2558, 3200, 3400, 3801} as the deploy adapter** for Notebook 03/06/07,
   per the already-recorded default recommendation — this evaluation newly confirms all four are
   statistically tied, which is precisely the situation the soup approach is designed for.
2. **Do not launch another full 4-hour training session against the 6,670-step plan.** The marginal
   value is now concentrated in 2–3 of 11 languages, so an undifferentiated continuation spends a full
   quota-day improving Assamese/Odia/Tamil by a small amount while risking further chrF drift on
   Bengali/Gujarati.
3. **If more training is wanted, make it targeted, not blanket**: a short, cheap continuation
   restricted to up-sampled Assamese/Odia/Tamil examples (Option B, per-language routing, from
   `evaluation/README.md`) would capture the one place real headroom remains, without spending quota on
   the eight already-plateaued languages. This is optional and lower priority than moving to Notebook 03.
4. **Before trusting the Bengali/Gujarati chrF regression and the Malayalam length-slope regression as
   final**, worth a cheap confirmatory re-run (`--mode all` at n=40–50, those three languages only,
   ckpt-2558 vs ckpt-3801) — inexpensive relative to a full training day, and removes the last
   statistical doubt before those findings go in the paper.
5. Notebook 03 (length-augmentation / paraphrase rewriting) can now proceed using the soup checkpoint as
   its base translator — nothing above blocks that work, and this evaluation was explicitly run early
   (before the ≥6,600-step gate) to unblock exactly this decision.

### Findings register update

- **F2** (epoch-2 memorization onset): unchanged/supported — this CE-only comparison is a much finer
  signal than the train/eval gap and shows the effect is small at the eval level (max drift 0.0063 nats)
  even though the train-eval gap widened sharply; both can be true simultaneously (memorization inflates
  train performance a lot while barely moving eval CE).
- **F6** (aggregate eval plateau): confirmed and extended — now flat across 4 consecutive evals
  (3400–3801), strongest evidence yet for early-stop.
- **F7** (phoneme-adherence eval, previously pending): **now delivered.** Headline: −79% relative
  adherence error, +37% chrF, checkpoint-soup-worthy tie among the top 4 checkpoints. New sub-finding
  (call it **F8**): per-language divergence is real but small in CE terms, and the two languages with
  genuine remaining headroom (Assamese, Odia) are not the two with early overfitting signatures
  (Gujarati, Telugu on CE; Bengali, Gujarati on chrF) — the "still improving" and "overfitting" sets are
  disjoint, which is exactly the scenario the per-language routing option was designed for.

---

## 5. Reproducibility

Raw files archived at `files_v3/evaluation/results/`: `eval_out_ce__eval_report.md`,
`eval_out_ce__trajectory_summary.csv`, `eval_out_ce__per_checkpoint_metrics.csv` (242 rows, all 22
checkpoints × 11 languages), `eval_out_all__eval_report.md`, `eval_out_all__trajectory_summary.csv`,
`eval_out_all__per_checkpoint_metrics.csv`, `eval_out_all__length_response.csv` (55 rows, 5 checkpoints ×
11 languages), `eval_sanity__eval_report.md`. Source: Kaggle notebook `notebook384d15794b`
(scriptVersionId 336371597), GPU T4×2, full run ≈4.4 GPU-hours. Method reference:
`files_v3/evaluation/README.md`. Prior context: `files_v3/research_logs/ANALYSIS_02_llm_finetune.md`,
`files_v3/research_logs/reports/combined_report_days1-3_2026-07-18.md`.

---

## 6. Addendum (2026-07-20) — Confirmatory re-run: bn/gu chrF++ and ml length-slope

The three findings in §3.3 flagged as needing confirmation before publication were re-tested at 3×
the sample count (n=45/language vs. the original n=15) on **checkpoint-2558 vs. checkpoint-3801 only**,
all 11 languages, `--mode all`, plus a raw-generation dump for bn/gu/ml (8 samples/language/checkpoint,
48 samples total). Source: Kaggle notebook `notebookaeed9e9468` ("02c — Confirmatory re-eval"),
scriptVersionId 336729124, GPU T4×2, 9624.6s (2h40m) — the long runtime relative to the original interim
pass reflects the 3× sample count across two full-generation checkpoints, not an error. Raw files:
`files_v3/evaluation/results/eval_out_confirm__eval_report.md`,
`eval_out_confirm__per_checkpoint_metrics.csv`, `eval_out_confirm__length_response.csv`,
`eval_out_confirm__samples.jsonl`.

**Verdict per finding:**

1. **Bengali chrF++ regression — REFUTED.** At n=15 the interim pass showed a chrF drop from ckpt-2558
   to ckpt-3801. At n=45 the direction reverses: chrF++ **increases** 25.97 → 27.19 (+1.22 points). The
   raw-generation dump supports this directly — several bn pairs show ckpt-3801 producing more fluent,
   closer-to-reference translations than ckpt-2558 (e.g. "We also know law" chrF 41.8→56.9; "A profound
   change had occurred" chrF 23.1→43.7; "About twice as old" chrF 7.8→21.6). One clear counter-example
   exists ("A mortal is someone who dies," chrF 11.0→4.0, ckpt-3801 degenerates into a repetitive
   near-nonsense phrase), so the checkpoint isn't uniformly better — but the aggregate direction at n=15
   was simply sampling noise. **Conclusion: no real Bengali chrF regression. Drop this from the findings
   register; do not report it in the paper.**
2. **Gujarati chrF++ regression — CONFIRMED, but smaller than originally measured.** At n=45, chrF++
   still falls 28.97 → 27.18 (−1.79 points) from ckpt-2558 to ckpt-3801, versus the −3.5 points measured
   at n=15 — real, but roughly half the originally estimated magnitude. Qualitatively, several ckpt-3801
   gu generations show mild semantic drift relative to the reference that ckpt-2558 doesn't ("And I'm
   lucky enough to have that" → ckpt-3801 renders roughly as "I have to earn from that," chrF 28.2→12.6;
   "children... very different ways" chrF 45.8→37.2), while others improve (chrF 21.7→47.1, 13.7→17.3).
   This mixed-but-net-negative pattern is consistent with the F8 "early overfitting" classification
   already assigned to Gujarati on CE grounds — the chrF dip looks like the same phenomenon, not a
   separate anomaly. **Conclusion: keep this finding, but report the corrected, smaller effect size
   (−1.79, not −3.5) and note it as consistent with mild overfitting rather than a distinct failure mode.**
3. **Malayalam length-slope anomaly — REPRODUCED, root cause now visible qualitatively.** length_slope
   stays low and essentially flat across checkpoints (0.449 → 0.461, both far from the 1.0 ideal — this
   run doesn't re-test the base-model comparison, but the low-and-stable pattern itself replicates
   cleanly at 3× the sample size, so it isn't an n=15 fluke). The raw-generation dump shows *why*: for
   long requested lengths the model stops well short — e.g. one segment requested 122 phonemes and both
   checkpoints generated only 71–72 — while short/medium requests track much closer (43→39, 48→44-53).
   The undershoot is concentrated on long targets, not uniform across the length range, and both fidelity
   (chrF ~14–33, no worse than other languages) and length behavior are consistent between checkpoints —
   this is a **stable, checkpoint-independent Malayalam-specific length-conditioning weakness**, most
   likely a phoneme/token-count mismatch specific to Malayalam's agglutinative orthography (a single
   Malayalam word can span many phonemes per token relative to other Indic scripts, so the model may be
   learning a token-budget rather than a phoneme-budget stop rule). **Conclusion: keep this finding;
   reframe it from "checkpoint anomaly" to "systematic, checkpoint-independent Malayalam weakness,
   concentrated at long requested lengths" — this is now a good candidate for targeted continuation
   (oversample long-N Malayalam examples) rather than something a different checkpoint choice would fix.**

**Net effect on the hand-off:** the checkpoint-soup decision (§4, Option A) is unaffected — none of these
three findings discriminate between individual checkpoints in a way that would favor one over the soup;
if anything, Malayalam's checkpoint-independence *reinforces* that soup vs. single-checkpoint choice
doesn't matter for this weakness. The finding that changes the targeted-continuation plan is Malayalam
now qualifying alongside Assamese/Odia/Gujarati as languages worth oversampling in the next fine-tuning
round (see `ANALYSIS_02_llm_finetune.md` F8 update), specifically with long-N examples for Malayalam.
