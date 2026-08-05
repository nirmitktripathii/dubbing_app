# Which model manufactures the elastic corpus

Measured 2026-08-05. Same 66 sentences (6 per language × 11), same three scales
(0.60×, 0.75×, 1.30×), same seed, same gates. 198 requested rewrites per model.

A rewrite is kept only if it is **in the right script** and **lands within 15% of the
requested phoneme count**, measured with `common/phonemes.py`. The semantic gate is not
applied here — the embedder segfaults on the dev machine, so gate 2 runs separately on
Kaggle and is being calibrated per language after the finding below.

## Yield

| language | gemini-3.1-flash-lite | gemini-3.5-flash-lite | gemma-4-31b-it | gemma-4-26b-a4b-it |
|---|---|---|---|---|
| as | 44% | 22% | 50% | 39% |
| bn | 33% | 33% | 44% | **56%** |
| gu | 17% | 6% | 28% | **56%** |
| hi | 11% | 11% | 44% | 50% |
| kn | 28% | 17% | 44% | **56%** |
| ml | 22% | 28% | 33% | 50% |
| mr | 6% | 6% | 50% | **56%** |
| or | 28% | 44% | 33% | **56%** |
| pa | 33% | 17% | 22% | 33% |
| ta | 17% | 22% | **67%** | 44% |
| te | 22% | 33% | **61%** | 17% |
| **all** | **24%** | **22%** | **43%** | **46%** |
| compress share | 91% | 93% | 67% | 62% |

**The Gemma models are roughly twice the yield of the Gemini flash-lite models.** That was
not predictable from parameter count or release order, which is why it was measured.

`gemma-4-26b-a4b-it` wins overall at 46% and has the healthiest direction balance — 62%
compression, essentially the 60% the corpus plan asks for. The flash-lite models return
almost only compressions (91–93%), so they would satisfy the direction floor while
starving the expansion side.

**Decision:** `gemma-4-26b-a4b-it` as the primary generator, `gemma-4-31b-it` as a second
pass for whatever falls short. The tool is resumable and per-language, so a second pass
composes naturally. This matters because the per-language winner is not the overall winner:
Telugu yields 61% on 31b and 17% on 26b, Tamil 67% against 44%. No single model is best
everywhere, and Punjabi is weak on all four (17–33%) — worth watching as a possible
language-specific problem rather than a model one.

Gemma's free-tier quota also happens to be the highest of the four (30 RPM / 14,400 RPD
against flash-lite's ~1,000 RPD), so the throughput and the quality answer agree.

## Why the rejections happened — and what it says

Of 106 rejections in the 26b run:

| reason | count | share |
|---|---|---|
| missed length: **too long** | 93 | 88% |
| missed length: too short | 13 | 12% |
| wrong script | **0** | 0% |
| code-mixed | **0** | 0% |
| moved too little | 0 | 0% |

Two things follow, and neither is "loosen the tolerance".

**Nothing drifted out of script.** The script gate cost nothing here, but it stays: it is
the gate that would catch a silent regression, and espeak will happily phonemize Latin text
through an Assamese voice and return a plausible number.

**The failure is systematic, not random.** 88% of misses are in one direction — the models
under-compress. Asked for 60% of the original length they return something above 69%. A
systematic bias is not a reason to widen the gate; it is a reason to **pre-compensate the
request**. Ask for a more aggressive scale than you want, measure where it actually lands,
and the same generator clears the same gate far more often. That is the next change, and it
should raise yield without touching a single threshold.

## Follow-up: the "systematic bias" was oversold, and correcting it does nothing

The 88%-too-long figure above came from three scales, two of which were compressions. Re-run
across 0.5×–1.45× (433 landings, every candidate recorded including rejects), the profile is
much more two-sided:

| | bake-off (0.6 / 0.75 / 1.3) | wide probe (0.5 … 1.45) |
|---|---|---|
| missed: too long | 88% | 59% |
| missed: too short | 12% | 33% |

So "the model systematically under-compresses" was largely an artifact of the scale set used
to measure it. A real bias remains — mean `achieved − asked` = **+0.081** — and the fit is

```
achieved = -0.080 + 1.174 x asked     R2 = 0.62, n = 433
```

The slope above 1 says the generator *over-responds*: it exaggerates whatever change is
asked for. Per language the slope runs 0.888 (ml) to 1.603 (gu), with R² between 0.48 (ta)
and 0.86 (hi).

**A/B, same 66 sentences, same scales, same gates, one arm asking for the target scale and
one asking for the calibrated scale:**

| arm | rows | of 198 | yield | mean \|landing error\| | compress |
|---|---|---|---|---|---|
| uncalibrated | 109 | 198 | 55% | 0.181 | 63% |
| calibrated | 109 | 198 | **55%** | 0.164 | 60% |

**Identical yield.** Pre-compensation shrank the mean landing error by 9% and produced
exactly zero extra rows. The bias was never the binding constraint — the residual spread is,
and shifting a distribution by 0.081 does not change how much of it falls inside a ±15%
window.

Decision: **do not use the scale calibration.** It buys nothing measurable and adds a
per-language fit that can misfire where R² is low. The fitting tool and the raw-landing
recorder are kept, because they are how this was answered and how it would be re-answered
for a different generator.

Worth noting for its own sake: yield was 55% here against 46% in the bake-off, on different
sentences. Sample-to-sample variance at this size is large, which is why both arms of the
A/B ran on identical sentences.

## Arithmetic for the full run

At 46% yield, the 6,334 rows the corpus plan needs require ~13,800 requested rewrites =
~4,600 sentences = **~920 requests** at five sentences per call. That fits inside a single
day on Gemma's quota with room to spare, and would only just fit on flash-lite's. Bias
pre-compensation should reduce it further.

Caveat worth recording: the Gemma calls were substantially slower per request than
flash-lite's. The full run is a long background job, not an interactive one.
