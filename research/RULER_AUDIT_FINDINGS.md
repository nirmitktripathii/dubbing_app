# The Ruler Was Wrong: A Corpus-Wide Mislabelling in the Isochrony Fine-Tune

**Date:** 2026-08-04  
**Scoring ruler:** `phonemes:espeak-ng-1.50`  
**Corpus:** Samanantar-derived, 53,350 train / 1,650 val, 11 Indic languages  
**Verdict:** `RETRAIN`  
**Reproduce:** `files_v3/kaggle_runs/02h_ruler_audit.ipynb` (CPU, zero GPU quota)

---

## Summary

The phoneme budgets this model was trained on were never phoneme counts. They were
**non-space character counts**, for every row of the corpus, in all eleven languages.

The cause was a silent fallback in the grapheme-to-phoneme converter. espeak-ng is a
system package installed separately from the `phonemizer` pip package that calls it; it
was absent from the session that generated the corpus, and the converter's
`except Exception: return [c for c in text if not c.isspace()]` caught every row while
logging at DEBUG. Nothing surfaced.

The mislabelling survived two months of training and evaluation because **the same wrong
ruler measured both the target and the result**. Prompts looked correct, loss curves
behaved, and the reported metrics were plausible. No aggregate statistic could have
exposed it. A per-row identity check exposed it immediately.

## Evidence

`n_phonemes` was compared against both candidate rulers computed from each row's own
target text. Across the full corpus:

| split | rows | languages | fraction where `n_phonemes` == non-space char count |
|---|---|---|---|
| train | 53,350 | 11 | **1.00 in every language** |
| val | 1,650 | 11 | **1.00 in every language** |

Not a correlation — an identity. Source: `ruler_audit_before.json`.

## Per-language measurements (train split)

`ph/char` is the mean true-phoneme-to-character ratio. `CV` is its coefficient of
variation *within* the language — the quantity the salvage decision turns on. `label err`
is how wrong each label is, per row, measured in true phonemes. `shift` is how far
relabelling moved that language's mean budget.

| lang | n | ph/char | CV | R² | label err | shift | verdict |
|---|---|---|---|---|---|---|---|
| or | 4,839 | 1.128 | **0.272** | 0.841 | 0.107 | +14.8% | RETRAIN |
| as | 4,839 | 1.105 | **0.191** | 0.889 | 0.097 | +11.2% | RETRAIN |
| bn | 4,855 | 1.039 | **0.176** | 0.952 | 0.071 | +3.5% | RETRAIN |
| ml | 4,851 | 0.941 | **0.130** | 0.956 | 0.111 | -6.1% | RESCALE_THEN_MEASURE |
| te | 4,855 | 1.028 | **0.119** | 0.961 | 0.074 | +2.4% | RESCALE_THEN_MEASURE |
| ta | 4,836 | 0.945 | **0.111** | 0.980 | 0.089 | -4.9% | RESCALE_THEN_MEASURE |
| kn | 4,839 | 1.060 | **0.098** | 0.976 | 0.079 | +5.9% | RESCALE_THEN_MEASURE |
| pa | 4,875 | 1.026 | **0.094** | 0.989 | 0.069 | +3.3% | RESCALE_THEN_MEASURE |
| gu | 4,855 | 1.047 | **0.087** | 0.985 | 0.071 | +4.6% | RESCALE_THEN_MEASURE |
| hi | 4,869 | 0.997 | **0.081** | 0.988 | 0.059 | -0.2% | RESCALE_THEN_MEASURE |
| mr | 4,836 | 0.997 | **0.076** | 0.988 | 0.057 | -0.7% | SALVAGE |

### Why nothing looked broken

Mean phonemes-per-character spans only **0.94–1.13**. The two rulers agree closely
*on average*, so an aggregate length error of 10.3% was entirely plausible and no summary
statistic could have flagged it.

### The finding that changes expectations

Per-row label error runs **5.7%–11.1%**. The model's *entire* previously reported
length error was **10.3%**. A substantial share of what was attributed to the model was
label noise: the target itself was wrong by an amount comparable to the miss being
measured. The fine-tune is therefore likely to be **better** than its own evaluation ever
showed, and the published 0.495 → 0.103 improvement understates rather than overstates it.

### A falsifiable prediction

Assamese and Odia were systematically *under*-labelled — relabelling moves their mean
budgets **+11.2%** and **+14.8%**. The model was told "40" when the truth
was 46, so it should **under-produce** in exactly those two languages. If a corrected-ruler
evaluation does not show `as` and `or` with the most negative signed adherence, the causal
account here is wrong and the retrain decision must be revisited.

### Why the three failures cluster

The languages exceeding the pre-registered `CV > 0.15` retrain threshold are
`or`, `as`, `bn`. `as` and `bn` share the Bengali script; `or` is Odia. These are
orthographies where espeak-ng's schwa-deletion and conjunct handling are least
deterministic with respect to character count — so the character label carried the least
duration information precisely there. `mr` alone clears SALVAGE.

## Decision rule, pre-registered

Written into `tools/ruler_audit.py` **before** any numbers were seen, so the outcome could
not be rationalised afterwards:

| Condition | Verdict | Meaning |
|---|---|---|
| CV ≤ 0.08 in all languages | `SALVAGE` | right capability, wrong unit — a per-language constant recovers it at inference, free |
| CV > 0.15 in any language | `RETRAIN` | label too weakly coupled to duration to have taught obedience |
| otherwise | `RESCALE_THEN_MEASURE` | apply the rescale, let the probe decide |

The 0.08 threshold is not arbitrary: the model's own reported length error is ~10%, so a
rescale whose noise floor exceeds ~8% would add more error than it removes.

**Result: `RETRAIN`**

Per-language conversion constants (phoneme budget ÷ k = the character budget the model was
actually taught), from `budget_scale.json`:

```json
{
  "as": 1.1162,
  "bn": 1.0283,
  "gu": 1.0428,
  "hi": 0.9965,
  "kn": 1.058,
  "ml": 0.9375,
  "mr": 0.9899,
  "or": 1.1715,
  "pa": 1.0391,
  "ta": 0.9576,
  "te": 1.0176
}
```

## G2P accuracy investigation

A separate question: is the *corrected* counter itself accurate? Three configurations were
measured on the full corpus.

| lang | ph/char raw | +IndicNLP | +recompose | CV raw | +IndicNLP | +recompose |
|---|---|---|---|---|---|---|
| or | 1.131 | 1.128 | **1.128** | 0.275 | 0.272 | **0.272** |
| as | 1.147 | 1.604 | **1.105** | 0.265 | 0.421 | **0.191** |
| bn | 1.032 | 1.020 | **1.039** | 0.181 | 0.185 | **0.176** |
| ml | 0.941 | 0.941 | **0.941** | 0.130 | 0.130 | **0.130** |
| te | 1.027 | 1.028 | **1.028** | 0.119 | 0.119 | **0.119** |
| ta | 0.945 | 0.945 | **0.945** | 0.111 | 0.111 | **0.111** |
| kn | 1.060 | 1.060 | **1.060** | 0.098 | 0.098 | **0.098** |
| pa | 1.025 | 1.026 | **1.026** | 0.093 | 0.094 | **0.094** |
| gu | 1.047 | 1.047 | **1.047** | 0.087 | 0.087 | **0.087** |
| hi | 0.997 | 0.997 | **0.997** | 0.081 | 0.081 | **0.081** |
| mr | 0.997 | 1.007 | **0.997** | 0.076 | 0.102 | **0.076** |

**AI4Bharat IndicNLP normalisation alone made Assamese materially worse.** It canonicalises
*toward decomposed* forms — `য়` U+09DF becomes U+09AF + U+09BC — and rewrites 50.8% of
Assamese rows that way. espeak-ng's rule files are written against the *precomposed*
letters, so its `as` voice mis-parses the result: ph/char inflated to 1.604 while Bengali,
same script and the same substitution on 36.3% of its rows, stayed flat at 1.020. Two
languages sharing an orthography cannot legitimately diverge 57%.

`unicodedata.normalize("NFC", ...)` does not fix this — Indic nukta letters are Unicode
**composition exclusions**, so NFC deliberately leaves them decomposed. `recompose_indic()`
rebuilds the mapping from `unicodedata.decomposition()` (44 pairs, derived not hand-listed)
and restores the precomposed letters *after* IndicNLP has done its genuine work: ZWJ/ZWNJ
removal, punctuation canonicalisation, Malayalam chillu handling.

Rows changed by normalisation, per language: ml 55.5%, as 50.8%, bn 36.3%, pa 24.2%,
kn 21.9%, te 20.6%, or 19.6%, ta 11.1%, hi 10.4%, mr 8.1%, gu 3.9%.

### Why espeak-ng remains the engine

`epitran` was evaluated and rejected on evidence:

- **No mapping for `as`, `gu`, `or`** (`DatafileError`) — two of which are the worst-CV
  languages, where a second opinion is most needed.
- **It leaks source graphemes into its own IPA**: Marathi `कॅल्शियम` → `kəॅlɕijmə`, passing
  U+0945 DEVANAGARI VOWEL SIGN CANDRA E through untranslated.
- Inconsistent schwa deletion (`कमल` → `kəməl`, final schwa retained) and lost nasalisation
  (`में` → `men`).

### Inventory validation

A phoneme *count* cannot tell you whether the symbols being counted are phonemes. A phoneme
*inventory* can. `validate_inventory()` flags any output symbol containing a character from
the language's own script — proof of an unmapped grapheme riding through untranslated. This
is what exposes the epitran defect above in one glance.

Result across all 11 languages with espeak-ng: **no leaks**. Inventory sizes:

| lang | distinct symbols |
|---|---|
| as | 84 |
| bn | 83 |
| gu | 104 |
| hi | 107 |
| kn | 73 |
| ml | 74 |
| mr | 99 |
| or | 78 |
| pa | 119 |
| ta | 65 |
| te | 72 |

## Repair

- `53,350` train rows → `53,349` (1 dropped as unphonemizable, logged)
- `1,650` val rows → `1,650`
- Both stamped `ruler: phonemes:espeak-ng-1.50` and re-audited to PASS (`ruler_audit_after.json`)

Fraction of labels changed by relabelling, per language:

| lang | changed | mean before | mean after |
|---|---|---|---|
| as | 89% | 35.0 | 38.9 |
| bn | 83% | 33.8 | 35.0 |
| gu | 86% | 40.1 | 41.9 |
| hi | 86% | 52.2 | 52.1 |
| kn | 90% | 49.4 | 52.3 |
| ml | 94% | 56.4 | 52.9 |
| mr | 84% | 45.0 | 44.7 |
| or | 89% | 39.3 | 45.1 |
| pa | 86% | 46.1 | 47.6 |
| ta | 92% | 50.8 | 48.3 |
| te | 88% | 46.6 | 47.8 |

## Impact on previously published numbers

Every figure in `eval_out_all__eval_report.md` was produced with the character ruler on
both sides, and must be treated as unverified until re-measured:

| Figure | Status |
|---|---|
| length error 0.495 → 0.103 | measured with the character ruler; direction almost certainly holds, magnitude unverified |
| chrF++ 22.0 → 30.1 | **unaffected** — chrF++ compares text to text and never touches the phoneme count |
| signed error +0.290 → −0.043 | unverified; the sign is what downstream depends on |
| length slope 0.593 → 0.687 | **two problems**: measured with the wrong ruler, *and* it is the population estimator (regressed across different sentences), while the surrounding prose describes the probe (sentence fixed, budget swept). A budget-ignoring model still scores well on the population estimator. |
| résumé claim "79% relative reduction" | inherits the length-error caveat |

## Reproduction

```bash
# CPU, zero GPU quota
python -m tools.ruler_audit --jsonl <train.jsonl> <val.jsonl> --out ruler_audit.json
python -m tools.relabel_dataset --in <train.jsonl> --out train.phonemes.jsonl
python -m tools.ruler_audit --jsonl train.phonemes.jsonl   # must print PASS

# offline guards, no espeak/GPU needed
python tests/test_phoneme_ruler.py                          # 27/27
```

`train.phonemes.jsonl` (31 MB) is not committed — it is deterministically regenerable from
the notebook. `val.phonemes.jsonl` and both manifests are committed.

## Open questions

1. **Odia.** `CV 0.272` is unmoved by normalisation, so its instability is not an encoding
   artefact. It is the language where the phoneme count is least well-behaved.
2. **Which counter best predicts real duration.** Internal consistency cannot settle this;
   it needs aligned audio — the same evidence the Phase 03 aligner gate requires, so the
   two should be one session.
3. **`forced_alignment.py` has only ever been validated on synthetic audio**, and it
   produces both the duration predictor's labels and the CTC checks in `train_tts.py`. A
   biased aligner there would be invisible, because every downstream stage would learn the
   bias. Same defect class as this one, sitting upstream of the entire TTS phase.

## The pattern

Every defect found in this investigation has the same shape: **a check that passed for the
wrong reason.**

| Check | Why it passed | What was true |
|---|---|---|
| "the phonemizer works" | it returned non-empty output | it had fallen back to characters |
| "the labels look sane" | letters ≈ sounds on average | per-row error 5.7–11.3% |
| "sacrebleu scored it" | no exception raised | `import` was inside the `try`; 440 silent `None`s |
| "the slope is rising" | a slope was computed | it was the confounded estimator |
| "normalisation improves accuracy" | it fixed the encoding | it broke espeak's Assamese voice |
| "espeak has an Assamese voice" | it does | it cannot parse decomposed nukta |

The defence is the same in every case: **do not test that the machinery ran — test that the
output has the property you need.** Not "did `phonemize()` return something" but "are these
symbols demonstrably not the input's own letters". Not "is there a checkpoint" but "did the
step counter advance".

## Artifacts

| Path | Contents |
|---|---|
| `research/evaluation/results/ruler_audit/ruler_audit_before.json` | full per-language audit of the original corpus |
| `research/evaluation/results/ruler_audit/ruler_audit_after.json` | post-repair audit, verifying PASS |
| `research/evaluation/results/ruler_audit/DECISION.json` | verdict + per-language constants |
| `research/evaluation/results/ruler_audit/budget_scale.json` | phonemes-per-character constants |
| `research/evaluation/results/ruler_audit/*.manifest.json` | relabelling statistics per language |
| `research/evaluation/results/ruler_audit/val.phonemes.jsonl` | repaired validation set |
| `pipeline_v3/common/phonemes.py` | the canonical counter |
| `pipeline_v3/tools/ruler_audit.py` | the audit, with pre-registered thresholds |
| `pipeline_v3/tools/relabel_dataset.py` | the repair |
| `pipeline_v3/tests/test_phoneme_ruler.py` | 27 offline guards |
| `PHASE02_REPAIR_PLAN.md` | sequenced plan with decision gates |
