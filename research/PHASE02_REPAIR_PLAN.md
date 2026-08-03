# Phase 02 — Repair Plan

**Status: the pipeline is not where the previous plan said it was.** That plan was written
before the corpus was audited. The audit changed the diagnosis, and this document
supersedes it.

The earlier framing was "checkpoint 3801 stopped early on the wrong metric; re-diagnose on
slope and decide whether to continue." That framing assumed the labels were correct. They
are not. Every conclusion downstream of them — including which languages are "overfitting"
— was drawn from numbers measured in the wrong unit.

---

## 1. What was actually wrong

Five defects. The first is the root cause; the rest are why it survived two months.

### D1 — The corpus is labelled in characters, not phonemes ⚠️ root cause

`translation/duration_predictor.phonemize_text` ended:

```python
except Exception as e:
    logger.debug("Phonemization failed ...; falling back to characters.")
return [c for c in text if not c.isspace()]
```

espeak-ng is a **system** package; `pip install phonemizer` does not install it. It was
absent from the corpus-generation session, so the fallback fired for every row and logged
at DEBUG. Nothing surfaced.

**Evidence** (`data/translation_dataset/train_final.jsonl`, 1,289 base rows):

| | rows | `n_phonemes` == non-space char count | chars/label ratio |
|---|---|---|---|
| base rows (from `dataset_generator.py`) | 1,289 | **1,289 (100.0%)** | min 1.000, max 1.000 |
| augmented rows (from `length_augmentation.py`) | 1,064 | 89 (8.4%) | 0.198 – 1.500 |

Exactly 1.000 at both extremes over 1,289 rows is not a coincidence — it is an identity.

### D2 — The corpus mixes two rulers, which is worse than one wrong ruler

The augmentation session (2026-07-26) *did* have espeak-ng, so those 1,064 rows carry real
phoneme counts. Both row types use the same prompt token, `[Target Phonemes: N]`.

A uniformly mislabelled corpus teaches one consistent task in the wrong unit — recoverable
by rescaling. A **mixed** corpus teaches two contradictory tasks under one token, and the
model can only split the difference. This is the most likely reason length slope sits at
0.687 instead of near 1.0, and it is not a modelling problem at all.

### D3 — The augmentation length gate compared incompatible units

`augment_example` admitted a paraphrase if
`(variant_phonemes - source_phonemes) / source_phonemes` moved ≥10% in the intended
direction. `variant_phonemes` was a real phoneme count; `source_phonemes` was read from
the base row's `n_phonemes`, a character count. Verified: `augmentation.source_phonemes`
equals the base row's character count in **1,064 / 1,064** rows.

That expression is not a relative length change — the units do not cancel. Both its sign
and its magnitude were untrustworthy, so the gate admitted and rejected rows arbitrarily.

### D4 — chrF++ was being swallowed

```python
def chrf_pp(hypothesis, reference):
    try:
        import sacrebleu                    # ← import inside the try
        return float(sacrebleu.sentence_chrf(...).score)
    except Exception:                       # ← bare
        return None
```

The 02g log shows a pip failure at 94.2 s. sacrebleu never installed, `ImportError` was
caught, `None` was returned 440 times without one log line, and the report was written with
a blank fidelity column — the one column that says whether the length constraint is being
paid for out of meaning.

### D5 — Two slope estimators; the report used the wrong one

Not duplicates — they measure different things:

| | varies | measures |
|---|---|---|
| `eval_checkpoint` → the reported number | **different sentences**, each with its own budget | whether translations are appropriately scaled — a fluency property |
| `length_response_probe` → written to a separate CSV, never in the report | **the budget only**, sentence held fixed | budget obedience — the actual capability |

A model that ignores the budget entirely still scores high on the first: longer English
produces longer Hindi regardless. The headline `0.593 → 0.687` is the confounded
estimator, while the prose describing it describes the probe. Additionally
`--probe_sentences_per_lang` defaulted to 10, so per-language orderings (te 0.282 vs
ml 0.374) were not trustworthy.

Minor, same file: `summarize` used the population variance divisor and returned
`xs_sorted[n // 2]` — the upper-middle value — as the median for even *n*.

---

## 2. What is now fixed

| File | Change |
|---|---|
| `common/phonemes.py` **(new)** | The one canonical counter. No fallback — raises. Preflight asserts output is phonemes, not passthrough. `ruler_id()` stamps every artifact. |
| `tools/ruler_audit.py` **(new)** | Detects which ruler wrote a corpus; fits chars→phonemes per language; emits a **pre-registered** SALVAGE / RETRAIN verdict. |
| `tools/relabel_dataset.py` **(new)** | Recomputes labels, rewrites prompts, re-validates the augmentation gate in consistent units, writes a provenance manifest. |
| `tests/test_phoneme_ruler.py` **(new)** | 18 tests, no GPU/espeak needed. Pins the passthrough check and the stopping rule. |
| `evaluation/phoneme_adherence_eval.py` | chrF++ un-swallowed; both slopes named and the **probe** reported; probe budgets derived from text not labels; sample stats; per-language table **first**; semantic-vs-anchor scoring; batched generation; `--budget_scale_json` salvage path; stopping rule as code. |
| `training/train_translation_llm.py` | Resume provenance line; `expect_resume` hard gate; missing-`scheduler.pt` error; post-run progress assertion; `assert_corpus_ruler` refuses a non-phoneme-ruled corpus. |
| `training/dataset_generator.py` | Canonical counter; preflight before emitting a row; drops unmeasurable rows; stamps `ruler`. |
| `training/length_augmentation.py` | Word-count fallback removed; source measured with the same function as the variant; preflight; stamps `ruler`. |
| `translation/duration_predictor.py` | `phonemize_text` delegates and raises. Inference degrades **visibly** via a counter and an ERROR, not silently. |

Local test suite: **18/18 passing**.

---

## 3. Sequenced plan

Each session has one question and one gate. Do not start a session whose gate has not been
read.

### Session A — RESULT: **RETRAIN** ✅ ran 2026-08-03, CPU, 0 GPU quota

Notebook `02h-ruler-audit`, self-contained. Audited the real corpus:
`01-wiring-and-dataset/data/translation_dataset/{train,val}.jsonl` — 53,350 / 1,650 rows.

**D1 confirmed at full scale.** `n_phonemes == non-space character count` for **1.00** of
rows in **every one of the 11 languages**, across all 53,350 training rows. Not a subset
artefact.

**D2 needs correcting, and in the project's favour.** `train.jsonl` — the file the
3,801-step run actually consumed (53,350 rows, matching the documented size, with
`n_augmented = 0`) — is *uniformly* character-ruled. It is **not** mixed. The two-ruler
mixture is confined to `train_final.jsonl`, the augmented file built later by notebook 03.
So the model was taught **one consistent wrong unit**, not two contradictory ones. That is
a materially less damaging situation than stated before the audit ran.

| lang | =chars | ph/char | CV | R² | label err | relabel shift | verdict |
|---|---|---|---|---|---|---|---|
| or | 1.00 | 1.131 | **0.275** | 0.839 | 0.108 | +15.2% | RETRAIN |
| as | 1.00 | 1.147 | **0.265** | 0.800 | 0.113 | +16.5% | RETRAIN |
| bn | 1.00 | 1.032 | **0.181** | 0.949 | 0.075 | +2.9% | RETRAIN |
| ml | 1.00 | 0.941 | **0.130** | 0.957 | 0.111 | -6.1% | RESCALE_THEN_MEASURE |
| te | 1.00 | 1.027 | **0.119** | 0.961 | 0.074 | +2.3% | RESCALE_THEN_MEASURE |
| ta | 1.00 | 0.945 | **0.111** | 0.980 | 0.089 | -4.9% | RESCALE_THEN_MEASURE |
| kn | 1.00 | 1.060 | **0.098** | 0.977 | 0.079 | +6.0% | RESCALE_THEN_MEASURE |
| pa | 1.00 | 1.025 | **0.093** | 0.989 | 0.068 | +3.3% | RESCALE_THEN_MEASURE |
| gu | 1.00 | 1.047 | **0.087** | 0.985 | 0.071 | +4.5% | RESCALE_THEN_MEASURE |
| hi | 1.00 | 0.997 | **0.081** | 0.988 | 0.059 | -0.2% | RESCALE_THEN_MEASURE |
| mr | 1.00 | 0.997 | **0.076** | 0.988 | 0.057 | -0.7% | SALVAGE |

**Why nothing ever looked broken.** Mean phonemes-per-character sits between 0.94 and 1.18
— the two rulers agree closely *on average*. Aggregate length error of 10.3% was therefore
entirely plausible, and no summary statistic could have exposed this. Only a per-row
identity check could.

**The finding that changes expectations.** Per-row `label err` — how wrong each label is in
true phonemes — runs **5.7% to 11.3%**. The model's entire reported length error was
**10.3%**. A substantial share of what was measured as *model* error is *label* noise. The
model is probably more obedient than the evaluation has ever shown, and the headline
0.495 → 0.103 understates the fine-tune rather than overstating it.

**A falsifiable prediction for Session B.** Assamese and Odia were systematically
*under*-labelled: relabelling moves their mean budget **+16.5%** and **+15.2%**. The model
was told "40" when the truth was 46, so it should run systematically **short** in exactly
those two languages. If Session B does not show `as` and `or` with the most negative signed
adherence, this causal story is wrong and the diagnosis needs revisiting.

**Why the three failures cluster.** The three languages over the CV > 0.15 retrain
threshold are `or` (0.275), `as` (0.265), `bn` (0.181). `as` and `bn` share the Bengali
script; `or` is Odia. These are orthographies where espeak-ng's schwa-deletion and conjunct
handling are least deterministic with respect to character count — so the character label
carried the least duration information precisely there. `mr` alone clears SALVAGE (0.076).

**Artifacts** (`files_v3/kaggle_runs/results_02h_ruler_audit/`):
`data/train.phonemes.jsonl` (53,349 rows — one unphonemizable row dropped and logged),
`data/val.phonemes.jsonl` (1,650), both stamped `ruler: phonemes:espeak-ng-1.50` and
re-audited to PASS. `audit/budget_scale.json` holds the per-language constants.

**Bugs found and fixed during this session:** `ruler_id()` returned a tuple-formatted
version, `phonemes:espeak-ng-(1, 50)` — a comma inside a provenance string that gets
stamped into every row and would silently shift every column of any CSV it landed in. The
eval harness's `_write_csv` also joined fields by hand rather than quoting; it now uses
`csv.writer`. A provenance field that can corrupt its own container is not provenance.

### Phoneme counter accuracy — IndicNLP normalisation (added 2026-08-03)

The counter is now **AI4Bharat IndicNLP normalisation -> nukta recomposition -> espeak-ng
G2P**, with a phoneme-inventory validator. Measured, three-way, on all 53,350 rows:

| lang | ph/char raw | +IndicNLP | +recompose | CV raw | +IndicNLP | +recompose | R² final |
|---|---|---|---|---|---|---|---|
| as | 1.147 | 1.604 | **1.105** | 0.265 | 0.421 | **0.191** | 0.889 |
| bn | 1.032 | 1.020 | 1.039 | 0.181 | 0.185 | **0.176** | 0.952 |
| mr | 0.997 | 1.007 | 0.997 | 0.076 | 0.102 | **0.076** | 0.988 |
| or | 1.131 | 1.128 | 1.128 | 0.275 | 0.272 | 0.272 | 0.841 |
| gu, hi, kn, ml, pa, ta, te | — | unchanged | unchanged | — | unchanged | unchanged | 0.956–0.989 |

**Normalisation alone made Assamese worse, and that is the interesting part.** IndicNLP
canonicalises *toward decomposed* forms: `য়` U+09DF becomes U+09AF + U+09BC. It rewrites
50.8% of Assamese rows this way. espeak-ng's rule files are written against the
*precomposed* letters, so its `as` voice mis-parses the result — ph/char inflated to 1.604
while Bengali, same script and the same substitution on 36.3% of its rows, stayed flat at
1.020. Two languages sharing an orthography cannot legitimately diverge 57%.

`unicodedata.normalize("NFC", ...)` does not fix this: Indic nukta letters are Unicode
**composition exclusions**, so NFC deliberately leaves them decomposed.
`recompose_indic()` therefore rebuilds the mapping from `unicodedata.decomposition()` (44
pairs, derived not hand-listed) and restores the precomposed letters *after* IndicNLP has
done its genuine work — ZWJ/ZWNJ removal, punctuation canonicalisation, Malayalam chillu
handling.

Net effect: Assamese CV **0.265 → 0.191** (−28%) and R² **0.800 → 0.889**, better than
either the raw baseline or normalisation alone. Marathi returns to its raw values.
Everything else is unchanged, which is the correct outcome — their normalisation deltas
were punctuation and zero-width joiners, which carry no phonemes.

**Why espeak-ng remains the G2P engine.** epitran was evaluated and rejected on evidence,
not preference: it has no mapping for **`as`, `gu`, `or`** (`DatafileError`) — two of which
are the worst-CV languages where a second opinion is most needed — and it leaks source
graphemes into its own IPA (Marathi `कॅल्शियम` → `kəॅlɕijmə`, passing U+0945 CANDRA E
through untranslated). The proposed snippet's `t[1] == 1` filter also returns zero
phonemes: that tuple index is an uppercase flag, not a phoneme indicator.

**Inventory validation.** `phoneme_inventory()` / `validate_inventory()` flag any output
symbol containing a character from the language's own script — proof of an unmapped
grapheme riding through untranslated. This is what exposes the epitran defect above in one
glance, and a phoneme *count* never could. Result across all 11 languages with espeak-ng:
**no leaks**. The inventories run 65–119 distinct symbols per language.

**Still open: Odia.** CV 0.272 is unmoved by normalisation, so its instability is not an
encoding artefact. It is the one language where the phoneme count is least well-behaved,
and the decisive test — which counter best predicts real spoken duration — needs aligned
audio, which is the same evidence the Phase 03 aligner gate requires.

### Session B — re-diagnose on the corrected metrics (GPU, ~2–3 h)

```bash
python -m evaluation.phoneme_adherence_eval \
  --val_jsonl data/val.phonemes.jsonl \
  --checkpoints_glob "checkpoints/translation_llm/checkpoint-*" \
  --base_baseline --output_dir eval_out_corrected --mode all \
  --budget_scale_json audit/budget_scale.json \
  --probe_sentences_per_lang 30 --batch_size 8
```

This is the decisive session. It produces, per language, on the right ruler:
probe slope, signed adherence, chrF++, semantic-vs-anchor, and a degraded-segment rate.

Three things it settles that are currently open:

1. **Whether "overfitting" was ever real.** It was diagnosed on validation CE — the metric
   the article itself proves is blind to this objective. Expect a good part of that table
   to be overturned, which shrinks the catastrophic-forgetting concern correspondingly.
2. **Whether 3,801 was early.** The stopping rule now runs as code against the probe slope.
3. **Whether 02f's targeted pass helped or hurt.** Currently unanswerable: the reported
   comparison used the confounded estimator on both sides.

**Gate:** if `stopping_verdict` says CONTINUE → Session C. If STOP → skip to Session D.

### Session C — retrain or continue (GPU, ~7.5 h) — *only if B says so*

Uniform sampling, unchanged. **No per-language tilt** — it moves shared adapter capacity
away from the other languages and makes the second half of the run a different experiment
from the first, so nothing before 3,801 stays comparable.

Set `expect_resume: true`. If `scheduler.pt` is missing the run now errors rather than
restarting warmup and spiking the LR back to 2e-4 — which is how a continuation destroys
what the first half learned.

If Session A returned RETRAIN, this is a fresh run on `train.phonemes.jsonl`, not a resume.

### Session D — per-language checkpoint selection (GPU, ~2 h)

Divergent per-language schedules do **not** require divergent training. Train one shared
run uniformly, checkpoint densely, then select per language over the trajectory. LoRA
adapters at r=16 are tens of megabytes and the target language is known at dub time, so
per-language adapters can be drawn from the same run — zero interference, because the
training distribution never changed.

Soup only across checkpoints from one continuous run; weights from different runs are not
directly averageable.

### Session E — semantic gate at corpus scale (GPU, ~2 h)

Already produced as a column by Sessions B/D. What remains is the tied-pair confirmation:
that the gate separates a faithful compression from a gutted one (+0.951 / −0.229) on real
data, not just in the unit test.

### Then, and only then — the Phase 03 gate

`common/forced_alignment.py` has only ever been validated on **synthetic** audio, and it
produces both the duration predictor's labels and the CTC checks inside `train_tts.py`. A
biased aligner is a biased duration predictor is a mistimed dub, and the bias would be
invisible because everything downstream would learn it.

**Validate the aligner against hand-labelled real audio before any TTS training.** This is
the same class of defect as D1 — a measurement function trusted without being checked —
and it sits upstream of the entire remaining pipeline.

---

## 4. Misconceptions, corrected

| # | Belief | Correction |
|---|---|---|
| 1 | The labels are phoneme counts | They are character counts, at 100.0%, and the corpus mixes both units |
| 2 | CE/perplexity plateau ⇒ training is done | CE is a which-words-in-which-order measurement; slope was still climbing at the stop point |
| 3 | Rising val CE ⇒ that language is overfit | Unestablished — diagnosed on the metric that is blind to this objective. Session B re-tests it |
| 4 | Slope 0.687 means the model half-learned length | Partly it means the model was taught two contradictory units. Fix the corpus before concluding anything about capability |
| 5 | Divergent schedules ⇒ per-language training | A selection-time problem. Same run, different checkpoints, per-language adapters |
| 6 | Sampling tilt is a cheap fix | It moves capacity away from the other languages *and* destroys comparability with the first half of the run |
| 7 | Semantic preservation needs a better model | Inference-time selection is built and was never measured. Most of the available win needs no retraining |
| 8 | Resume works because the README says so | The glob depth and scheduler restoration are two separate silent failures; both are now gated |
| 9 | Targeted per-language passes are better | 11× cost, shared-adapter interference, full re-eval after every pass |

---

## 5. What is blocking right now

Everything above is built and tested locally (18/18). The only outstanding step is Kaggle
auth, which needs you for one command:

```bash
kaggle auth login
```

OAuth browser flow — no token for anyone to store. Then Session A is fully scripted:

```bash
python -m tools.kaggle_session --discover   # find which notebook output holds the corpus
python -m tools.kaggle_session --push       # push + run, CPU only, 0 GPU quota
python -m tools.kaggle_session --status     # poll
python -m tools.kaggle_session --pull       # results + DECISION.json
```

### Environment fixes applied along the way

- **TLS interception.** Avast's Web/Mail Shield re-signs every HTTPS connection with
  `CN=Avast Web/Mail Shield Root`, which is in the Windows store but not in certifi's
  bundle — so pip and the Kaggle API both failed `CERTIFICATE_VERIFY_FAILED` while the
  same URLs loaded fine in a browser. Fixed with `truststore`, which validates against the
  OS store. Note what that is *not*: `verify=False` would have "worked" and would also
  have accepted a genuinely hostile certificate.
- **Namespace shadowing.** `E:\Dubbing app\kaggle\` is a directory without
  `__init__.py`, so from the repo root `import kaggle` resolves to it and *succeeds* while
  `kaggle.api...` fails. An early check passed for the wrong reason — the same shape of
  defect as D1. `tools/kaggle_session.py` now detects and names it.
