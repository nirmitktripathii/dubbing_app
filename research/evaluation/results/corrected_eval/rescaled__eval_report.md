# Phoneme-Adherence Evaluation Report

- **Ruler:** `phonemes:espeak-ng-1.50`
- **Val set:** `/kaggle/input/notebooks/nktthegreat/02h-ruler-audit/data/val.phonemes.jsonl` (1650 rows)
- **Generated:** 2026-08-04T07:44:07.355074+00:00
- **Adherence samples/lang:** 40  **Probe sentences/lang:** 25
- **Budget rescale ACTIVE** (phoneme budget converted to the character budget the model was taught): `{'as': 1.1162, 'bn': 1.0283, 'gu': 1.0428, 'hi': 0.9965, 'kn': 1.058, 'ml': 0.9375, 'mr': 0.9899, 'or': 1.1715, 'pa': 1.0391, 'ta': 0.9576, 'te': 1.0176}`

> Read the per-language table first. Eleven languages across two families do not plateau together; every aggregate number below hides that.

## Per-language

### checkpoint-3801 (step 3801)

| Lang | relErr | signed | probe slope | pop slope | chrF++ | semantic | degraded |
|---|---|---|---|---|---|---|---|
| as | 0.104 | -0.070 | 0.685 | 0.790 | 33.0 | 0.849 | 0.300 |
| bn | 0.115 | -0.035 | 0.866 | 0.763 | 26.5 | 0.779 | 0.500 |
| gu | 0.107 | -0.048 | 0.920 | 0.738 | 28.5 | 0.717 | 0.575 |
| hi | 0.072 | -0.012 | 0.898 | 1.057 | 38.3 | 0.795 | 0.425 |
| kn | 0.116 | -0.047 | 0.435 | 0.522 | 31.2 | 0.922 | 0.125 |
| ml | 0.141 | -0.074 | 0.411 | 0.527 | 27.2 | 0.884 | 0.200 |
| mr | 0.113 | -0.006 | 0.835 | 0.897 | 32.7 | 0.768 | 0.575 |
| or | 0.204 | -0.107 | 0.363 | 0.464 | 26.3 | 0.832 | 0.375 |
| pa | 0.152 | -0.041 | 0.386 | 0.443 | 33.8 | 0.835 | 0.325 |
| ta | 0.132 | 0.013 | 0.803 | 0.565 | 34.1 | 0.865 | 0.275 |
| te | 0.115 | -0.021 | 0.391 | 0.555 | 32.4 | 0.869 | 0.200 |

## Aggregate (read second)

| Checkpoint | Step | CE | PPL | relErr | signed | chrF++ | semantic | probe slope | pop slope |
|---|---|---|---|---|---|---|---|---|---|
| checkpoint-3801 | 3801 | - | - | 0.125 | -0.041 | 31.3 | 0.828 | 0.636 | 0.666 |

## Stopping verdict

**INSUFFICIENT_DATA** — need at least two checkpoints

> `length_slope_probe` holds the sentence fixed and sweeps only the budget: it is the capability measurement. `length_slope_population` regresses across different sentences and is confounded by sentence length — a model that ignores the budget entirely still scores high on it. Never select a checkpoint on the population slope.

