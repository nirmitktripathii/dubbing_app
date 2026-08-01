Phoneme-Adherence Evaluation Report

| Checkpoint | Step | CE | Perplexity | relErr | signed | chrF++ | length_slope |
|---|---|---|---|---|---|---|---|
| checkpoint-2558 | 2558 | 0.5167 | 1.722 | 0.110 | -0.010 | 31.0 | 0.639 |
| checkpoint-3801 | 3801 | 0.5100 | 1.710 | 0.093 | -0.033 | 31.1 | 0.677 |

Source: Kaggle notebook `notebookaeed9e9468` ("02c — Confirmatory re-eval"), scriptVersionId 336729124,
run 2026-07-20, GPU T4 x2, 9624.6s (2h40m). n=45/language (3x the original n=15 interim pass), all 11
languages, checkpoint-2558 vs checkpoint-3801 only. Purpose: confirm/refute the Bengali/Gujarati chrF++
regression and Malayalam length-slope anomaly flagged in `reports/final_eval_report_2026-07-19.md` at
n=15 before treating them as settled findings.
