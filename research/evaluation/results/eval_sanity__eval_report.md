# Phoneme-Adherence Evaluation Report (sanity check; checkpoint-3801 only, 5 samples/lang)

| Checkpoint | Step | CE | Perplexity |
|---|---|---|---|
| checkpoint-3801 | 3801 | 0.4890 | 1.686 |

Note: small-sample sanity subset (5 samples/language) run before the full passes; CE differs from the
full-sample eval_out_ce/eval_out_all figures for checkpoint-3801 (0.5071 / 0.5100) due to sampling noise
at n=5. Not used as an analysis input — retained only as a pipeline health check record.
Source: Kaggle notebook `notebook384d15794b` (scriptVersionId 336371597), `eval_sanity/eval_report.md`, retrieved 2026-07-19.
