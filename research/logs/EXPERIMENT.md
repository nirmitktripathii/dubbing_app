# research_logs — Canonical Observability & Research Archive (Indic Dubbing Pipeline V3)

Canonical location for ALL training/evaluation telemetry, session logs, incident
records, analysis documents, and (at milestones) model checkpoints, for every pipeline
stage (notebooks 01–07). Purpose: a citable, reproducible record feeding the final
research paper/report.

Layout:
- `ANALYSIS_<stage>.md` — living in-depth analysis per stage (currently: ANALYSIS_02_llm_finetune.md)
- `sessions/` — one factual log per daily training session
- `metrics/` — CSV time series (schema documented in ANALYSIS_02 §8)
- `checkpoints/` — stage-final/best adapters pulled from Kaggle (Kaggle version outputs
  remain the daily checkpoint archive of record)
- `analysis/` — cross-session statistical analyses and plots for the paper

External systems of record, cross-referenced throughout:
- W&B project: wandb.ai/nktthegreat-soccernet/indic-dubbing-v3
- Kaggle notebook versions: kaggle.com/code/nktthegreat/02-llm-finetune (and 01–07 siblings)

Experiment identity, research angles, findings register: see ANALYSIS_02_llm_finetune.md.
(Supersedes the earlier `E:\Dubbing app\research_logs\` location, migrated 2026-07-17.)
