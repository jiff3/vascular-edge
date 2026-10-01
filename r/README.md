# R cohort analysis

Run from the repository root:

```text
Rscript r/run_models.R results/analysis/cohort/analysis_table.csv results/analysis/cohort/perilesional_profiles.csv results/analysis/statistics 2026
```

Required packages are `ggplot2` and `broom`; repeated-measures models additionally require `lme4`, `lmerTest`, and `broom.mixed`. Missing packages or unsupported sample structure are recorded as `not_estimable` in `model_status.csv`.

On a machine with R installed, install and validate with:

```text
Rscript -e "install.packages(c('ggplot2','broom','lme4','lmerTest','broom.mixed'), repos='https://cloud.r-project.org')"
Rscript -e "parse(file='r/functions.R'); parse(file='r/run_models.R')"
Rscript r/run_models.R results/analysis/cohort/analysis_table.csv results/analysis/cohort/perilesional_profiles.csv results/analysis/statistics 2026
```

Outputs are `model_estimates.csv`, `model_status.csv`, `model_estimates.png` when models are estimable, `r_analysis_config.txt`, and `r_session_info.txt`.
