# Reproducibility

## One-command execution

```bash
vascular-edge run --config configs/analysis.yaml --demo
vascular-edge run --config configs/analysis.yaml
```

The first command is self-contained and network-free. The second uses only files already present under `paths.bids_root`. Paths in YAML are resolved relative to the repository, so no machine-specific location is required.

## Recorded state

Each run root contains:

- `pipeline_status.json`: status and reason for every stage;
- `pipeline.log`: timestamped execution summary;
- `cohort_manifest.json` and `input_validation.csv`: included inputs and validation;
- `artifacts.json`: resolved products, seed, and configuration location;
- derivative JSON: parameters, inputs, warnings, units, and software versions;
- `cohort/analysis_table.csv`, data dictionary, missingness, and inclusion records;
- `statistics/`: estimates, intervals, p/q values, model sample sizes, and random seed;
- `figures/captions.json` and `report/research_report.md`.

Subject-level processing functions detect complete compatible outputs and return `resumed`; `--force` recomputes them. Optional unavailable modalities are `skipped` with reasons. Required failures stop the command and remain in status.

## Determinism

The default analysis seed is `2026`. Cohort ordering, synthetic inputs, bootstrap sampling, and injected/null fixtures use the recorded seed. CPU registration can still vary slightly across library/platform versions, which is why the exact tested Python packages are pinned in `environment/requirements-lock.txt`.

## Validation commands

```bash
python -m pytest
ruff check .
ruff format --check .
vascular-edge --help
vascular-edge run --config configs/analysis.yaml --demo
Rscript -e "parse(file='r/functions.R'); parse(file='r/run_models.R')"
Rscript r/run_models.R results/demo/cohort/analysis_table.csv results/demo/cohort/perilesional_profiles.csv results/demo/statistics 2026
```

R is optional in the Python environment. When it or required packages are absent, Python reporting explicitly says the prespecified R models have not run. Installing R is not silently attempted.

## Clean-room check

Delete only the configured demo output directory, rerun the demo command, inspect all eight PNGs and the report, then run the command once more to verify subject-level stages report `resumed`. Do not delete or overwrite a real run to perform this check.

## Version control boundaries

Commit source, configuration, tests, documentation, the lock file, and selected synthetic figures in `docs/assets/demo/`. Do not commit participant NIfTI/tabular data, real derivatives, raw OpenNeuro/Datalad metadata, model weights without redistribution rights, logs containing private paths, access tokens, credentials, or generated `results/`.
