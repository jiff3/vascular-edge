# Synthetic demonstration

## Run

From an installed editable checkout:

```bash
vascular-edge run --config configs/analysis.yaml --demo
```

The command creates `results/demo/` and requires no download, private data, GPU, R, or external segmentation model. On a typical CPU it should finish in roughly a minute; timing depends on registration libraries and hardware.

## What is synthetic

The demo contains one two-session 3D structural/ASL/hypercapnia-BOLD phantom for exercising image processing and QC, plus a deterministic 48-subject tabular cohort. The cohort injects an obvious physiology–progression relationship so recovery can be tested and separately includes a null variable so the reporter can be checked against unsupported claims. These are software fixtures, not simulated population evidence.

## Expected products

- `pipeline_status.json` with every configured stage accounted for;
- `derivatives/` with structural, WMH, physiology, perilesional, and longitudinal files;
- `cohort/analysis_table.csv` and run-specific data dictionary;
- `statistics/` with missingness, distributions, correlations, bootstrap intervals, profiles, and conversion summaries;
- `figures/01_...png` through `08_...png` plus captions;
- `report/research_report.md` with explicit synthetic labeling;
- `artifacts.json` linking the products.

## Review checklist

1. Confirm the report banner says **Synthetic demonstration; not empirical evidence**.
2. Inspect segmentation overlays rather than accepting a numeric QC status alone.
3. Confirm perilesional contours are nested in physical-distance order.
4. Confirm the longitudinal panel distinguishes baseline, follow-up, converting tissue, and the uncertainty exclusion.
5. Inspect statistical tables for both the injected association and retained null results.
6. Run the command again and verify resumable subject-level steps report `resumed`.

## R models

The Python demo completes without R. To run the prespecified R component after installing its packages:

```bash
Rscript r/run_models.R results/demo/cohort/analysis_table.csv results/demo/cohort/perilesional_profiles.csv results/demo/statistics 2026
vascular-edge build-results --statistics-dir results/demo/statistics --output results/demo/statistics/statistical_results.md
```

Regenerate the top-level report afterward by rerunning the demo command. A missing or non-estimable R model remains explicitly labelled and is never converted into a finding.
