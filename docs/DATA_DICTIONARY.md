# Analysis data dictionary

The authoritative run-specific dictionary is `cohort/analysis_data_dictionary.csv`; it records source file, sheet, original column, harmonized name, role, and transformation. Columns appear only when data genuinely exist.

## Identifiers and time

| Field/pattern | Meaning |
|---|---|
| `subject` | BIDS participant identifier, normalized without inventing identities |
| `session` | BIDS session or harmonized wave/timepoint |
| `baseline_session`, `followup_session` | Sessions defining a longitudinal pair |
| `interval_*` | Dataset-provided elapsed time; original unit retained/documented |

## Demographics and vascular covariates

| Field/pattern | Meaning |
|---|---|
| `age` / source age columns | MRI-session age where identifiable; original fields retained |
| `sex` or `gender` | Dataset-native coding; no recoding beyond normalized missingness |
| `education*`, `race*`, `ethnicity*`, `handedness*` | Available source variables, preserved |
| `bmi*`, blood-pressure/vascular fields | Available source covariates, preserved with source names |

## Imaging features

| Field/pattern | Meaning/unit |
|---|---|
| `wmh_volume_ml` | Session WMH volume, mL |
| `baseline_wmh_volume_ml`, `baseline_wmh_ml` | Baseline WMH burden, mL |
| `followup_wmh_ml` | Follow-up WMH burden, mL |
| `lesion_progression_ml`, `wmh_change_ml` | Follow-up minus baseline WMH, mL |
| `wmh_change_percent` | WMH change relative to baseline, percent |
| `new_wmh_volume_ml` | Newly labelled follow-up WMH, mL |
| `*_converting_mean` | Baseline map mean in tissue later labelled WMH |
| `*_stable_nawm_mean` | Baseline map mean in stable normal-appearing white matter |
| `map__region__mean` | Map mean in a physical-distance region |
| `perfusion*` | Absolute CBF only when calibrated; otherwise explicitly relative units |
| `cvr*` | `%/mmHg` only with CO2 calibration; otherwise percent-BOLD/regressor unit |

## Perilesional long table

One row represents subject × session × map × distance region (and lesion when lesion-level output is requested). Core columns include `subject`, `session`, `map`, `units`, `distance_region`, `distance_inner_mm`, `distance_outer_mm`, `n_voxels`, `mean`, `sd`, `median`, and QC/coverage fields.

## Cognition

Harmonized source variables are prefixed `cog_` and encode source workbook/domain/measure. Their scale and direction remain source-specific. Configured composites are named from the configuration and have mean approximately zero/SD approximately one in the contributing sample. A composite is absent unless at least two documented components and directions were provided.

## Missingness

Blank strings, `NA`, `n/a`, `xx`, and `.` are missing—not zero. Derived values are missing when inputs, valid coverage, or minimum voxel/sample requirements are absent. Model tables state complete-case `n`; `not_estimable` is a model status, not a numeric result.
