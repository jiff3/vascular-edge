# Methods

## Objective and analysis units

Vascular Edge estimates whether MRI physiology varies with distance from visible white-matter hyperintensity (WMH), and whether baseline physiology differs in white matter that is labelled WMH at follow-up. It estimates associations, not causal or clinical effects. Subject-session images, repeated distance-region measurements, and one-row-per-longitudinal-pair outcomes remain separate to avoid voxel-level pseudoreplication.

## Cohort discovery and validation

The workflow inventories T1w, FLAIR (including DLBS `acq-FLAIR_T2w` naming), ASL, and BOLD files without requiring every modality. A subject-session enters structural processing only when T1w and FLAIR exist. The deterministic cohort cap and seed come from the YAML configuration. Input files are checked for existence, readable NIfTI headers, finite affine matrices, positive voxel sizes, and expected dimensionality. The manifest, validation table, inclusion state, and exclusions are saved.

## Structural processing and WMH labels

Images are converted to canonical orientation. T1w data undergo N4 bias correction; FLAIR is registered to T1 space with SimpleITK. Transparent intensity/morphology-based masks provide a CPU-only fallback for tests and sensitivity analyses. They are not validated clinical segmentations. Production analyses can select the TrUE-Net backend with separately installed versioned software and weights. Component filtering is performed in physical cubic millimeters. QC montages show structural registration, tissue masks, and WMH overlays, and provenance records volumes and warnings.

## Perfusion and CVR

ASL processing reads sidecar/context metadata, motion-corrects when configured, constructs control-label differences, and emits absolute CBF only when calibration metadata are adequate. Otherwise output is explicitly relative perfusion in percent mean control signal. Hypercapnia BOLD requires an explicit regressor or events record; it is not inferred from a generic BOLD filename. Without end-tidal CO2 in mmHg, the result is percent-BOLD response per regressor unit rather than calibrated CVR. Physiology maps are registered to structural space and receive coverage and tissue QC.

## Perilesional analysis

Euclidean distance is computed from WMH in physical millimeters using NIfTI voxel spacing. Regions are WMH core, 0–2, 2–4, 4–6, 6–10 mm, and remote normal-appearing white matter (>10 mm). They are restricted to white matter, avoid CSF, and require valid physiology coverage. Nearest-lesion assignment prevents overlapping lesion neighborhoods from duplicating voxels. Tables contain counts, means, dispersion, units, subject, session, lesion identity, map, and distance region.

## Longitudinal analysis

For each subject with at least two structural sessions, the earliest and latest sessions define the pair. Follow-up structural and lesion images are registered to baseline. Baseline WMH, follow-up WMH, new/converting tissue, stable normal-appearing white matter, and an excluded boundary-uncertainty band are retained. Baseline physiology is summarized in future converting and stable tissue only when minimum voxel counts are met. Saved outputs include registration checks, volumes, percent change, tissue counts, and overlays.

## Metadata and cognition harmonization

`build-analysis-table` scans TSV, CSV, XLS, and XLSX participant/behavioral files. Identifier and session fields are detected from values and common aliases; wide `_W1`/`_W2` variables are transformed to subject-session rows. Dataset-native sex/gender, race, ethnicity, education, handedness, vascular variables, and cognitive scores are preserved when available. Missing strings are never converted to zero. MRI age is preferred for an MRI session while source columns remain documented.

No cognitive variables are automatically averaged. A standardized domain composite is created only when configuration names at least two components and their directions. Each component is z-standardized within the analysis sample, signed to a documented common direction, and averaged with complete components. Original variables and a generated data dictionary are retained.

## Python descriptive and exploratory statistics

Outputs include sample size/missingness, distributions (5th, 25th, 50th, 75th, and 95th percentiles), Pearson and Spearman correlations, age and baseline-WMH relationships, distance profiles, and conversion summaries. Pearson 95% confidence intervals use subject-row percentile bootstrap sampling with the configured seed. Models use complete cases and record their actual sample size. Exploratory families receive separate Benjamini–Hochberg correction; raw p values and q values remain available.

## Prespecified R models

The readable scripts in `r/` implement:

1. `physiology ~ distance_region + age + sex + baseline_WMH_burden + (1 | subject)`
2. `WMH_change ~ baseline_physiology + age + baseline_WMH_burden + available covariates`
3. `cognitive_score ~ vascular_or_perfusion + age + sex + education_or_available_covariates`

A random-intercept model is attempted only with at least 10 subjects, at least two observations per subject on average, and at least two regions. Missing packages, insufficient complete cases, rank deficiency, singularity, or failed convergence produce `not_estimable`; the software does not force a coefficient. Cognitive analyses are exploratory unless an external protocol prespecifies the exact outcome/exposure. Estimates, standard errors, confidence intervals, exact p values, sample sizes, and within-family BH q values are written by `broom`/`broom.mixed`.

## Reporting policy

Prespecified and exploratory analyses are labeled separately. Prose is assembled from saved estimates, intervals, q values, and model status. Unavailable, null, failed, and non-estimable results remain visible. The report never converts a missing result into evidence of no association and never claims a decrease or increase without a supporting estimate.

## External model acknowledgement

TrUE-Net is optional and not included. Its repository documents Apache-2.0 licensing and the citation: Sundaresan et al., *Medical Image Analysis* 73 (2021), 102184, DOI `10.1016/j.media.2021.102184`. The fallback backend does not reproduce or contain TrUE-Net code or weights.
