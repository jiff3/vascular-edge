# Perilesional vascular gradients

## Scope and hypothesis

This module asks whether physiology differs with distance from a visible WMH. A
**perilesional vascular gradient** is a descriptive spatial profile, not a
validated clinical biomarker. Its presence does not establish causality,
diagnosis, prognosis, or a particular biological mechanism.

```text
WMH core | 0–2 mm | 2–4 mm | 4–6 mm | 6–10 mm | remote NAWM >10 mm
████████ | ▓▓▓▓▓▓ | ▒▒▒▒▒▒ | ░░░░░░ | ······· |       white matter
         └──────── increasing physical distance from nearest WMH ────────>
```

The analysis uses the final postprocessed WMH mask in the shared T1/FLAIR
space. It computes the Euclidean distance from each voxel center to the nearest
WMH voxel center with `scipy.ndimage.distance_transform_edt`, passing the three
voxel spacings as physical sampling. This is therefore a center-to-center
distance, in millimeters, and correctly handles anisotropic sampling.

## Regions and masks

The core has distance zero. Outside intervals are `(lower, upper]`: `(0,2]`,
`(2,4]`, `(4,6]`, and `(6,10]` mm. Remote normal-appearing white matter (NAWM)
is `>10` mm. Outer regions are restricted to the WM mask and exclude WMH, CSF,
and ventricles. They are also intersected with valid physiology coverage. By
default, finite nonzero voxels common to all supplied maps define coverage;
use an explicit `--coverage-mask` when zero is a meaningful physiological
value or map-specific acquisition coverage is known.

Disconnected WMHs are labeled with 18-connectivity. Components smaller than
`min_lesion_volume_mm3` are recorded in QC and omitted from ring construction.
Global rings are a single partition, so they cannot overlap. For lesion-level
rows, each voxel is assigned to the closest retained component. Exact ties use
SciPy's deterministic array-order resolution. Thus nearby-lesion neighborhoods
are partitioned and never counted twice; this ownership choice must be retained
when interpreting lesion-level results.

## Outputs

- Physical distance, region-label, and retained-component NIfTI images.
- Long/tidy CSV, with optional Parquet (`pip install -e ".[parquet]"`).
- Subject and lesion rows with voxel count, physical volume, mean, median,
  sample standard deviation, IQR, and 5th/25th/75th/95th percentiles.
- Component volume and centroid in voxel and world coordinates.
- JSON QC with retained/excluded lesions, image-boundary contact, mask geometry,
  physiology coverage, CSF overlap, and low/empty region flags.
- A subject figure using identical contours on FLAIR, perfusion, and CVR, plus
  an unfitted physiology-versus-distance profile.
- Cohort figures showing subject profiles, distributions, and WMH-burden
  scatterplots. No favorable trend or direction is hard-coded.

## Commands

```powershell
vascular-edge analyze-perilesional `
  --wmh derivatives/vascular_edge/sub-001/ses-1/anat/sub-001_ses-1_desc-wmh_mask.nii.gz `
  --wm derivatives/vascular_edge/sub-001/ses-1/anat/sub-001_ses-1_label-WM_mask.nii.gz `
  --csf derivatives/vascular_edge/sub-001/ses-1/anat/sub-001_ses-1_label-CSF_mask.nii.gz `
  --flair derivatives/vascular_edge/sub-001/ses-1/anat/sub-001_ses-1_space-T1w_FLAIR.nii.gz `
  --perfusion derivatives/vascular_edge/sub-001/ses-1/perf/sub-001_ses-1_desc-relativeperfusion_map.nii.gz `
  --perfusion-units relative --subject sub-001 --session ses-1 `
  --output-dir derivatives/vascular_edge/sub-001/ses-1/perilesional --config configs/default.yaml

vascular-edge plot-perilesional `
  --table results/perilesional_cohort.csv --output-dir results/figures/perilesional
```

All inputs must already share the structural grid. A mismatched array shape or
physiology affine fails loudly instead of being silently resampled. Run one
subject/session at a time on the target hardware; the largest transient arrays
are the distance transform and nearest-index arrays.

## Limitations

Distances are influenced by registration, WMH segmentation, partial volume,
WM segmentation, lesion topology, and voxel size. Rings around adjacent lesions
cannot be attributed independently in tissue equidistant from both. Shared-map
coverage improves comparability but can shrink regions. Spatial autocorrelation
means voxels are not independent samples; cohort inference should use
subject-level summaries or an explicitly specified multilevel model. Empty and
small rings are retained and flagged rather than silently excluded.
