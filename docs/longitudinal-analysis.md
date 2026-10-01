# Longitudinal WMH conversion analysis

## Scientific question and dataset fit

This analysis asks whether baseline physiology differs between baseline
normal-appearing white matter (NAWM) that is WMH-labelled at follow-up and NAWM
that remains normal. Results are exploratory associations. They cannot show
that reduced perfusion, altered CVR, or any other exposure causes WMH growth.

A metadata-only audit of pinned OpenNeuro snapshot `ds004856` version `1.3.0`
found 464 `ses-wave1`, 309 `ses-wave2`, and 196 `ses-wave3` participant/session
directories. DLBS encodes FLAIR images as `acq-FLAIR_*_T2w`, which the discovery
layer now recognizes. There are 314 participants with at least two waves having
both MPRAGE T1w and acquisition-labelled FLAIR: 138 have two such waves and 176
have three. These are availability counts, not an analyzed cohort; QC, WMH
segmentation, physiology coverage, and consent/data completeness can reduce the
usable sample. The dataset describes approximately 4–5 years between waves,
but subject-specific timing/demographics should be read from provided metadata.

## Session ordering

DLBS labels are parsed numerically (`ses-wave1`, `ses-wave2`, `ses-wave3`). For
other BIDS layouts, the software can order a sessions table by `acq_time`,
`scan_date`, `days_since_baseline`, `years_since_baseline`, or `timepoint`.
Ambiguous labels fail rather than being sorted alphabetically. The default
research comparison is earliest-to-latest; adjacent-wave sensitivity analyses
can use consecutive visits.

## Registration and definitions

The follow-up T1 is registered to baseline T1 with a deterministic SimpleITK
rigid stage followed by affine refinement using Mattes mutual information and a
4/2/1 multiresolution pyramid. When grids have the same dimensions, a discrete
phase-correlation translation supplies a robust initialization; it is still
represented and applied as a SimpleITK physical-space transform. Candidate
selection combines foreground Dice and within-overlap intensity correlation.
Affine candidates are rejected when their determinant or singular values imply
implausible scaling. No nonlinear deformation is used. Follow-up FLAIR is resampled with
linear interpolation; WMH and coverage masks use nearest-neighbor interpolation.
The transform, registered images, foreground-overlap metric, and red/green
overlay are retained.

Both sessions must use the same WMH segmentation backend and configuration.
Pass both segmentation metrics/provenance JSON files to have the command enforce
matching backends and parameters; supplying only one or a mismatched pair fails.
The deterministic fallback is flagged as non-biological even when matched.
Given registered masks:

- **Persistent baseline lesion:** baseline WMH ∩ follow-up WMH.
- **Converting/newly affected tissue:** baseline WM, not baseline WMH, labelled
  WMH at follow-up, within common coverage, beyond the configurable baseline
  boundary-uncertainty margin, and part of a component meeting the minimum
  physical-volume threshold.
- **Stable NAWM:** eligible baseline WM outside baseline WMH and its uncertainty
  margin that remains non-WMH at follow-up.
- **Unsuitable:** CSF, missing physiology/registration coverage, non-WM,
  boundary-uncertain tissue, or other excluded voxels.

The default 1 mm uncertainty margin deliberately rejects apparent one-voxel
growth close to the baseline boundary. It improves specificity but can miss
real subtle progression. Report sensitivity analyses over plausible margins and
component thresholds.

## Outputs and inference unit

Outputs include all class masks, baseline distance-to-WMH, registered follow-up
images, transforms, a subject-level outcomes row, physiology summaries for
converting and stable tissue, and conversion fractions for 0–2, 2–4, 4–6,
6–10, and >10 mm baseline-distance bins. Subject rows include baseline and
follow-up WMH volume, absolute/percent change, confident newly affected volume,
centroid coordinates, map-specific means/medians, and available demographic
fields.

Voxel distributions are descriptive visualizations. Cohort models must use
subject-level summaries or an explicitly specified hierarchical model; millions
of spatially correlated voxels must not be treated as independent observations.

QC flags failed/low-overlap registration, low common coverage, >2-fold
resolution differences, configurable implausible lesion-volume jumps, missing
inputs/follow-up, and insufficient conversion voxels. Subjects are never
silently discarded.

## Commands

Run the structural and WMH pipeline identically for both waves, then:

```powershell
vascular-edge analyze-longitudinal `
  --baseline-t1 derivatives/vascular_edge/sub-1003/ses-wave1/anat/sub-1003_ses-wave1_desc-N4_T1w.nii.gz `
  --baseline-flair derivatives/vascular_edge/sub-1003/ses-wave1/anat/sub-1003_ses-wave1_space-T1w_desc-N4_FLAIR.nii.gz `
  --baseline-wmh derivatives/vascular_edge/sub-1003/ses-wave1/anat/sub-1003_ses-wave1_space-T1w_desc-wmh_mask.nii.gz `
  --baseline-wm derivatives/vascular_edge/sub-1003/ses-wave1/anat/sub-1003_ses-wave1_label-WM_probseg.nii.gz `
  --followup-t1 derivatives/vascular_edge/sub-1003/ses-wave2/anat/sub-1003_ses-wave2_desc-N4_T1w.nii.gz `
  --followup-flair derivatives/vascular_edge/sub-1003/ses-wave2/anat/sub-1003_ses-wave2_space-T1w_desc-N4_FLAIR.nii.gz `
  --followup-wmh derivatives/vascular_edge/sub-1003/ses-wave2/anat/sub-1003_ses-wave2_space-T1w_desc-wmh_mask.nii.gz `
  --subject sub-1003 --baseline-session ses-wave1 --followup-session ses-wave2 `
  --output-dir derivatives/vascular_edge/sub-1003/longitudinal --config configs/default.yaml

vascular-edge qc-longitudinal --provenance `
  derivatives/vascular_edge/sub-1003/longitudinal/sub-1003_from-wave2_to-wave1_desc-longitudinal.json
```

On the target 8 GB CPU-only machine, run one subject at a time. Registration is
linear and float32 where practical; runtime is typically driven by the two-stage
registration rather than feature extraction.
