# Structural MRI and WMH pipeline

This workflow processes one subject/session at a time and writes a
BIDS-Derivatives-inspired tree under `derivatives/vascular_edge/sub-*/ses-*/`.
Existing complete outputs are reused unless `--force` is passed.

## Algorithms

1. Nibabel validates 3-D shape, affine invertibility, voxel sizes, finite values,
   orientation, and nonzero coverage. `as_closest_canonical` reorders voxel data
   to RAS+ while updating the affine to preserve physical coordinates.
2. Brain extraction uses an Otsu threshold, closing/filling, and largest connected
   component. It is deliberately lightweight and must be visually checked.
3. SimpleITK N4 estimates the T1 and aligned-FLAIR bias fields on twofold-shrunk
   images by default, then evaluates each field at full resolution. Brain-masked
   copies are also written for backends that require skull-stripped inputs.
4. FLAIR is registered to N4 T1 with deterministic Mattes mutual information:
   rigid initialization followed by affine optimization. Affine, rigid, and
   initialized transforms are compared by foreground overlap to catch optimizer
   drift. No nonlinear registration is used.
5. Tissue masks use three-class multi-Otsu T1 intensity segmentation within the
   brain mask. Classes are ordered as CSF, GM, and WM by T1 intensity. This is a
   replaceable, nonclinical backend and is less robust than FAST/SPM on difficult scans.
6. The default WMH fallback uses robust FLAIR z scores only for deterministic
   synthetic tests and sensitivity checks. It is **not biologically validated**.
   Components below the configured physical volume are removed and the result is
   constrained to one-voxel-dilated WM territory. Raw and postprocessed masks are saved.

## External WMH backend

The optional TrUE-Net adapter invokes an external executable with `evaluate` and
`--use_cpu`; no opaque network code or weights are copied into this package. The
tested source is pinned in configuration to upstream commit
`4c539165932f7f0607693363731fe5d4e8356590`; record the exact FSL conda package
build and model (`mwsc` or `ukbb`) in the environment lock used for real analysis.
TrUE-Net and its pretrained models are not installed automatically. Set
`segmentation.backend: truenet`, `model_dir`, and `executable`. Missing programs
or weights produce an actionable error rather than falling back silently.

## Commands

```powershell
vascular-edge preprocess-structural --t1 data/..._T1w.nii.gz --flair data/..._FLAIR.nii.gz `
  --derivatives-root derivatives/vascular_edge --subject sub-001 --session ses-wave1

vascular-edge segment-wmh --t1 derivatives/..._desc-N4_T1w.nii.gz `
  --flair derivatives/..._space-T1w_desc-N4_FLAIR.nii.gz `
  --brain-mask derivatives/..._desc-brain_mask.nii.gz --wm-mask derivatives/..._label-WM_probseg.nii.gz `
  --output-dir derivatives/vascular_edge/sub-001/ses-wave1/anat --subject sub-001 --session ses-wave1 `
  --config configs/default.yaml

vascular-edge qc-structural --t1 derivatives/..._desc-N4_T1w.nii.gz `
  --flair derivatives/..._space-T1w_desc-N4_FLAIR.nii.gz --csf derivatives/...CSF...nii.gz `
  --gm derivatives/...GM...nii.gz --wm derivatives/...WM...nii.gz --wmh derivatives/...WMH_mask.nii.gz `
  --output-dir derivatives/vascular_edge/sub-001/ses-wave1/qc --subject sub-001 --session ses-wave1
```

## Resource expectations and limitations

For a typical 1-mm 256×256×176 scan, float32 data occupy about 46 MB per volume.
Registration and N4 hold several working images; expect roughly 1–3 GB peak RAM
and 5–20 minutes per session on a modern CPU. Actual time depends strongly on
matrix size and disk speed. Only one session should run at once on an 8 GB host.

All lightweight masks require visual QC. Multi-Otsu does not model tissue priors;
atrophy, strong bias, pathology, motion, or incomplete coverage can invalidate it.
Registration foreground overlap detects gross drift but is not proof of anatomical
alignment. TrUE-Net predictions require dataset-specific validation and must not
be treated as clinical findings.
