# Perfusion and optional CVR processing

## What is present in DLBS ds004856

Metadata inspection of snapshot `1.3.0` found BIDS `perf/*_asl.nii.gz`, JSON,
and `aslcontext.tsv` files. Representative wave-1 files contain 60 alternating
`label`/`control` volumes and identify PCASL with labeling duration 1.65 s and
post-labeling delay 1.525 s. Crucially, the inspected JSON states
`M0Type: Absent`. The pipeline rechecks every session and does not assume these
values are universal.

Hypercapnia runs are named `task-Hypercapnia_*_bold.nii.gz`. Representative
metadata provide TR=2 s, TE=25 ms, and `TaskName`, but the subject tree contains
no hypercapnia events file, end-tidal CO2 trace, or other physiological trace.
Accordingly, the project does not derive a gas-normalized CVR map from native
DLBS files alone.

## ASL quantities

Control/label pairs are motion-corrected rigidly, differenced as

`DeltaM = M_control - M_label`,

screened for pair-level temporal outliers, and averaged. When no M0 image is
available, the output is explicitly relative perfusion:

`relative perfusion (%) = 100 * mean(DeltaM) / mean(M_control)`.

This is a unitless percentage of local control signal. It is not CBF in
mL/100 g/min and should not be compared as absolute flow across scanners or
protocols without additional validation.

When an actual M0 image and the required metadata are present, the implementation
supports the single-compartment PCASL consensus equation:

`CBF = 6000 * lambda * DeltaM * exp(PLD/T1b) /
       [2 * alpha * M0 * T1b * (1 - exp(-tau/T1b))]`.

Definitions:

- `CBF`: mL/100 g/min
- `lambda`: blood-brain partition coefficient, default 0.9 mL/g
- `alpha`: labeling efficiency, default 0.85 for PCASL
- `PLD`: post-labeling delay in seconds
- `T1b`: arterial-blood T1, default 1.65 s at 3 T
- `tau`: labeling duration in seconds
- `M0`: separately supplied equilibrium magnetization image
- `DeltaM`: mean control-label difference

Absolute CBF is refused or downgraded to relative perfusion when M0, labeling
type, PLD, or labeling duration are missing. A 3-D derived map is accepted as
quantitative only when its metadata explicitly declare recognized CBF units.

## Hypercapnia/CVR quantities

Without a regressor the CVR stage writes an `unsupported` provenance record and
no map. If a user supplies one value per BOLD volume, ordinary least squares
estimates the response slope and divides it by temporal mean signal:

`response = 100 * beta_regressor / mean(BOLD)`.

With a calibrated end-tidal CO2 regressor in mmHg the units are `% BOLD/mmHg`.
Otherwise the result is labelled `% BOLD/regressor unit`, not quantitative CVR.
Response delay is not estimated because native DLBS timing/trace information is
insufficient to distinguish vascular delay from an unspecified stimulus schedule.

## Registration, QC, and resources

Mean control and mean BOLD reference images are registered to the N4 T1 space
using rigid then affine SimpleITK registration. The same transform resamples the
derived physiology map. GM and WM summaries use structural masks in that space.
QC includes a representative ASL frame, map, T1 overlay, tissue distributions,
optional response map, registration provenance, nonfinite counts, pair rejection,
and range warnings.

Motion correction and registration process frames serially. Typical ASL peak RAM
is below 2 GB with float32 arrays; a 60-volume run may take 5-20 minutes on CPU.
Run one session at a time. BOLD motion correction can take longer because of the
larger time series.

## Commands

```powershell
vascular-edge process-perfusion --asl data/..._asl.nii.gz --metadata data/..._asl.json `
  --context data/..._aslcontext.tsv --t1 derivatives/..._desc-N4_T1w.nii.gz `
  --gm derivatives/..._label-GM_probseg.nii.gz --wm derivatives/..._label-WM_probseg.nii.gz `
  --output-dir derivatives/vascular_edge/sub-001/ses-wave1/perf `
  --subject sub-001 --session ses-wave1 --config configs/default.yaml

# Native DLBS hypercapnia data normally returns an informative unsupported record:
vascular-edge process-cvr --bold data/..._task-Hypercapnia_bold.nii.gz `
  --metadata data/..._task-Hypercapnia_bold.json --t1 derivatives/..._desc-N4_T1w.nii.gz `
  --output-dir derivatives/vascular_edge/sub-001/ses-wave1/func `
  --subject sub-001 --session ses-wave1
```

Do not manufacture a block regressor from scan duration or task name. Supply a
regressor only when its timing and provenance are independently documented.
