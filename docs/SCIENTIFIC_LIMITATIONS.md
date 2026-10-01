# Scientific limitations

- This observational pipeline can estimate association, not causation, lesion mechanism, individual prognosis, or clinical utility.
- The deterministic demo deliberately injects one relationship and one null scenario. Its figures validate software behavior only and are not biological evidence.
- The built-in fallback WMH segmenter is transparent test/sensitivity machinery. It must not be treated as a validated diagnostic model. External models require acquisition-specific validation and independent licensing.
- Registration error can resemble lesion conversion. The workflow uses conservative linear registration, QC, and a baseline boundary-uncertainty exclusion, but these do not eliminate error.
- Perilesional distance is Euclidean in image space; it does not represent axonal paths, vascular territories, or causal spread.
- Partial volume, tissue-mask error, lesion confluence, limited field of view, and physiology coverage can change region summaries. Sparse regions are reported rather than silently extrapolated.
- ASL without adequate M0/calibration metadata is relative perfusion, not absolute CBF. BOLD response without end-tidal CO2 is not calibrated CVR in `%/mmHg`.
- Motion and physiological noise can bias ASL and BOLD estimates. QC flags do not guarantee usable physiology.
- Earliest-to-latest pairing discards intermediate trajectory information. More complex longitudinal models require adequate repeated sessions and prespecification.
- Complete-case analysis can be biased when missingness is informative. Every model reports its own sample size; this does not solve selection bias.
- Cognitive construct files can mix raw, adjusted, percentile, accuracy, and latency measures. Automatic composites are therefore disabled unless components and directions are explicitly justified.
- Multiple-comparison adjustment controls only the declared family and does not rescue post hoc hypothesis selection. Exploratory findings require independent replication.
- Mixed models can be unstable in small or unbalanced cohorts. The R component reports unsupported models as `not_estimable` rather than forcing convergence.
- DLBS/OpenNeuro availability may not include calibrated physiology, explicit gas traces, usable follow-up for every modality, or all desired vascular covariates. Missing modalities are not negative findings.
