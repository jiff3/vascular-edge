"""Conservative hypercapnia BOLD analysis with explicit regressor requirements."""

from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import SimpleITK as sitk

from .perfusion import motion_correct_asl
from .preprocessing import register_flair_to_t1


def cvr_slope(bold: np.ndarray, regressor: np.ndarray) -> np.ndarray:
    """OLS signal slope per regressor unit, with time on the last BOLD axis."""
    data = np.asarray(bold, np.float32)
    x = np.asarray(regressor, np.float32)
    if data.shape[-1] != x.size:
        raise ValueError("Regressor length must equal number of BOLD volumes.")
    x = x - x.mean()
    denominator = float(x @ x)
    if denominator == 0:
        raise ValueError("CVR regressor has zero variance.")
    centered = data - data.mean(axis=-1, keepdims=True)
    return np.tensordot(centered, x, axes=(-1, 0)) / denominator


def percent_bold_response(bold: np.ndarray, regressor: np.ndarray) -> np.ndarray:
    slope = cvr_slope(bold, regressor)
    mean = np.nanmean(bold, axis=-1)
    return np.divide(100 * slope, mean, out=np.full_like(slope, np.nan, np.float32), where=mean > 0)


def read_regressor(path: str | Path, column: str = "stimulus") -> np.ndarray:
    path = Path(path)
    table = pd.read_csv(path, sep="\t" if path.suffix.lower() == ".tsv" else ",")
    if column not in table:
        raise ValueError(f"Regressor file must contain column '{column}'.")
    values = table[column].to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Regressor contains NaN/Inf values.")
    return values


def _resample_map(
    map_file: Path, t1_file: str | Path, transform_file: Path, output_file: Path
) -> Path:
    moving = sitk.ReadImage(str(map_file))
    fixed = sitk.ReadImage(str(t1_file))
    transform = sitk.ReadTransform(str(transform_file))
    output_file.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(
        sitk.Resample(moving, fixed, transform, sitk.sitkLinear, 0.0, sitk.sitkFloat32),
        str(output_file),
    )
    return output_file


def process_cvr(
    bold_file: str | Path,
    metadata_file: str | Path,
    t1_file: str | Path,
    output_dir: str | Path,
    subject: str,
    session: str,
    regressor_file: str | Path | None = None,
    regressor_column: str = "stimulus",
    trace_units: str | None = None,
    motion_correct: bool = True,
    force: bool = False,
) -> dict:
    """Estimate percent BOLD per regressor unit; never infer missing gas timing/traces."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    prefix = f"{subject}_{session}"
    provenance = root / f"{prefix}_desc-CVR_provenance.json"
    if provenance.exists() and not force:
        record = json.loads(provenance.read_text(encoding="utf-8"))
        record["resumed"] = True
        return record
    metadata = json.loads(Path(metadata_file).read_text(encoding="utf-8"))
    warnings: list[str] = []
    if regressor_file is None:
        record = {
            "subject": subject,
            "session": session,
            "status": "unsupported",
            "quantity": "none",
            "units": "none",
            "warnings": [
                "Hypercapnia BOLD has no supplied event/gas regressor; CVR cannot be estimated without inventing timing"
            ],
            "metadata": metadata,
            "resumed": False,
        }
        provenance.write_text(json.dumps(record, indent=2), encoding="utf-8")
        return record
    image = nib.load(str(bold_file))
    bold = image.get_fdata(dtype=np.float32)
    if bold.ndim != 4:
        raise ValueError(f"Hypercapnia BOLD must be 4D, got {bold.shape}.")
    regressor = read_regressor(regressor_file, regressor_column)
    if regressor.size != bold.shape[3]:
        raise ValueError("Regressor length does not match BOLD volumes.")
    motion = []
    if motion_correct:
        bold, motion = motion_correct_asl(bold, image.affine)
    mean_bold = np.nanmean(bold, axis=3).astype(np.float32)
    positive = mean_bold[np.isfinite(mean_bold) & (mean_bold > 0)]
    foreground_threshold = 0.1 * float(np.percentile(positive, 99)) if positive.size else np.inf
    foreground = mean_bold > foreground_threshold
    response = percent_bold_response(bold, regressor)
    response[~foreground] = 0
    if trace_units and trace_units.lower() in {"mmhg", "mm hg"}:
        quantity, units = "cvr_magnitude", "% BOLD/mmHg"
    else:
        quantity, units = "hypercapnia_percent_bold_response", "% BOLD/regressor unit"
        warnings.append(
            "No calibrated end-tidal CO2 units supplied; output is not quantitative CVR in %/mmHg"
        )
    nonfinite = int((~np.isfinite(response) & foreground).sum())
    response = np.where(np.isfinite(response), response, 0).astype(np.float32)
    if nonfinite:
        warnings.append(f"replaced {nonfinite} nonfinite response voxels with zero")
    native = root / f"{prefix}_space-BOLD_desc-{quantity}_map.nii.gz"
    header = image.header.copy()
    header.set_data_dtype(np.float32)
    nib.save(nib.Nifti1Image(response, image.affine, header), native)
    reference = root / f"{prefix}_space-BOLD_desc-mean_bold.nii.gz"
    nib.save(nib.Nifti1Image(mean_bold, image.affine, header), reference)
    registered_reference = root / f"{prefix}_space-T1w_desc-mean_bold.nii.gz"
    transform = root / f"{prefix}_from-BOLD_to-T1w_xfm.tfm"
    registration = register_flair_to_t1(reference, t1_file, registered_reference, transform)
    t1_map = _resample_map(
        native, t1_file, transform, root / f"{prefix}_space-T1w_desc-{quantity}_map.nii.gz"
    )
    values = nib.load(str(t1_map)).get_fdata(dtype=np.float32)
    finite = values[np.isfinite(values) & (values != 0)]
    if finite.size and np.percentile(np.abs(finite), 99) > 20:
        warnings.append("extreme percent-BOLD response values exceed 20%")
    record = {
        "subject": subject,
        "session": session,
        "status": "warn" if warnings else "pass",
        "quantity": quantity,
        "units": units,
        "warnings": warnings,
        "inputs": {
            "bold": str(bold_file),
            "metadata": str(metadata_file),
            "regressor": str(regressor_file),
        },
        "outputs": {
            "native_map": str(native),
            "t1_map": str(t1_map),
            "mean_bold": str(reference),
            "registered_mean_bold": str(registered_reference),
            "transform": str(transform),
        },
        "registration": registration,
        "motion": motion,
        "trace_units": trace_units,
        "parameters": {
            "regressor_column": regressor_column,
            "motion_correction": motion_correct,
            "repetition_time_s": metadata.get("RepetitionTime"),
        },
        "foreground_threshold": foreground_threshold,
        "resumed": False,
    }
    provenance.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record
