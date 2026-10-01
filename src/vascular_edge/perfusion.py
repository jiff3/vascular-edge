"""Metadata-gated ASL differencing, quantification, registration, and summaries."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
import SimpleITK as sitk

from .preprocessing import register_flair_to_t1


@dataclass
class ASLAssessment:
    organization: str
    quantitative_cbf_supported: bool
    output_quantity: str
    units: str
    warnings: list[str]


def read_asl_context(path: str | Path) -> list[str]:
    frame = pd.read_csv(path, sep="\t")
    if "volume_type" not in frame:
        raise ValueError("ASL context must contain a volume_type column.")
    return frame["volume_type"].astype(str).str.lower().tolist()


def assess_asl(
    shape: tuple[int, ...], metadata: dict[str, Any], context: list[str] | None
) -> ASLAssessment:
    warnings: list[str] = []
    if len(shape) == 3:
        units = str(metadata.get("Units") or metadata.get("CBFUnits") or "arbitrary units")
        quantitative = units.lower().replace(" ", "") in {"ml/100g/min", "ml/100g/minute"}
        if not quantitative:
            warnings.append("3D perfusion map lacks recognized quantitative CBF units")
        return ASLAssessment(
            "derived_perfusion_map",
            quantitative,
            "cerebral_blood_flow" if quantitative else "relative_perfusion",
            units,
            warnings,
        )
    if len(shape) != 4:
        return ASLAssessment(
            "unsupported", False, "none", "none", [f"ASL must be 3D or 4D, got {shape}"]
        )
    if context is None or len(context) != shape[3]:
        return ASLAssessment(
            "unsupported",
            False,
            "none",
            "none",
            ["ASL context is missing or does not match volume count"],
        )
    if not {"control", "label"}.issubset(set(context)):
        return ASLAssessment(
            "unsupported", False, "none", "none", ["ASL context lacks control/label volumes"]
        )
    required = ("ArterialSpinLabelingType", "PostLabelingDelay", "LabelingDuration")
    quantitative = (
        all(metadata.get(key) is not None for key in required)
        and str(metadata.get("M0Type", "")).lower() != "absent"
    )
    if str(metadata.get("M0Type", "")).lower() == "absent":
        warnings.append("M0Type is Absent; absolute CBF is not defensible")
    elif not quantitative:
        warnings.append("ASL kinetic/calibration metadata are incomplete")
    return ASLAssessment(
        "control_label",
        quantitative,
        "cerebral_blood_flow" if quantitative else "relative_perfusion_percent_control",
        "mL/100g/min" if quantitative else "% mean control signal",
        warnings,
    )


def pair_control_label(
    data: np.ndarray, context: list[str]
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Pair adjacent control/label volumes, preserving acquisition order."""
    differences, controls, skipped = [], [], []
    index = 0
    while index < len(context) - 1:
        pair = context[index : index + 2]
        if set(pair) == {"control", "label"}:
            control_i = index + pair.index("control")
            label_i = index + pair.index("label")
            controls.append(data[..., control_i])
            differences.append(data[..., control_i] - data[..., label_i])
            index += 2
        else:
            skipped.append(index)
            index += 1
    if not differences:
        raise ValueError("No adjacent control/label pairs were found.")
    return (
        np.stack(differences, axis=-1).astype(np.float32),
        np.stack(controls, axis=-1).astype(np.float32),
        skipped,
    )


def reject_pair_outliers(
    differences: np.ndarray, z_threshold: float = 3.5
) -> tuple[np.ndarray, list[int]]:
    scores = np.nanmean(np.abs(differences.reshape(-1, differences.shape[-1])), axis=0)
    median = np.nanmedian(scores)
    mad = np.nanmedian(np.abs(scores - median))
    if mad <= np.finfo(np.float32).eps:
        return np.ones(scores.size, bool), []
    robust_z = np.abs(scores - median) / (1.4826 * mad)
    keep = robust_z <= z_threshold
    return keep, np.flatnonzero(~keep).astype(int).tolist()


def quantify_pcasl(
    delta_m: np.ndarray,
    m0: np.ndarray,
    labeling_duration_s: float,
    pld_s: float,
    labeling_efficiency: float = 0.85,
    blood_t1_s: float = 1.65,
    blood_brain_partition: float = 0.9,
) -> np.ndarray:
    """Single-compartment PCASL CBF in mL/100g/min (Alsop consensus model)."""
    denominator = (
        2 * labeling_efficiency * blood_t1_s * m0 * (1 - np.exp(-labeling_duration_s / blood_t1_s))
    )
    numerator = 6000 * blood_brain_partition * delta_m * np.exp(pld_s / blood_t1_s)
    return np.divide(
        numerator, denominator, out=np.zeros_like(delta_m, np.float32), where=denominator > 0
    )


def relative_perfusion(delta_m: np.ndarray, mean_control: np.ndarray) -> np.ndarray:
    return np.divide(
        100 * delta_m, mean_control, out=np.zeros_like(delta_m, np.float32), where=mean_control > 0
    )


def motion_correct_asl(data: np.ndarray, affine: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    """Rigidly align each volume to the temporal mean, one frame at a time."""
    reference_array = np.nanmean(data, axis=3).astype(np.float32)
    reference = sitk.GetImageFromArray(np.transpose(reference_array, (2, 1, 0)))
    zooms = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    reference.SetSpacing(tuple(float(v) for v in zooms))
    corrected = np.empty_like(data, dtype=np.float32)
    records: list[dict] = []
    for i in range(data.shape[3]):
        moving = sitk.GetImageFromArray(np.transpose(data[..., i].astype(np.float32), (2, 1, 0)))
        moving.CopyInformation(reference)
        transform = sitk.CenteredTransformInitializer(
            reference,
            moving,
            sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        )
        reg = sitk.ImageRegistrationMethod()
        reg.SetMetricAsMeanSquares()
        reg.SetMetricSamplingStrategy(reg.RANDOM)
        reg.SetMetricSamplingPercentage(0.1, seed=2026 + i)
        reg.SetInterpolator(sitk.sitkLinear)
        reg.SetOptimizerAsRegularStepGradientDescent(1.0, 0.01, 30)
        reg.SetOptimizerScalesFromPhysicalShift()
        reg.SetShrinkFactorsPerLevel([2, 1])
        reg.SetSmoothingSigmasPerLevel([1, 0])
        reg.SetInitialTransform(transform)
        final = reg.Execute(reference, moving)
        volume = sitk.Resample(moving, reference, final, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
        corrected[..., i] = np.transpose(sitk.GetArrayFromImage(volume), (2, 1, 0))
        records.append(
            {
                "frame": i,
                "parameters": [float(v) for v in final.GetParameters()],
                "metric": float(reg.GetMetricValue()),
            }
        )
    return corrected, records


def _save(data: np.ndarray, reference: nib.Nifti1Image, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = reference.header.copy()
    header.set_data_dtype(np.float32)
    nib.save(nib.Nifti1Image(np.asarray(data, np.float32), reference.affine, header), path)
    return path


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


def tissue_summaries(
    image: np.ndarray, gm: np.ndarray, wm: np.ndarray
) -> dict[str, dict[str, float | int]]:
    result = {}
    for name, mask in (("gm", gm), ("wm", wm)):
        values = image[np.asarray(mask, bool) & np.isfinite(image)]
        result[name] = {
            "n_voxels": int(values.size),
            "mean": float(np.mean(values)) if values.size else float("nan"),
            "median": float(np.median(values)) if values.size else float("nan"),
            "sd": float(np.std(values)) if values.size else float("nan"),
        }
    return result


def process_perfusion(
    asl_file: str | Path,
    metadata_file: str | Path,
    t1_file: str | Path,
    gm_mask_file: str | Path,
    wm_mask_file: str | Path,
    output_dir: str | Path,
    subject: str,
    session: str,
    context_file: str | Path | None = None,
    m0_file: str | Path | None = None,
    motion_correct: bool = True,
    force: bool = False,
    parameters: dict[str, float] | None = None,
) -> dict:
    """Produce quantitative CBF only with M0/timing, otherwise labelled relative perfusion."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    prefix = f"{subject}_{session}"
    provenance = root / f"{prefix}_desc-perfusion_provenance.json"
    if provenance.exists() and not force:
        record = json.loads(provenance.read_text(encoding="utf-8"))
        record["resumed"] = True
        return record
    image = nib.load(str(asl_file))
    data = image.get_fdata(dtype=np.float32)
    metadata = json.loads(Path(metadata_file).read_text(encoding="utf-8"))
    context = read_asl_context(context_file) if context_file else None
    effective_parameters = {
        "labeling_efficiency": 0.85,
        "blood_t1_s": 1.65,
        "blood_brain_partition": 0.9,
        **(parameters or {}),
    }
    assessment = assess_asl(data.shape, metadata, context)
    warnings = list(assessment.warnings)
    motion = []
    if assessment.organization == "unsupported":
        record = {
            "subject": subject,
            "session": session,
            "status": "unsupported",
            "assessment": asdict(assessment),
            "warnings": warnings,
            "resumed": False,
        }
        provenance.write_text(json.dumps(record, indent=2), encoding="utf-8")
        return record
    if data.ndim == 3:
        perfusion = data
        mean_control = data
        rejected: list[int] = []
    else:
        if motion_correct:
            data, motion = motion_correct_asl(data, image.affine)
        differences, controls, skipped = pair_control_label(data, context or [])
        keep, rejected = reject_pair_outliers(differences)
        warnings.extend([f"skipped unpaired frame {i}" for i in skipped])
        if rejected:
            warnings.append(f"rejected ASL pairs as temporal outliers: {rejected}")
        mean_delta = np.nanmean(differences[..., keep], axis=3)
        mean_control = np.nanmean(controls[..., keep], axis=3)
        if assessment.quantitative_cbf_supported and m0_file:
            m0 = nib.load(str(m0_file)).get_fdata(dtype=np.float32)
            if m0.shape != mean_delta.shape:
                raise ValueError("M0 and ASL spatial grids must match before quantification.")
            perfusion = quantify_pcasl(
                mean_delta,
                m0,
                float(metadata["LabelingDuration"]),
                float(metadata["PostLabelingDelay"]),
                **effective_parameters,
            )
        else:
            if assessment.quantitative_cbf_supported and not m0_file:
                warnings.append(
                    "calibration metadata suggests M0, but no M0 image was supplied; using relative perfusion"
                )
            assessment.quantitative_cbf_supported = False
            assessment.output_quantity = "relative_perfusion_percent_control"
            assessment.units = "% mean control signal"
            perfusion = relative_perfusion(mean_delta, mean_control)
    perfusion_foreground_threshold = None
    if data.ndim == 4:
        positive_control = mean_control[np.isfinite(mean_control) & (mean_control > 0)]
        perfusion_foreground_threshold = (
            0.1 * float(np.percentile(positive_control, 99)) if positive_control.size else np.inf
        )
        perfusion[mean_control <= perfusion_foreground_threshold] = 0
    nonfinite = int((~np.isfinite(perfusion)).sum())
    perfusion = np.where(np.isfinite(perfusion), perfusion, 0).astype(np.float32)
    if nonfinite:
        warnings.append(f"replaced {nonfinite} nonfinite perfusion voxels with zero")
    native_map = _save(
        perfusion, image, root / f"{prefix}_space-ASL_desc-{assessment.output_quantity}_map.nii.gz"
    )
    control_map = _save(
        mean_control, image, root / f"{prefix}_space-ASL_desc-meancontrol_map.nii.gz"
    )
    registered_control = root / f"{prefix}_space-T1w_desc-meancontrol_map.nii.gz"
    transform = root / f"{prefix}_from-ASL_to-T1w_xfm.tfm"
    registration = register_flair_to_t1(control_map, t1_file, registered_control, transform)
    t1_map = _resample_map(
        native_map,
        t1_file,
        transform,
        root / f"{prefix}_space-T1w_desc-{assessment.output_quantity}_map.nii.gz",
    )
    registered = nib.load(str(t1_map)).get_fdata(dtype=np.float32)
    gm = nib.load(str(gm_mask_file)).get_fdata() > 0
    wm = nib.load(str(wm_mask_file)).get_fdata() > 0
    summaries = tissue_summaries(registered, gm, wm)
    finite_values = registered[np.isfinite(registered) & (registered != 0)]
    if (
        assessment.quantitative_cbf_supported
        and finite_values.size
        and (np.percentile(finite_values, 99) > 200 or np.percentile(finite_values, 1) < -20)
    ):
        warnings.append("quantitative CBF contains extreme values outside -20 to 200 mL/100g/min")
    record = {
        "subject": subject,
        "session": session,
        "status": "warn" if warnings else "pass",
        "assessment": asdict(assessment),
        "warnings": warnings,
        "inputs": {
            "asl": str(asl_file),
            "metadata": str(metadata_file),
            "context": str(context_file) if context_file else None,
            "m0": str(m0_file) if m0_file else None,
        },
        "outputs": {
            "native_map": str(native_map),
            "t1_map": str(t1_map),
            "mean_control": str(control_map),
            "registered_control": str(registered_control),
            "transform": str(transform),
        },
        "registration": registration,
        "motion": motion,
        "rejected_pairs": rejected,
        "tissue_summaries": summaries,
        "parameters": effective_parameters,
        "acquisition_metadata": {
            key: metadata.get(key)
            for key in (
                "ArterialSpinLabelingType",
                "PostLabelingDelay",
                "LabelingDuration",
                "M0Type",
                "TotalAcquiredPairs",
                "RepetitionTime",
            )
        },
        "perfusion_foreground_threshold": perfusion_foreground_threshold,
        "resumed": False,
    }
    provenance.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def mean_perfusion(image: np.ndarray, mask: np.ndarray) -> float:
    values = np.asarray(image, np.float32)[np.asarray(mask, bool)]
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else float("nan")
