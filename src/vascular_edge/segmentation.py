"""Replaceable WMH inference backends and deterministic postprocessing."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage


class WMHBackendUnavailable(RuntimeError):
    """Raised when a configured external segmentation backend is unavailable."""


@dataclass
class LesionMetrics:
    raw_voxels: int
    postprocessed_voxels: int
    lesion_volume_ml: float
    component_count: int
    removed_component_count: int


def fallback_wmh_mask(
    flair: np.ndarray,
    brain_mask: np.ndarray | None = None,
    z_threshold: float = 2.5,
    min_voxels: int = 3,
) -> np.ndarray:
    """Deterministic robust-z synthetic/testing fallback; not biologically validated."""
    data = np.asarray(flair, dtype=np.float32)
    valid = np.isfinite(data) & (data > 0)
    if brain_mask is not None:
        valid &= np.asarray(brain_mask, dtype=bool)
    values = data[valid]
    if values.size == 0:
        return np.zeros(data.shape, dtype=bool)
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    scale = max(1.4826 * mad, np.finfo(np.float32).eps)
    candidate = valid & ((data - median) / scale >= z_threshold)
    labels, count = ndimage.label(candidate)
    if count == 0:
        return candidate
    sizes = np.bincount(labels.ravel())
    keep = sizes >= min_voxels
    keep[0] = False
    return keep[labels]


def run_truenet(
    t1_file: str | Path,
    flair_file: str | Path,
    output_file: str | Path,
    model_dir: str | Path,
    executable: str = "truenet",
) -> Path:
    """Thin CPU wrapper for TrUE-Net; external code/weights remain separately installed.

    The configured executable must accept the documented project adapter flags.
    No model code or weights are vendored into this repository.
    """
    command = shutil.which(executable)
    if not command:
        raise WMHBackendUnavailable(
            f"TrUE-Net executable '{executable}' was not found. Install the pinned external backend "
            "or set segmentation.backend=fallback for synthetic/testing use only."
        )
    model = Path(model_dir)
    if not model.exists():
        raise WMHBackendUnavailable(f"TrUE-Net model directory does not exist: {model}")
    output = Path(output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="truenet-", dir=output.parent) as temporary:
        work = Path(temporary)
        inputs = work / "inputs"
        predictions = work / "predictions"
        inputs.mkdir()
        predictions.mkdir()
        stem = "vascular_edge_case"
        for source, suffix in ((Path(t1_file), "T1"), (Path(flair_file), "FLAIR")):
            destination = inputs / f"{stem}_{suffix}.nii.gz"
            try:
                os.link(source, destination)
            except OSError:
                shutil.copy2(source, destination)
        subprocess.run(
            [
                command,
                "evaluate",
                "-i",
                str(inputs),
                "-m",
                str(model),
                "-o",
                str(predictions),
                "--use_cpu",
            ],
            check=True,
        )
        candidates = sorted(predictions.glob("*.nii*"))
        if not candidates:
            raise RuntimeError("TrUE-Net completed without producing a NIfTI prediction.")
        shutil.copy2(candidates[0], output)
    return output


def postprocess_wmh(
    raw_mask: np.ndarray,
    voxel_sizes_mm: tuple[float, float, float],
    min_component_mm3: float = 3.0,
    wm_mask: np.ndarray | None = None,
    wm_dilation_voxels: int = 1,
) -> tuple[np.ndarray, LesionMetrics]:
    """Remove small components and optionally constrain to dilated white matter."""
    raw = np.asarray(raw_mask, bool)
    _, raw_component_count = ndimage.label(raw)
    constrained = raw.copy()
    if wm_mask is not None:
        territory = ndimage.binary_dilation(
            np.asarray(wm_mask, bool), iterations=wm_dilation_voxels
        )
        constrained &= territory
    labels, _ = ndimage.label(constrained)
    voxel_volume = float(np.prod(voxel_sizes_mm))
    minimum = max(1, int(np.ceil(min_component_mm3 / voxel_volume)))
    sizes = np.bincount(labels.ravel())
    keep = sizes >= minimum
    keep[0] = False
    result = keep[labels]
    _, component_count = ndimage.label(result)
    metrics = LesionMetrics(
        int(raw.sum()),
        int(result.sum()),
        float(result.sum() * voxel_volume / 1000.0),
        int(component_count),
        max(0, int(raw_component_count - component_count)),
    )
    return result, metrics


def segment_wmh_files(
    flair_file: str | Path,
    t1_file: str | Path,
    brain_mask_file: str | Path,
    wm_mask_file: str | Path,
    output_dir: str | Path,
    subject: str,
    session: str,
    backend: str = "fallback",
    force: bool = False,
    z_threshold: float = 2.5,
    min_component_mm3: float = 3.0,
    model_dir: str | Path | None = None,
    executable: str = "truenet",
) -> dict:
    """Run inference, retain the raw mask, postprocess, and save metrics/provenance."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    prefix = f"{subject}_{session}_space-T1w"
    raw_file = root / f"{prefix}_desc-WMHraw_mask.nii.gz"
    final_file = root / f"{prefix}_desc-WMH_mask.nii.gz"
    metrics_file = root / f"{prefix}_desc-WMH_metrics.json"
    if not force and all(p.exists() for p in (raw_file, final_file, metrics_file)):
        record = json.loads(metrics_file.read_text(encoding="utf-8"))
        record["resumed"] = True
        return record
    flair_img = nib.load(str(flair_file))
    flair = flair_img.get_fdata(dtype=np.float32)
    brain = nib.load(str(brain_mask_file)).get_fdata() > 0
    wm = nib.load(str(wm_mask_file)).get_fdata() > 0
    if not (flair.shape == brain.shape == wm.shape):
        raise ValueError("FLAIR, brain, and WM masks must share the T1 grid.")
    if backend == "fallback":
        raw = fallback_wmh_mask(flair, brain, z_threshold=z_threshold, min_voxels=1)
        header = flair_img.header.copy()
        header.set_data_dtype(np.uint8)
        nib.save(nib.Nifti1Image(raw.astype(np.uint8), flair_img.affine, header), str(raw_file))
        backend_note = "Deterministic robust-z fallback for tests/sensitivity analyses; not biologically validated."
    elif backend == "truenet":
        if model_dir is None:
            raise WMHBackendUnavailable("TrUE-Net requires segmentation.model_dir.")
        run_truenet(t1_file, flair_file, raw_file, model_dir, executable)
        raw = nib.load(str(raw_file)).get_fdata() > 0.5
        backend_note = (
            "External TrUE-Net CPU inference; version and weights must match project configuration."
        )
    else:
        raise ValueError(f"Unknown WMH backend: {backend}")
    zooms = tuple(float(v) for v in flair_img.header.get_zooms()[:3])
    final, metrics = postprocess_wmh(raw, zooms, min_component_mm3, wm)
    header = flair_img.header.copy()
    header.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(final.astype(np.uint8), flair_img.affine, header), str(final_file))
    record = {
        "subject": subject,
        "session": session,
        "backend": backend,
        "backend_note": backend_note,
        "parameters": {"z_threshold": z_threshold, "min_component_mm3": min_component_mm3},
        "raw_mask": str(raw_file),
        "final_mask": str(final_file),
        "metrics": asdict(metrics),
        "resumed": False,
    }
    metrics_file.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record
