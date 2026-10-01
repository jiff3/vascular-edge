"""Structural and WMH quality-control figures and machine-readable records."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.colors import ListedColormap


def _display(data: np.ndarray) -> np.ndarray:
    values = data[np.isfinite(data) & (data != 0)]
    if values.size == 0:
        return np.zeros_like(data)
    lo, hi = np.percentile(values, [1, 99])
    return np.clip((data - lo) / max(hi - lo, 1e-6), 0, 1)


def _slice(data: np.ndarray, index: int) -> np.ndarray:
    return np.rot90(data[:, :, index])


def structural_qc(
    t1_file: str | Path,
    flair_file: str | Path,
    tissue_files: dict[str, str | Path],
    wmh_file: str | Path,
    output_dir: str | Path,
    subject: str,
    session: str,
    registration_metric: float | None = None,
    force: bool = False,
    wmh_backend: str | None = None,
) -> dict:
    """Generate a six-panel structural report and QC JSON; never silently excludes."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    prefix = f"{subject}_{session}"
    figure_file = root / f"{prefix}_desc-structural_qc.png"
    json_file = root / f"{prefix}_desc-structural_qc.json"
    if not force and figure_file.exists() and json_file.exists():
        record = json.loads(json_file.read_text(encoding="utf-8"))
        record["resumed"] = True
        return record
    t1_img = nib.load(str(t1_file))
    t1 = t1_img.get_fdata(dtype=np.float32)
    flair = nib.load(str(flair_file)).get_fdata(dtype=np.float32)
    tissues = {k: nib.load(str(v)).get_fdata() > 0 for k, v in tissue_files.items()}
    wmh = nib.load(str(wmh_file)).get_fdata() > 0
    arrays = [flair, wmh, *tissues.values()]
    if any(array.shape != t1.shape for array in arrays):
        raise ValueError("All QC inputs must share the T1 grid.")
    warnings: list[str] = []
    brain_voxels = int(np.logical_or.reduce(list(tissues.values())).sum()) if tissues else 0
    lesion_voxels = int(wmh.sum())
    voxel_ml = float(np.prod(t1_img.header.get_zooms()[:3]) / 1000)
    lesion_ml = lesion_voxels * voxel_ml
    if brain_voxels == 0:
        warnings.append("empty tissue/brain masks")
    if lesion_voxels == 0:
        warnings.append("empty WMH mask")
    if brain_voxels and lesion_voxels / brain_voxels > 0.10:
        warnings.append("WMH burden exceeds 10% of segmented brain")
    if registration_metric is None or not np.isfinite(registration_metric):
        warnings.append("registration metric unavailable/non-finite")
    overlap_denominator = min(int((t1 != 0).sum()), int((flair != 0).sum()))
    registration_overlap = (
        float(((t1 != 0) & (flair != 0)).sum() / overlap_denominator)
        if overlap_denominator
        else 0.0
    )
    if registration_overlap < 0.5:
        warnings.append(f"low T1/FLAIR foreground overlap: {registration_overlap:.3f}")
    if wmh_backend == "fallback":
        warnings.append("WMH fallback is synthetic/testing-only and not biologically validated")
    z_indices = [int(t1.shape[2] * f) for f in (0.3, 0.5, 0.7)]
    mid = z_indices[1]
    t1n, flairn = _display(t1), _display(flair)
    tissue_label = np.zeros(t1.shape, np.uint8)
    for value, key in enumerate(("csf", "gm", "wm"), start=1):
        if key in tissues:
            tissue_label[tissues[key]] = value
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), facecolor="white")
    axes[0, 0].imshow(_slice(t1n, mid), cmap="gray")
    axes[0, 0].set_title("N4 T1w")
    axes[0, 1].imshow(_slice(flairn, mid), cmap="gray")
    axes[0, 1].set_title("FLAIR in T1w space")
    rgb = np.zeros((*_slice(t1n, mid).shape, 3), np.float32)
    rgb[..., 0] = _slice(t1n, mid)
    rgb[..., 1] = _slice(flairn, mid)
    axes[0, 2].imshow(rgb)
    axes[0, 2].set_title("Registration: T1 red / FLAIR green")
    axes[1, 0].imshow(_slice(t1n, mid), cmap="gray")
    axes[1, 0].imshow(
        np.ma.masked_where(_slice(tissue_label, mid) == 0, _slice(tissue_label, mid)),
        cmap=ListedColormap(["#4c78a8", "#f58518", "#54a24b"]),
        alpha=0.55,
        vmin=1,
        vmax=3,
    )
    axes[1, 0].set_title("Tissues: CSF / GM / WM")
    axes[1, 1].imshow(_slice(flairn, mid), cmap="gray")
    lesion_slice = _slice(wmh, mid)
    if lesion_slice.any():
        axes[1, 1].contour(lesion_slice, levels=[0.5], colors="cyan", linewidths=1)
    axes[1, 1].set_title(f"WMH overlay ({lesion_ml:.2f} mL)")
    montage = np.concatenate([_slice(flairn, z) for z in z_indices], axis=1)
    montage_wmh = np.concatenate([_slice(wmh, z) for z in z_indices], axis=1)
    axes[1, 2].imshow(montage, cmap="gray")
    if montage_wmh.any():
        axes[1, 2].contour(montage_wmh, levels=[0.5], colors="cyan", linewidths=0.7)
    axes[1, 2].set_title("FLAIR/WMH representative slices")
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle(f"Structural QC — {subject} {session}", fontsize=14)
    fig.tight_layout()
    fig.savefig(figure_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    status = "fail" if brain_voxels == 0 else ("warn" if warnings else "pass")
    record = {
        "subject": subject,
        "session": session,
        "status": status,
        "warnings": warnings,
        "metrics": {
            "brain_voxels": brain_voxels,
            "wmh_voxels": lesion_voxels,
            "wmh_volume_ml": lesion_ml,
            "registration_metric": registration_metric,
            "registration_foreground_overlap": registration_overlap,
            "wmh_backend": wmh_backend,
        },
        "figure": str(figure_file),
        "resumed": False,
    }
    json_file.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def save_mask_overlay(image: np.ndarray, mask: np.ndarray, output_file: str | Path) -> Path:
    data = np.asarray(image, np.float32)
    lesion = np.asarray(mask, bool)
    index = data.shape[2] // 2
    output = Path(output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(_slice(_display(data), index), cmap="gray")
    current = _slice(lesion, index)
    if current.any():
        ax.contour(current, levels=[0.5], colors="cyan", linewidths=0.7)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output
