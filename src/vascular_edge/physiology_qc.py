"""Static QC reports for perfusion and optional CVR outputs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def _scale(data: np.ndarray, mask: np.ndarray | None = None) -> tuple[float, float]:
    valid = np.isfinite(data) & (data != 0)
    if mask is not None:
        valid &= mask
    values = data[valid]
    return (
        (float(np.percentile(values, 2)), float(np.percentile(values, 98)))
        if values.size
        else (0.0, 1.0)
    )


def _sl(data: np.ndarray, z: int) -> np.ndarray:
    return np.rot90(data[:, :, z])


def qc_physiology(
    t1_file: str | Path,
    flair_file: str | Path,
    perfusion_file: str | Path | None,
    gm_file: str | Path,
    wm_file: str | Path,
    output_dir: str | Path,
    subject: str,
    session: str,
    perfusion_units: str | None = None,
    raw_asl_file: str | Path | None = None,
    cvr_file: str | Path | None = None,
    cvr_units: str | None = None,
    force: bool = False,
    registered_reference_file: str | Path | None = None,
) -> dict:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    prefix = f"{subject}_{session}"
    figure = root / f"{prefix}_desc-physiology_qc.png"
    metrics_file = root / f"{prefix}_desc-physiology_qc.json"
    if not force and figure.exists() and metrics_file.exists():
        record = json.loads(metrics_file.read_text(encoding="utf-8"))
        record["resumed"] = True
        return record
    t1 = nib.load(str(t1_file)).get_fdata(dtype=np.float32)
    gm = nib.load(str(gm_file)).get_fdata() > 0
    wm = nib.load(str(wm_file)).get_fdata() > 0
    warnings: list[str] = []
    z = t1.shape[2] // 2
    perfusion = (
        nib.load(str(perfusion_file)).get_fdata(dtype=np.float32) if perfusion_file else None
    )
    cvr = nib.load(str(cvr_file)).get_fdata(dtype=np.float32) if cvr_file else None
    if perfusion is None:
        warnings.append("perfusion map unavailable")
    elif perfusion.shape != t1.shape:
        raise ValueError("Perfusion and T1 grids differ.")
    if cvr is not None and cvr.shape != t1.shape:
        raise ValueError("CVR and T1 grids differ.")
    if perfusion_units and "ml/100g/min" not in perfusion_units.lower().replace(" ", ""):
        warnings.append("perfusion is relative, not absolute CBF")
    if cvr_units and "mmhg" not in cvr_units.lower().replace(" ", ""):
        warnings.append("BOLD response is not calibrated by end-tidal CO2")
    if perfusion is not None:
        tissue_values = perfusion[(gm | wm) & np.isfinite(perfusion)]
        if (
            tissue_values.size
            and perfusion_units
            and "%" in perfusion_units
            and np.percentile(np.abs(tissue_values), 99) > 50
        ):
            warnings.append("relative perfusion has extreme tissue values above 50% control signal")
        if (
            tissue_values.size
            and perfusion_units
            and "ml/100g/min" in perfusion_units.lower().replace(" ", "")
        ):
            if np.percentile(tissue_values, 99) > 200 or np.percentile(tissue_values, 1) < -20:
                warnings.append("CBF tissue values extend outside -20 to 200 mL/100g/min")
    if cvr is not None:
        response_values = cvr[(gm | wm) & np.isfinite(cvr)]
        if response_values.size and np.percentile(np.abs(response_values), 99) > 20:
            warnings.append("BOLD response tissue values exceed 20%")
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    tlo, thi = _scale(t1)
    if registered_reference_file:
        reference = nib.load(str(registered_reference_file)).get_fdata(dtype=np.float32)
        if reference.shape != t1.shape:
            raise ValueError("Registered physiology reference and T1 grids differ.")
        rlo, rhi = _scale(reference)
        red = np.clip((_sl(t1, z) - tlo) / max(thi - tlo, 1e-6), 0, 1)
        green = np.clip((_sl(reference, z) - rlo) / max(rhi - rlo, 1e-6), 0, 1)
        rgb = np.zeros((*red.shape, 3))
        rgb[..., 0] = red
        rgb[..., 1] = green
        axes[0, 0].imshow(rgb)
        axes[0, 0].set_title("Registration: T1 red / physiology green")
    else:
        axes[0, 0].imshow(_sl(t1, z), cmap="gray", vmin=tlo, vmax=thi)
        axes[0, 0].set_title("T1w reference")
    if raw_asl_file:
        raw = nib.load(str(raw_asl_file)).get_fdata(dtype=np.float32)
        frame = raw[..., 0] if raw.ndim == 4 else raw
        lo, hi = _scale(frame)
        axes[0, 1].imshow(_sl(frame, frame.shape[2] // 2), cmap="gray", vmin=lo, vmax=hi)
        axes[0, 1].set_title("Representative ASL frame")
    else:
        axes[0, 1].text(0.5, 0.5, "ASL unavailable", ha="center")
        axes[0, 1].set_title("ASL")
    if perfusion is not None:
        lo, hi = _scale(perfusion, gm | wm)
        im = axes[0, 2].imshow(_sl(perfusion, z), cmap="magma", vmin=lo, vmax=hi)
        fig.colorbar(im, ax=axes[0, 2], fraction=0.046)
        axes[0, 2].set_title(f"Perfusion ({perfusion_units or 'units unknown'})")
        axes[1, 0].imshow(_sl(t1, z), cmap="gray", vmin=tlo, vmax=thi)
        axes[1, 0].imshow(
            np.ma.masked_where(_sl((gm | wm), z) == 0, _sl(perfusion, z)),
            cmap="magma",
            vmin=lo,
            vmax=hi,
            alpha=0.65,
        )
        axes[1, 0].set_title("Perfusion over T1w")
        gm_values = perfusion[gm & np.isfinite(perfusion)]
        wm_values = perfusion[wm & np.isfinite(perfusion)]
        axes[1, 1].hist(gm_values, bins=40, alpha=0.6, label="GM")
        axes[1, 1].hist(wm_values, bins=40, alpha=0.6, label="WM")
        axes[1, 1].legend()
        axes[1, 1].set_title("Tissue distributions")
    else:
        for ax in (axes[0, 2], axes[1, 0], axes[1, 1]):
            ax.text(0.5, 0.5, "No perfusion output", ha="center")
    if cvr is not None:
        lo, hi = _scale(cvr, gm | wm)
        bound = max(abs(lo), abs(hi))
        im = axes[1, 2].imshow(_sl(cvr, z), cmap="coolwarm", vmin=-bound, vmax=bound)
        fig.colorbar(im, ax=axes[1, 2], fraction=0.046)
        axes[1, 2].set_title(f"CVR/response ({cvr_units or 'units unknown'})")
    else:
        axes[1, 2].text(0.5, 0.5, "CVR unavailable", ha="center")
        axes[1, 2].set_title("Optional CVR")
    for ax in axes.ravel():
        if ax is not axes[1, 1]:
            ax.axis("off")
    fig.suptitle(f"Physiology QC — {subject} {session}")
    fig.tight_layout()
    fig.savefig(figure, dpi=150)
    plt.close(fig)
    summaries = {}
    if perfusion is not None:
        for name, mask in (("gm", gm), ("wm", wm)):
            values = perfusion[mask & np.isfinite(perfusion)]
            summaries[name] = {
                "n": int(values.size),
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "sd": float(np.std(values)),
            }
    record = {
        "subject": subject,
        "session": session,
        "status": "warn" if warnings else "pass",
        "warnings": warnings,
        "perfusion_units": perfusion_units,
        "cvr_units": cvr_units,
        "tissue_summaries": summaries,
        "figure": str(figure),
        "resumed": False,
    }
    metrics_file.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record
