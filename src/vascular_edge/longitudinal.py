"""Conservative within-subject longitudinal WMH conversion analysis."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage
from skimage.registration import phase_cross_correlation

DISTANCE_EDGES_MM = (0.0, 2.0, 4.0, 6.0, 10.0, np.inf)


def _wave_number(session: str) -> int | None:
    match = re.fullmatch(r"(?:ses-)?wave[-_]?([0-9]+)", str(session), flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def order_sessions(
    sessions: Sequence[str], sessions_table: pd.DataFrame | None = None
) -> list[str]:
    """Order sessions from metadata timing, or explicit DLBS wave labels."""
    unique = list(dict.fromkeys(str(x) for x in sessions))
    if len(unique) < 2:
        raise ValueError("At least two sessions are required for longitudinal analysis.")
    if sessions_table is not None:
        frame = sessions_table.copy()
        key = next((x for x in ("session_id", "session", "ses") if x in frame), None)
        order_key = next(
            (
                x
                for x in (
                    "acq_time",
                    "scan_date",
                    "days_since_baseline",
                    "years_since_baseline",
                    "timepoint",
                )
                if x in frame
            ),
            None,
        )
        if key and order_key:
            subset = frame[frame[key].astype(str).isin(unique)].copy()
            if len(subset) == len(unique):
                values = (
                    pd.to_datetime(subset[order_key], errors="coerce")
                    if order_key in {"acq_time", "scan_date"}
                    else pd.to_numeric(subset[order_key], errors="coerce")
                )
                if values.notna().all() and not values.duplicated().any():
                    subset["_order"] = values
                    return subset.sort_values("_order")[key].astype(str).tolist()
    waves = [_wave_number(value) for value in unique]
    if all(value is not None for value in waves) and len(set(waves)) == len(waves):
        return [session for _, session in sorted(zip(waves, unique))]
    raise ValueError(
        "Session order is ambiguous; supply a sessions.tsv with acquisition/timing metadata."
    )


def choose_longitudinal_pair(
    sessions: Sequence[str],
    sessions_table: pd.DataFrame | None = None,
    strategy: str = "earliest-latest",
) -> tuple[str, str]:
    ordered = order_sessions(sessions, sessions_table)
    if strategy == "earliest-latest":
        return ordered[0], ordered[-1]
    if strategy == "consecutive-first":
        return ordered[0], ordered[1]
    raise ValueError(f"Unknown session-pair strategy: {strategy}")


def _remove_small(mask: np.ndarray, voxel_mm3: float, minimum_mm3: float) -> np.ndarray:
    labels, count = ndimage.label(mask, structure=ndimage.generate_binary_structure(3, 2))
    if count == 0:
        return np.zeros_like(mask, bool)
    sizes = np.bincount(labels.ravel()) * voxel_mm3
    keep = np.flatnonzero(sizes >= minimum_mm3)
    keep = keep[keep != 0]
    return np.isin(labels, keep)


def lesion_change(baseline_mask: np.ndarray, followup_mask: np.ndarray) -> dict[str, np.ndarray]:
    """Label raw new, persistent, and resolved voxels after registration."""
    baseline = np.asarray(baseline_mask, dtype=bool)
    followup = np.asarray(followup_mask, dtype=bool)
    if baseline.shape != followup.shape:
        raise ValueError("Masks must share a grid; register/resample before comparison.")
    return {
        "new": followup & ~baseline,
        "persistent": followup & baseline,
        "resolved": baseline & ~followup,
    }


def classify_longitudinal_change(
    baseline_wmh: np.ndarray,
    followup_wmh: np.ndarray,
    baseline_wm: np.ndarray,
    valid_coverage: np.ndarray,
    voxel_sizes_mm: Sequence[float],
    boundary_uncertainty_mm: float = 1.0,
    min_new_component_mm3: float = 3.0,
) -> dict[str, np.ndarray]:
    """Define conservative conversion and stable-NAWM analysis classes.

    A physical margin outside baseline WMH is unsuitable, preventing resampling
    jitter at the lesion boundary from being labeled as incident tissue.
    """
    baseline = np.asarray(baseline_wmh, bool)
    followup = np.asarray(followup_wmh, bool)
    wm = np.asarray(baseline_wm, bool)
    valid = np.asarray(valid_coverage, bool)
    if not (baseline.shape == followup.shape == wm.shape == valid.shape):
        raise ValueError("Longitudinal masks must share the baseline grid.")
    spacing = tuple(float(x) for x in voxel_sizes_mm)
    distance = ndimage.distance_transform_edt(~baseline, sampling=spacing).astype(np.float32)
    uncertain = (
        (~baseline) & (distance <= float(boundary_uncertainty_mm))
        if baseline.any() and boundary_uncertainty_mm > 0
        else np.zeros_like(baseline)
    )
    analysis_nawm = wm & ~baseline & ~uncertain & valid
    raw_new = followup & ~baseline
    converting = _remove_small(
        raw_new & analysis_nawm, float(np.prod(spacing)), min_new_component_mm3
    )
    stable = analysis_nawm & ~followup
    persistent = baseline & followup
    resolved = baseline & ~followup
    unsuitable = ~(stable | converting | persistent | resolved)
    return {
        "persistent": persistent,
        "converting": converting,
        "stable_nawm": stable,
        "resolved": resolved,
        "unsuitable": unsuitable,
        "raw_new": raw_new,
        "boundary_uncertainty": uncertain,
        "baseline_distance_mm": distance,
        "baseline_wmh": baseline,
        "followup_wmh": followup,
    }


def _resample(
    moving_file: str | Path,
    reference_file: str | Path,
    transform_file: str | Path,
    output_file: str | Path,
    nearest: bool = False,
    pixel_type: int = sitk.sitkFloat32,
) -> Path:
    moving = sitk.ReadImage(str(moving_file))
    reference = sitk.ReadImage(str(reference_file))
    interpolation = sitk.sitkNearestNeighbor if nearest else sitk.sitkLinear
    result = sitk.Resample(
        moving, reference, sitk.ReadTransform(str(transform_file)), interpolation, 0.0, pixel_type
    )
    output = Path(output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(result, str(output))
    return output


def register_followup_to_baseline(
    followup_t1: str | Path,
    baseline_t1: str | Path,
    output_file: str | Path,
    transform_file: str | Path,
    sampling_percentage: float = 0.20,
) -> dict:
    """Deterministic rigid/guarded-affine T1 registration with honest QC scores."""
    fixed = sitk.Cast(sitk.ReadImage(str(baseline_t1)), sitk.sitkFloat32)
    moving = sitk.Cast(sitk.ReadImage(str(followup_t1)), sitk.sitkFloat32)
    initial = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY
    )
    phase_initial = sitk.Euler3DTransform(initial)
    phase_shift_voxels = None
    if fixed.GetSize() == moving.GetSize():
        fixed_xyz = np.transpose(sitk.GetArrayFromImage(fixed), (2, 1, 0))
        moving_xyz = np.transpose(sitk.GetArrayFromImage(moving), (2, 1, 0))
        phase_shift_voxels = np.asarray(
            phase_cross_correlation(fixed_xyz, moving_xyz, upsample_factor=1)[0]
        )
        sample_offset = -phase_shift_voxels * np.asarray(fixed.GetSpacing())
        physical_offset = np.asarray(fixed.GetDirection()).reshape(3, 3) @ sample_offset
        phase_initial.SetTranslation(tuple(np.asarray(initial.GetTranslation()) + physical_offset))

    def method(iterations: int, learning_rate: float) -> sitk.ImageRegistrationMethod:
        reg = sitk.ImageRegistrationMethod()
        reg.SetMetricAsMattesMutualInformation(40)
        reg.SetMetricSamplingStrategy(reg.RANDOM)
        reg.SetMetricSamplingPercentage(sampling_percentage, seed=2026)
        reg.SetInterpolator(sitk.sitkLinear)
        reg.SetOptimizerAsGradientDescent(learning_rate, iterations, 1e-6, 10)
        reg.SetOptimizerScalesFromPhysicalShift()
        reg.SetShrinkFactorsPerLevel([4, 2, 1])
        reg.SetSmoothingSigmasPerLevel([2, 1, 0])
        reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        return reg

    rigid_method = method(80, 0.50)
    rigid_method.SetInitialTransform(phase_initial, inPlace=False)
    rigid = rigid_method.Execute(fixed, moving)
    rigid_base = rigid.GetBackTransform() if isinstance(rigid, sitk.CompositeTransform) else rigid
    affine = sitk.AffineTransform(3)
    affine.SetCenter(rigid_base.GetCenter())
    affine.SetMatrix(rigid_base.GetMatrix())
    affine.SetTranslation(rigid_base.GetTranslation())
    affine_method = method(50, 0.35)
    affine_method.SetInitialTransform(affine, inPlace=False)
    affine_error = None
    try:
        refined = affine_method.Execute(fixed, moving)
    except RuntimeError as exc:
        refined = None
        affine_error = str(exc).splitlines()[-1]

    fixed_array = sitk.GetArrayFromImage(fixed).astype(np.float32)
    fixed_fg = fixed_array > 0

    def score(transform: sitk.Transform) -> tuple[float, float, float, sitk.Image]:
        image = sitk.Resample(moving, fixed, transform, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
        array = sitk.GetArrayFromImage(image).astype(np.float32)
        foreground = array > 0
        denominator = int(fixed_fg.sum() + foreground.sum())
        dice = float(2 * (fixed_fg & foreground).sum() / denominator) if denominator else 0.0
        overlap = fixed_fg & foreground
        if overlap.sum() > 32:
            x, y = fixed_array[overlap], array[overlap]
            correlation = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else 0.0
        else:
            correlation = 0.0
        return dice + 0.50 * max(correlation, 0), dice, correlation, image

    candidates = {
        "rigid": rigid,
        "phase_initialized": phase_initial,
        "geometry_initialized": initial,
    }
    affine_determinant = None
    affine_singular_values = None
    if refined is not None:
        refined_base = (
            refined.GetBackTransform() if isinstance(refined, sitk.CompositeTransform) else refined
        )
        matrix = np.asarray(refined_base.GetMatrix()).reshape(3, 3)
        affine_determinant = float(np.linalg.det(matrix))
        affine_singular_values = np.linalg.svd(matrix, compute_uv=False)
        if 0.85 <= affine_determinant <= 1.15 and np.all(
            (affine_singular_values >= 0.90) & (affine_singular_values <= 1.10)
        ):
            candidates["affine"] = refined
        else:
            affine_error = f"affine rejected by scale guard: det={affine_determinant:.3f}, singular_values={affine_singular_values.tolist()}"
    evaluated = {name: score(transform) for name, transform in candidates.items()}
    selected_stage = max(evaluated, key=lambda name: evaluated[name][0])
    selected = candidates[selected_stage]
    _, dice, correlation, resampled = evaluated[selected_stage]
    output = Path(output_file)
    transform_path = Path(transform_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(resampled, str(output))
    sitk.WriteTransform(selected, str(transform_path))
    return {
        "metric": float(rigid_method.GetMetricValue()),
        "optimizer_stop": rigid_method.GetOptimizerStopConditionDescription(),
        "transform": str(transform_path),
        "selected_stage": selected_stage,
        "foreground_dice": dice,
        "intensity_correlation": correlation,
        "foreground_overlap": dice,
        "affine_determinant": affine_determinant,
        "affine_singular_values": affine_singular_values.tolist()
        if affine_singular_values is not None
        else None,
        "phase_shift_voxels_xyz": phase_shift_voxels.tolist()
        if phase_shift_voxels is not None
        else None,
        "affine_error": affine_error,
        "candidate_scores": {
            name: {
                "combined": values[0],
                "foreground_dice": values[1],
                "intensity_correlation": values[2],
            }
            for name, values in evaluated.items()
        },
    }


def _save(
    array: np.ndarray, reference: nib.Nifti1Image, path: Path, dtype: np.dtype = np.uint8
) -> Path:
    header = reference.header.copy()
    header.set_data_dtype(dtype)
    nib.save(nib.Nifti1Image(np.asarray(array, dtype=dtype), reference.affine, header), path)
    return path


def _stats(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, np.float32)
    values = values[np.isfinite(values)]
    if not values.size:
        return {
            "n_voxels": 0,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "iqr": np.nan,
            "p05": np.nan,
            "p95": np.nan,
        }
    p05, p25, p75, p95 = np.percentile(values, (5, 25, 75, 95))
    return {
        "n_voxels": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "iqr": float(p75 - p25),
        "p05": float(p05),
        "p95": float(p95),
    }


def summarize_longitudinal(
    classes: Mapping[str, np.ndarray],
    physiology: Mapping[str, np.ndarray],
    voxel_sizes_mm: Sequence[float],
    subject: str,
    baseline_session: str,
    followup_session: str,
    units: Mapping[str, str] | None = None,
    demographics: Mapping[str, object] | None = None,
    affine: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Create region summaries and distance-bin conversion fractions."""
    units = units or {}
    demographics = demographics or {}
    voxel_ml = float(np.prod(voxel_sizes_mm) / 1000)
    converting = np.asarray(classes["converting"], bool)
    stable = np.asarray(classes["stable_nawm"], bool)
    distance = np.asarray(classes["baseline_distance_mm"], np.float32)
    rows = []
    for tissue, mask in (("converting", converting), ("stable_nawm", stable)):
        for name, array in physiology.items():
            rows.append(
                {
                    "subject": subject,
                    "baseline_session": baseline_session,
                    "followup_session": followup_session,
                    "tissue": tissue,
                    "map": name,
                    "units": units.get(name, "unknown"),
                    "volume_ml": float(mask.sum() * voxel_ml),
                    **_stats(np.asarray(array)[mask]),
                }
            )
        rows.append(
            {
                "subject": subject,
                "baseline_session": baseline_session,
                "followup_session": followup_session,
                "tissue": tissue,
                "map": "baseline_distance",
                "units": "mm",
                "volume_ml": float(mask.sum() * voxel_ml),
                **_stats(distance[mask]),
            }
        )
    physiology_table = pd.DataFrame(rows)
    distance_rows = []
    at_risk = converting | stable
    for lower, upper in zip(DISTANCE_EDGES_MM[:-1], DISTANCE_EDGES_MM[1:]):
        in_bin = at_risk & (distance > lower) & (distance <= upper)
        converted = in_bin & converting
        denominator = int(in_bin.sum())
        distance_rows.append(
            {
                "subject": subject,
                "baseline_session": baseline_session,
                "followup_session": followup_session,
                "distance_lower_mm": lower,
                "distance_upper_mm": upper,
                "at_risk_voxels": denominator,
                "converting_voxels": int(converted.sum()),
                "conversion_fraction": float(converted.sum() / denominator)
                if denominator
                else np.nan,
            }
        )
    distance_table = pd.DataFrame(distance_rows)
    baseline = np.asarray(
        classes.get("baseline_wmh", classes["persistent"] | classes["resolved"]), bool
    )
    followup = np.asarray(
        classes.get("followup_wmh", classes["persistent"] | classes["raw_new"]), bool
    )
    baseline_ml = float(baseline.sum() * voxel_ml)
    followup_ml = float(followup.sum() * voxel_ml)
    coordinates = np.argwhere(converting)
    centroid_ijk = coordinates.mean(axis=0) if coordinates.size else np.full(3, np.nan)
    centroid_xyz = (
        nib.affines.apply_affine(affine, centroid_ijk)
        if affine is not None and coordinates.size
        else np.full(3, np.nan)
    )
    subject_row = {
        "subject": subject,
        "baseline_session": baseline_session,
        "followup_session": followup_session,
        "baseline_wmh_ml": baseline_ml,
        "followup_wmh_ml": followup_ml,
        "absolute_wmh_change_ml": followup_ml - baseline_ml,
        "percent_wmh_change": float(100 * (followup_ml - baseline_ml) / baseline_ml)
        if baseline_ml
        else np.nan,
        "newly_affected_ml": float(converting.sum() * voxel_ml),
        "converting_voxels": int(converting.sum()),
        "stable_nawm_voxels": int(stable.sum()),
        "conversion_centroid_i": float(centroid_ijk[0]),
        "conversion_centroid_j": float(centroid_ijk[1]),
        "conversion_centroid_k": float(centroid_ijk[2]),
        "conversion_centroid_x_mm": float(centroid_xyz[0]),
        "conversion_centroid_y_mm": float(centroid_xyz[1]),
        "conversion_centroid_z_mm": float(centroid_xyz[2]),
        **demographics,
    }
    for _, row in physiology_table[physiology_table["map"] != "baseline_distance"].iterrows():
        subject_row[f"{row['map']}_{row['tissue']}_median"] = row["median"]
        subject_row[f"{row['map']}_{row['tissue']}_mean"] = row["mean"]
    return physiology_table, distance_table, subject_row


def _display(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array)[np.isfinite(array) & (array != 0)]
    if not values.size:
        return np.zeros_like(array)
    lo, hi = np.percentile(values, (1, 99))
    return np.clip((array - lo) / max(hi - lo, 1e-6), 0, 1)


def plot_longitudinal_qc(
    baseline_flair: np.ndarray,
    followup_flair: np.ndarray,
    baseline_wmh: np.ndarray,
    followup_wmh: np.ndarray,
    classes: Mapping[str, np.ndarray],
    physiology: Mapping[str, np.ndarray],
    physiology_table: pd.DataFrame,
    distance_table: pd.DataFrame,
    output_file: str | Path,
    title: str,
) -> Path:
    """Create the complete longitudinal evidence/QC figure."""
    converting = np.asarray(classes["converting"], bool)
    slices = np.where((baseline_wmh | followup_wmh).any(axis=(0, 1)))[0]
    z = int(slices[len(slices) // 2]) if slices.size else baseline_flair.shape[2] // 2

    def rot(array: np.ndarray) -> np.ndarray:
        return np.rot90(np.asarray(array)[:, :, z])

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    panels = (
        ("Baseline FLAIR + WMH", baseline_flair, baseline_wmh, "cyan"),
        ("Baseline WMH", np.zeros_like(baseline_flair), baseline_wmh, "cyan"),
        ("Follow-up FLAIR in baseline space", followup_flair, followup_wmh, "yellow"),
        ("Follow-up WMH", np.zeros_like(followup_flair), followup_wmh, "yellow"),
    )
    for ax, (name, image, mask, color) in zip(axes[0], panels):
        ax.imshow(rot(_display(image)), cmap="gray")
        if rot(mask).any():
            ax.contourf(rot(mask), levels=[0.5, 1.5], colors=[color], alpha=0.55)
        ax.set_title(name)
        ax.axis("off")
    axes[1, 0].imshow(rot(_display(baseline_flair)), cmap="gray")
    if rot(converting).any():
        axes[1, 0].contourf(rot(converting), levels=[0.5, 1.5], colors=["#ff2d55"], alpha=0.65)
    axes[1, 0].set_title("Confident newly affected tissue")
    axes[1, 0].axis("off")
    perf_name = next(iter(physiology), None)
    if perf_name:
        axes[1, 1].imshow(rot(physiology[perf_name]), cmap="magma")
        if rot(converting).any():
            axes[1, 1].contour(rot(converting), levels=[0.5], colors=["cyan"], linewidths=1)
        axes[1, 1].set_title(f"Baseline {perf_name} / future WMH")
    else:
        axes[1, 1].text(0.5, 0.5, "No baseline physiology", ha="center", va="center")
    axes[1, 1].axis("off")
    labels = ["0–2", "2–4", "4–6", "6–10", ">10"]
    axes[1, 2].plot(range(5), distance_table.conversion_fraction, marker="o", color="#d62728")
    axes[1, 2].set_xticks(range(5), labels)
    axes[1, 2].set_xlabel("Baseline distance to WMH (mm)")
    axes[1, 2].set_ylabel("Conversion fraction")
    axes[1, 2].set_ylim(bottom=0)
    axes[1, 2].set_title("Conversion by baseline distance")
    available = physiology_table[physiology_table["map"] != "baseline_distance"]
    axes[1, 3].axis("off")
    axes[1, 3].set_title("Baseline physiology summaries")
    if not available.empty:
        lines = []
        for map_name, group in available.groupby("map", sort=False):
            medians = group.set_index("tissue")["median"]
            unit = group["units"].iloc[0]
            lines.append(
                f"{map_name} ({unit})\n  converting: {medians.get('converting', np.nan):.3g}\n  stable NAWM: {medians.get('stable_nawm', np.nan):.3g}"
            )
        axes[1, 3].text(
            0.03,
            0.95,
            "Median values\n\n" + "\n\n".join(lines) + "\n\nExploratory; no causal interpretation.",
            va="top",
            fontsize=11,
        )
    else:
        axes[1, 3].text(0.5, 0.5, "No physiology comparison", ha="center", va="center")
    fig.suptitle(title + " — exploratory associations only", fontsize=14)
    fig.tight_layout()
    output = Path(output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_registration_overlay(
    baseline_t1: np.ndarray, followup_t1: np.ndarray, output_file: str | Path, title: str
) -> Path:
    """Save representative red/green linear-registration overlays."""
    indices = [int(baseline_t1.shape[2] * fraction) for fraction in (0.35, 0.50, 0.65)]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, z in zip(axes, indices):
        fixed = np.rot90(_display(baseline_t1)[:, :, z])
        moving = np.rot90(_display(followup_t1)[:, :, z])
        rgb = np.zeros((*fixed.shape, 3), np.float32)
        rgb[..., 0] = fixed
        rgb[..., 1] = moving
        ax.imshow(rgb)
        ax.set_title(f"Axial slice {z}")
        ax.axis("off")
    fig.suptitle(title + " — baseline red / registered follow-up green")
    fig.tight_layout()
    output = Path(output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_physiology_distributions(
    classes: Mapping[str, np.ndarray],
    physiology: Mapping[str, np.ndarray],
    units: Mapping[str, str] | None,
    output_file: str | Path,
    title: str,
    max_points: int = 50000,
) -> Path | None:
    """Plot converting/stable distributions with deterministic downsampling."""
    if not physiology:
        return None
    units = units or {}
    fig, axes = plt.subplots(1, len(physiology), figsize=(5 * len(physiology), 4.5), squeeze=False)
    rng = np.random.default_rng(2026)
    for ax, (name, array) in zip(axes[0], physiology.items()):
        groups = []
        for tissue in ("converting", "stable_nawm"):
            values = np.asarray(array)[np.asarray(classes[tissue], bool)]
            values = values[np.isfinite(values)]
            if values.size > max_points:
                values = rng.choice(values, max_points, replace=False)
            groups.append(values)
        if all(values.size for values in groups):
            parts = ax.violinplot(groups, showmedians=True, showextrema=False)
            for body, color in zip(parts["bodies"], ("#e45756", "#4c78a8")):
                body.set_facecolor(color)
                body.set_alpha(0.65)
        ax.set_xticks((1, 2), ("converting", "stable NAWM"))
        ax.set_ylabel(units.get(name, "unknown units"))
        ax.set_title(name)
    fig.suptitle(title + " — voxel distributions shown descriptively; inference is subject-level")
    fig.tight_layout()
    output = Path(output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output


def _load_demographics(path: str | Path | None, subject: str) -> dict:
    if path is None:
        return {}
    frame = pd.read_csv(
        path, sep="\t" if str(path).lower().endswith(".tsv") else ",", na_values=["NA", "xx"]
    )
    key = next((x for x in ("participant_id", "subject", "subject_id") if x in frame), None)
    if key is None:
        raise ValueError("Demographics table needs participant_id, subject, or subject_id.")
    rows = frame[frame[key].astype(str) == subject]
    if rows.empty:
        return {"demographics_status": "subject_not_found"}
    output = {"demographics_status": "available"}
    for name, value in rows.iloc[0].items():
        if name == key or pd.isna(value):
            continue
        if isinstance(value, (str, int, float, np.integer, np.floating, bool)):
            output[f"demographic_{name}"] = value.item() if hasattr(value, "item") else value
    return output


def analyze_longitudinal_files(
    baseline_t1: str | Path,
    baseline_flair: str | Path,
    baseline_wmh: str | Path,
    baseline_wm: str | Path,
    followup_t1: str | Path,
    followup_flair: str | Path,
    followup_wmh: str | Path,
    output_dir: str | Path,
    subject: str,
    baseline_session: str,
    followup_session: str,
    baseline_physiology: Mapping[str, str | Path] | None = None,
    units: Mapping[str, str] | None = None,
    baseline_csf: str | Path | None = None,
    baseline_coverage: str | Path | None = None,
    demographics_file: str | Path | None = None,
    boundary_uncertainty_mm: float = 1.0,
    min_new_component_mm3: float = 3.0,
    min_conversion_voxels: int = 20,
    huge_volume_jump_percent: float = 200.0,
    force: bool = False,
    baseline_wmh_provenance: str | Path | None = None,
    followup_wmh_provenance: str | Path | None = None,
) -> dict:
    """Register follow-up to baseline and quantify baseline predictors of conversion."""
    ordered = order_sessions([baseline_session, followup_session])
    if ordered != [baseline_session, followup_session]:
        raise ValueError(f"Session order is reversed; expected baseline then follow-up: {ordered}")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    prefix = f"{subject}_from-{followup_session.removeprefix('ses-')}_to-{baseline_session.removeprefix('ses-')}"
    provenance = root / f"{prefix}_desc-longitudinal.json"
    if not force and provenance.exists():
        record = json.loads(provenance.read_text(encoding="utf-8"))
        record["resumed"] = True
        return record
    segmentation_check = {"status": "not_provided"}
    if bool(baseline_wmh_provenance) != bool(followup_wmh_provenance):
        record = {
            "subject": subject,
            "baseline_session": baseline_session,
            "followup_session": followup_session,
            "status": "fail",
            "warnings": ["both WMH provenance files are required when either is supplied"],
            "outputs": {},
            "resumed": False,
        }
        provenance.write_text(json.dumps(record, indent=2), encoding="utf-8")
        return record
    if baseline_wmh_provenance and followup_wmh_provenance:
        baseline_seg = json.loads(Path(baseline_wmh_provenance).read_text(encoding="utf-8"))
        followup_seg = json.loads(Path(followup_wmh_provenance).read_text(encoding="utf-8"))
        comparable = baseline_seg.get("backend") == followup_seg.get(
            "backend"
        ) and baseline_seg.get("parameters") == followup_seg.get("parameters")
        segmentation_check = {
            "status": "matched" if comparable else "mismatch",
            "backend": baseline_seg.get("backend"),
            "baseline_parameters": baseline_seg.get("parameters"),
            "followup_parameters": followup_seg.get("parameters"),
        }
        if not comparable:
            record = {
                "subject": subject,
                "baseline_session": baseline_session,
                "followup_session": followup_session,
                "status": "fail",
                "warnings": ["baseline and follow-up WMH segmentation backend/parameters differ"],
                "segmentation_check": segmentation_check,
                "outputs": {},
                "resumed": False,
            }
            provenance.write_text(json.dumps(record, indent=2), encoding="utf-8")
            return record
    required = {
        "baseline_t1": baseline_t1,
        "baseline_flair": baseline_flair,
        "baseline_wmh": baseline_wmh,
        "baseline_wm": baseline_wm,
        "followup_t1": followup_t1,
        "followup_flair": followup_flair,
        "followup_wmh": followup_wmh,
    }
    missing = [name for name, value in required.items() if not Path(value).exists()]
    if missing:
        record = {
            "subject": subject,
            "baseline_session": baseline_session,
            "followup_session": followup_session,
            "status": "fail",
            "warnings": [f"missing required inputs: {missing}"],
            "outputs": {},
            "resumed": False,
        }
        provenance.write_text(json.dumps(record, indent=2), encoding="utf-8")
        return record
    registered_t1 = root / f"{prefix}_space-baseline_T1w.nii.gz"
    transform = root / f"{prefix}_mode-image_xfm.tfm"
    try:
        registration = register_followup_to_baseline(
            followup_t1, baseline_t1, registered_t1, transform
        )
    except RuntimeError as exc:
        record = {
            "subject": subject,
            "baseline_session": baseline_session,
            "followup_session": followup_session,
            "status": "fail",
            "warnings": [f"registration failure: {str(exc).splitlines()[-1]}"],
            "outputs": {},
            "resumed": False,
        }
        provenance.write_text(json.dumps(record, indent=2), encoding="utf-8")
        return record
    registered_flair = _resample(
        followup_flair, baseline_t1, transform, root / f"{prefix}_space-baseline_FLAIR.nii.gz"
    )
    registered_wmh = _resample(
        followup_wmh,
        baseline_t1,
        transform,
        root / f"{prefix}_space-baseline_desc-wmh_mask.nii.gz",
        True,
        sitk.sitkUInt8,
    )
    reference = nib.load(str(baseline_t1))
    flair0 = nib.load(str(baseline_flair)).get_fdata(dtype=np.float32)
    flair1 = nib.load(str(registered_flair)).get_fdata(dtype=np.float32)
    wmh0 = nib.load(str(baseline_wmh)).get_fdata() > 0
    wmh1 = nib.load(str(registered_wmh)).get_fdata() > 0
    wm = nib.load(str(baseline_wm)).get_fdata() > 0
    if any(x.shape != reference.shape for x in (flair0, flair1, wmh0, wmh1, wm)):
        raise ValueError("All baseline inputs must share the baseline T1 grid.")
    follow_image = sitk.ReadImage(str(followup_t1))
    moving_ones = sitk.Image(follow_image.GetSize(), sitk.sitkUInt8)
    moving_ones.CopyInformation(follow_image)
    moving_ones = moving_ones + 1
    coverage_tmp = root / f"{prefix}_space-baseline_desc-registration_coverage.nii.gz"
    coverage_result = sitk.Resample(
        moving_ones,
        sitk.ReadImage(str(baseline_t1)),
        sitk.ReadTransform(str(transform)),
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8,
    )
    sitk.WriteImage(coverage_result, str(coverage_tmp))
    valid = nib.load(str(coverage_tmp)).get_fdata() > 0
    if baseline_coverage:
        valid &= nib.load(str(baseline_coverage)).get_fdata() > 0
    csf = (
        nib.load(str(baseline_csf)).get_fdata() > 0
        if baseline_csf
        else np.zeros(reference.shape, bool)
    )
    valid &= ~csf
    physiology = {}
    for name, path in (baseline_physiology or {}).items():
        image = nib.load(str(path))
        data = image.get_fdata(dtype=np.float32)
        if data.shape != reference.shape or not np.allclose(
            image.affine, reference.affine, atol=1e-3
        ):
            raise ValueError(f"Baseline {name} does not share baseline T1 space.")
        physiology[name] = data
        valid &= np.isfinite(data) & (data != 0)
    classes = classify_longitudinal_change(
        wmh0,
        wmh1,
        wm,
        valid,
        reference.header.get_zooms()[:3],
        boundary_uncertainty_mm,
        min_new_component_mm3,
    )
    physiology_table, distance_table, subject_row = summarize_longitudinal(
        classes,
        physiology,
        reference.header.get_zooms()[:3],
        subject,
        baseline_session,
        followup_session,
        units,
        _load_demographics(demographics_file, subject),
        reference.affine,
    )
    physiology_csv = root / f"{prefix}_desc-longitudinal_physiology.csv"
    physiology_table.to_csv(physiology_csv, index=False)
    distance_csv = root / f"{prefix}_desc-conversion_by_distance.csv"
    distance_table.to_csv(distance_csv, index=False)
    subject_csv = root / f"{prefix}_desc-longitudinal_subject.csv"
    pd.DataFrame([subject_row]).to_csv(subject_csv, index=False)
    outputs = {
        "registered_t1": registered_t1,
        "registered_flair": registered_flair,
        "registered_wmh": registered_wmh,
        "registration_transform": transform,
        "registration_coverage": coverage_tmp,
        "physiology_csv": physiology_csv,
        "distance_csv": distance_csv,
        "subject_csv": subject_csv,
    }
    for name in (
        "persistent",
        "converting",
        "stable_nawm",
        "resolved",
        "unsuitable",
        "boundary_uncertainty",
    ):
        outputs[name] = _save(classes[name], reference, root / f"{prefix}_desc-{name}_mask.nii.gz")
    outputs["baseline_distance"] = _save(
        classes["baseline_distance_mm"],
        reference,
        root / f"{prefix}_desc-baseline_wmh_distance-mm.nii.gz",
        np.float32,
    )
    outputs["figure"] = plot_longitudinal_qc(
        flair0,
        flair1,
        wmh0,
        wmh1,
        classes,
        physiology,
        physiology_table,
        distance_table,
        root / f"{prefix}_desc-longitudinal_qc.png",
        f"{subject}: {baseline_session} → {followup_session}",
    )
    outputs["registration_overlay"] = plot_registration_overlay(
        reference.get_fdata(dtype=np.float32),
        nib.load(str(registered_t1)).get_fdata(dtype=np.float32),
        root / f"{prefix}_desc-registration_overlay.png",
        f"{subject}: {baseline_session} → {followup_session}",
    )
    distribution_figure = plot_physiology_distributions(
        classes,
        physiology,
        units,
        root / f"{prefix}_desc-physiology_distributions.png",
        f"{subject}: baseline physiology by later tissue class",
    )
    if distribution_figure:
        outputs["physiology_distributions"] = distribution_figure
    warnings = []
    if segmentation_check.get("backend") == "fallback":
        warnings.append(
            "WMH masks use the synthetic/testing fallback and are not biologically validated"
        )
    if registration["foreground_overlap"] < 0.70:
        warnings.append(
            f"low registration foreground overlap: {registration['foreground_overlap']:.3f}"
        )
    zoom0 = np.asarray(reference.header.get_zooms()[:3])
    zoom1 = np.asarray(nib.load(str(followup_t1)).header.get_zooms()[:3])
    if np.max(np.maximum(zoom0 / zoom1, zoom1 / zoom0)) > 2:
        warnings.append(
            f"large baseline/follow-up resolution mismatch: {tuple(zoom0)} vs {tuple(zoom1)}"
        )
    common_wm = float((valid & wm).sum() / wm.sum()) if wm.any() else 0.0
    if common_wm < 0.70:
        warnings.append(f"low common baseline-WM coverage: {common_wm:.1%}")
    if (
        np.isfinite(subject_row["percent_wmh_change"])
        and abs(subject_row["percent_wmh_change"]) > huge_volume_jump_percent
    ):
        warnings.append(
            f"implausibly large WMH volume change: {subject_row['percent_wmh_change']:.1f}%"
        )
    if subject_row["converting_voxels"] < min_conversion_voxels:
        warnings.append(
            f"insufficient conversion voxels: {subject_row['converting_voxels']} < {min_conversion_voxels}"
        )
    status = (
        "fail" if registration["foreground_overlap"] < 0.30 else ("warn" if warnings else "pass")
    )
    record = {
        "subject": subject,
        "baseline_session": baseline_session,
        "followup_session": followup_session,
        "status": status,
        "warnings": warnings,
        "registration": registration,
        "segmentation_check": segmentation_check,
        "parameters": {
            "registration": "phase-correlation initialization when grids match; SimpleITK rigid then guarded affine Mattes mutual information; no nonlinear warp",
            "mask_interpolation": "nearest neighbor",
            "image_interpolation": "linear",
            "boundary_uncertainty_mm": boundary_uncertainty_mm,
            "min_new_component_mm3": min_new_component_mm3,
            "min_conversion_voxels": min_conversion_voxels,
        },
        "inputs": {
            "baseline_t1": str(baseline_t1),
            "baseline_flair": str(baseline_flair),
            "baseline_wmh": str(baseline_wmh),
            "baseline_wm": str(baseline_wm),
            "followup_t1": str(followup_t1),
            "followup_flair": str(followup_flair),
            "followup_wmh": str(followup_wmh),
            "baseline_physiology": {k: str(v) for k, v in (baseline_physiology or {}).items()},
        },
        "outcomes": subject_row,
        "outputs": {k: str(v) for k, v in outputs.items()},
        "resumed": False,
    }
    provenance.write_text(
        json.dumps(
            record,
            indent=2,
            default=lambda value: value.item() if hasattr(value, "item") else str(value),
        ),
        encoding="utf-8",
    )
    return record


def qc_longitudinal(provenance_file: str | Path) -> dict:
    """Validate a completed analysis and return stored QC without recomputation."""
    path = Path(provenance_file)
    if not path.exists():
        return {"status": "fail", "warnings": ["missing longitudinal provenance"]}
    record = json.loads(path.read_text(encoding="utf-8"))
    missing = [
        name for name, output in record.get("outputs", {}).items() if not Path(output).exists()
    ]
    if missing:
        record["status"] = "fail"
        record.setdefault("warnings", []).append(f"missing outputs: {missing}")
    record["qc_checked"] = True
    return record
