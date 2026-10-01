"""Physical-distance WMH rings, lesion ownership, statistics, QC, and exports."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt, generate_binary_structure, label

DEFAULT_EDGES_MM = (0.0, 2.0, 4.0, 6.0, 10.0)
REGION_NAMES = ("wmh_core", "0_2_mm", "2_4_mm", "4_6_mm", "6_10_mm", "remote_gt_10_mm")


@dataclass(frozen=True)
class RingGeometry:
    distance_mm: np.ndarray
    regions: np.ndarray
    lesion_labels: np.ndarray
    nearest_lesion: np.ndarray
    retained_lesions: tuple[dict, ...]
    excluded_lesions: tuple[dict, ...]


def _check_shapes(reference: np.ndarray, **arrays: np.ndarray | None) -> None:
    for name, array in arrays.items():
        if array is not None and np.asarray(array).shape != reference.shape:
            raise ValueError(
                f"{name} shape {np.asarray(array).shape} does not match {reference.shape}."
            )


def build_ring_geometry(
    lesion_mask: np.ndarray,
    white_matter_mask: np.ndarray,
    voxel_sizes_mm: Sequence[float],
    csf_mask: np.ndarray | None = None,
    coverage_mask: np.ndarray | None = None,
    edges_mm: Sequence[float] = DEFAULT_EDGES_MM,
    min_lesion_volume_mm3: float = 3.0,
    connectivity: int = 2,
) -> RingGeometry:
    """Build physical-distance regions and nearest-lesion ownership.

    Outer intervals use ``(lower, upper]`` and remote NAWM uses ``> last edge``.
    When neighborhoods meet, each voxel belongs to its closest retained component.
    Exact ties follow SciPy's deterministic array-order choice and are not duplicated.
    """
    lesion = np.asarray(lesion_mask, bool)
    wm = np.asarray(white_matter_mask, bool)
    csf = np.zeros_like(lesion) if csf_mask is None else np.asarray(csf_mask, bool)
    coverage = np.ones_like(lesion) if coverage_mask is None else np.asarray(coverage_mask, bool)
    _check_shapes(lesion, white_matter=wm, csf=csf, coverage=coverage)
    spacing = tuple(float(v) for v in voxel_sizes_mm)
    if len(spacing) != lesion.ndim or any(not np.isfinite(v) or v <= 0 for v in spacing):
        raise ValueError("voxel_sizes_mm must contain one positive finite value per dimension.")
    edges = np.asarray(edges_mm, dtype=float)
    if edges.size < 2 or edges[0] != 0 or np.any(np.diff(edges) <= 0):
        raise ValueError("edges_mm must start at 0 and be strictly increasing.")

    components, count = label(
        lesion, structure=generate_binary_structure(lesion.ndim, connectivity)
    )
    voxel_mm3 = float(np.prod(spacing))
    kept = np.zeros_like(lesion)
    relabeled = np.zeros_like(components, np.int32)
    retained: list[dict] = []
    excluded: list[dict] = []
    next_id = 1
    for old_id in range(1, count + 1):
        mask = components == old_id
        voxels = int(mask.sum())
        volume = voxels * voxel_mm3
        centroid = tuple(float(x) for x in np.argwhere(mask).mean(axis=0))
        item = {
            "original_id": old_id,
            "voxel_count": voxels,
            "volume_mm3": volume,
            "centroid_ijk": centroid,
        }
        if volume < min_lesion_volume_mm3:
            excluded.append(item)
            continue
        relabeled[mask] = next_id
        kept |= mask
        retained.append({**item, "lesion_id": next_id})
        next_id += 1

    regions = np.full(lesion.shape, -1, np.int8)
    distance = np.full(lesion.shape, np.nan, np.float32)
    nearest = np.zeros(lesion.shape, np.int32)
    if kept.any():
        distance64, indices = distance_transform_edt(~kept, sampling=spacing, return_indices=True)
        distance = distance64.astype(np.float32)
        nearest = relabeled[tuple(indices)].astype(np.int32)
        valid_outer = wm & ~kept & ~csf & coverage
        regions[kept & ~csf & coverage] = 0
        for region_id, (lower, upper) in enumerate(zip(edges[:-1], edges[1:]), start=1):
            regions[valid_outer & (distance > lower) & (distance <= upper)] = region_id
        regions[valid_outer & (distance > edges[-1])] = len(edges)
    return RingGeometry(distance, regions, relabeled, nearest, tuple(retained), tuple(excluded))


def make_rings(
    lesion_mask: np.ndarray,
    voxel_sizes_mm: tuple[float, float, float],
    ring_width_mm: float = 2.0,
    max_distance_mm: float = 10.0,
    allowed_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Backward-compatible outside-ring labels (lesion and remote voxels are zero)."""
    lesion = np.asarray(lesion_mask, bool)
    allowed = np.ones_like(lesion) if allowed_mask is None else np.asarray(allowed_mask, bool)
    edges = np.arange(0.0, max_distance_mm + ring_width_mm, ring_width_mm)
    geometry = build_ring_geometry(
        lesion, allowed, voxel_sizes_mm, edges_mm=edges, min_lesion_volume_mm3=0.0
    )
    rings = geometry.regions.copy()
    rings[rings <= 0] = 0
    rings[rings == len(edges)] = 0
    return rings.astype(np.uint8)


def summarize_rings(image: np.ndarray, rings: np.ndarray) -> list[dict[str, float | int]]:
    records = []
    for ring in np.unique(rings):
        if ring == 0:
            continue
        values = np.asarray(image, np.float32)[rings == ring]
        values = values[np.isfinite(values)]
        records.append(
            {
                "ring": int(ring),
                "n_voxels": int(values.size),
                "mean": float(np.mean(values)) if values.size else float("nan"),
            }
        )
    return records


def _statistics(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, np.float32)
    values = values[np.isfinite(values)]
    if not values.size:
        return {
            "valid_voxels": 0,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "iqr": np.nan,
            "p05": np.nan,
            "p25": np.nan,
            "p75": np.nan,
            "p95": np.nan,
        }
    p05, p25, p75, p95 = np.percentile(values, (5, 25, 75, 95))
    return {
        "valid_voxels": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "iqr": float(p75 - p25),
        "p05": float(p05),
        "p25": float(p25),
        "p75": float(p75),
        "p95": float(p95),
    }


def quantify_regions(
    geometry: RingGeometry,
    maps: Mapping[str, np.ndarray],
    voxel_sizes_mm: Sequence[float],
    subject: str,
    session: str,
    units: Mapping[str, str] | None = None,
    affine: np.ndarray | None = None,
    edges_mm: Sequence[float] = DEFAULT_EDGES_MM,
) -> pd.DataFrame:
    """Return long/tidy subject- and lesion-level physiology summaries."""
    for name, array in maps.items():
        _check_shapes(geometry.regions, **{name: array})
    units = units or {}
    edges = tuple(float(x) for x in edges_mm)
    voxel_mm3 = float(np.prod(voxel_sizes_mm))
    rows: list[dict] = []
    scopes = [("subject", None, geometry.regions >= 0)]
    scopes += [
        ("lesion", item["lesion_id"], geometry.nearest_lesion == item["lesion_id"])
        for item in geometry.retained_lesions
    ]
    lesion_by_id = {item["lesion_id"]: item for item in geometry.retained_lesions}
    names = (
        ("wmh_core",)
        + tuple(f"{edges[i]:g}_{edges[i + 1]:g}_mm" for i in range(len(edges) - 1))
        + (f"remote_gt_{edges[-1]:g}_mm",)
    )
    for scope, lesion_id, owner in scopes:
        lesion_item = lesion_by_id.get(lesion_id)
        centroid = lesion_item["centroid_ijk"] if lesion_item else (np.nan,) * 3
        xyz = (
            tuple(nib.affines.apply_affine(affine, centroid))
            if affine is not None and lesion_item
            else (np.nan,) * 3
        )
        for region_id, region_name in enumerate(names):
            region_mask = (geometry.regions == region_id) & owner
            if region_id == 0:
                lower, upper = 0.0, 0.0
            elif region_id < len(edges):
                lower, upper = edges[region_id - 1], edges[region_id]
            else:
                lower, upper = edges[-1], np.inf
            base = {
                "subject": subject,
                "session": session,
                "scope": scope,
                "lesion_id": lesion_id,
                "region": region_name,
                "region_order": region_id,
                "distance_lower_mm": lower,
                "distance_upper_mm": upper,
                "region_voxels": int(region_mask.sum()),
                "region_volume_ml": float(region_mask.sum() * voxel_mm3 / 1000),
                "lesion_volume_ml": float(lesion_item["volume_mm3"] / 1000)
                if lesion_item
                else float(sum(x["volume_mm3"] for x in geometry.retained_lesions) / 1000),
                "centroid_i": centroid[0],
                "centroid_j": centroid[1],
                "centroid_k": centroid[2],
                "centroid_x_mm": xyz[0],
                "centroid_y_mm": xyz[1],
                "centroid_z_mm": xyz[2],
            }
            if maps:
                for map_name, array in maps.items():
                    rows.append(
                        {
                            **base,
                            "map": map_name,
                            "units": units.get(map_name, "unknown"),
                            **_statistics(np.asarray(array)[region_mask]),
                        }
                    )
            else:
                rows.append(
                    {
                        **base,
                        "map": "geometry",
                        "units": "not_applicable",
                        **_statistics(np.array([])),
                    }
                )
    return pd.DataFrame(rows)


def assess_qc(
    geometry: RingGeometry,
    lesion_mask: np.ndarray,
    wm_mask: np.ndarray,
    csf_mask: np.ndarray | None,
    coverage_mask: np.ndarray,
    min_ring_voxels: int = 20,
) -> dict:
    """Report, but never silently filter, questionable geometry and coverage."""
    lesion = np.asarray(lesion_mask, bool)
    wm = np.asarray(wm_mask, bool)
    coverage = np.asarray(coverage_mask, bool)
    warnings: list[str] = []
    retained = geometry.lesion_labels > 0
    boundary = np.zeros_like(retained)
    for axis in range(retained.ndim):
        start = [slice(None)] * retained.ndim
        start[axis] = 0
        boundary[tuple(start)] = True
        end = [slice(None)] * retained.ndim
        end[axis] = -1
        boundary[tuple(end)] = True
    boundary_ids = sorted(
        int(x) for x in np.unique(geometry.lesion_labels[retained & boundary]) if x
    )
    if boundary_ids:
        warnings.append(f"lesions touch image boundary: {boundary_ids}")
    wm_fraction = float(wm.mean())
    if wm_fraction < 0.01 or wm_fraction > 0.70:
        warnings.append(f"suspicious white-matter mask fraction: {wm_fraction:.3f}")
    outside_wm = float((retained & ~wm).sum() / retained.sum()) if retained.any() else 0.0
    if outside_wm > 0.25:
        warnings.append(f"{outside_wm:.1%} of retained WMH voxels lie outside the WM mask")
    csf_overlap = (
        float((retained & np.asarray(csf_mask, bool)).sum() / retained.sum())
        if csf_mask is not None and retained.any()
        else 0.0
    )
    if csf_overlap:
        warnings.append(f"{csf_overlap:.1%} of retained WMH voxels overlap CSF and are excluded")
    valid_tissue = wm & ~lesion
    coverage_fraction = (
        float((coverage & valid_tissue).sum() / valid_tissue.sum()) if valid_tissue.any() else 0.0
    )
    if coverage_fraction == 0:
        warnings.append("physiology has no valid overlap with non-lesional white matter")
    elif coverage_fraction < 0.50:
        warnings.append(f"low physiology/white-matter coverage: {coverage_fraction:.1%}")
    counts = {REGION_NAMES[i]: int((geometry.regions == i).sum()) for i in range(len(REGION_NAMES))}
    for name, count in counts.items():
        if count == 0:
            warnings.append(f"empty region: {name}")
        elif count < min_ring_voxels:
            warnings.append(f"insufficient voxels in {name}: {count} < {min_ring_voxels}")
    if not geometry.retained_lesions:
        warnings.append("no lesions meet the minimum volume criterion")
    status = (
        "fail"
        if not geometry.retained_lesions or coverage_fraction == 0
        else ("warn" if warnings else "pass")
    )
    return {
        "status": status,
        "warnings": warnings,
        "metrics": {
            "retained_lesions": len(geometry.retained_lesions),
            "excluded_tiny_lesions": len(geometry.excluded_lesions),
            "boundary_lesion_ids": boundary_ids,
            "wm_fraction": wm_fraction,
            "wmh_outside_wm_fraction": outside_wm,
            "wmh_csf_overlap_fraction": csf_overlap,
            "physiology_wm_coverage_fraction": coverage_fraction,
            "region_voxels": counts,
        },
    }


def _norm(array: np.ndarray) -> np.ndarray:
    finite = np.asarray(array)[np.isfinite(array)]
    if not finite.size:
        return np.zeros_like(array, dtype=np.float32)
    lo, hi = np.percentile(finite, (1, 99))
    return np.clip((array - lo) / max(hi - lo, 1e-6), 0, 1)


def plot_subject(
    flair: np.ndarray,
    geometry: RingGeometry,
    table: pd.DataFrame,
    output: str | Path,
    maps: Mapping[str, np.ndarray] | None = None,
    subject: str = "",
    session: str = "",
) -> Path:
    """Plot anatomy/maps with identical contours plus an unbiased distance profile."""
    maps = maps or {}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    lesion_slices = np.where((geometry.lesion_labels > 0).any(axis=(0, 1)))[0]
    z = int(lesion_slices[len(lesion_slices) // 2]) if lesion_slices.size else flair.shape[2] // 2
    panels = [("FLAIR", flair), *[(name, value) for name, value in maps.items()]]
    ncols = max(3, len(panels))
    fig, axes = plt.subplots(2, ncols, figsize=(5 * ncols, 8), squeeze=False)
    valid_slice = np.rot90(geometry.regions[:, :, z] >= 0)
    distance_slice = np.rot90(geometry.distance_mm[:, :, z])
    core_slice = np.rot90(geometry.lesion_labels[:, :, z] > 0)
    colors = ("#00ffff", "#ffcc00", "#ff8c00", "#e31a1c", "#984ea3")
    for ax, (title, data) in zip(axes[0], panels):
        ax.imshow(
            np.rot90(_norm(np.asarray(data)[:, :, z])), cmap="gray" if title == "FLAIR" else "magma"
        )
        if core_slice.any():
            ax.contour(core_slice, levels=[0.5], colors=[colors[0]], linewidths=1.0)
        for threshold, color in zip((2, 4, 6, 10), colors[1:]):
            neighborhood = valid_slice & (distance_slice <= threshold)
            if neighborhood.any() and (~neighborhood).any():
                ax.contour(neighborhood, levels=[0.5], colors=[color], linewidths=0.8)
        ax.set_title(f"{title} + WMH distance contours")
        ax.axis("off")
    for ax in axes[0, len(panels) :]:
        ax.axis("off")
    current = table[
        (table.scope == "subject") & (table["map"] != "geometry") & (table.region != "wmh_core")
    ]
    for profile_ax, (map_name, group) in zip(axes[1], current.groupby("map")):
        group = group.sort_values("region_order")
        profile_ax.plot(group.region_order, group["median"], marker="o", color="#4c78a8")
        profile_ax.set_xticks(range(1, 6), ["0–2", "2–4", "4–6", "6–10", ">10"])
        profile_ax.set_xlabel("Distance from WMH (mm)")
        unit = str(group["units"].iloc[0])
        profile_ax.set_ylabel(f"Median ({unit})")
        profile_ax.set_title(f"{map_name} profile (descriptive)")
    used_profiles = current["map"].nunique()
    qc_ax = axes[1, ncols - 1]
    qc_ax.axis("off")
    qc_ax.text(
        0,
        0.95,
        f"{subject} {session}\nRetained lesions: {len(geometry.retained_lesions)}\nExcluded tiny lesions: {len(geometry.excluded_lesions)}\n\nContours show physical-distance bins.\nNo trend direction is assumed.",
        va="top",
        fontsize=11,
    )
    for index in range(used_profiles, ncols - 1):
        axes[1, index].axis("off")
    fig.suptitle("Perilesional vascular gradient QC", fontsize=14)
    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_cohort(table: pd.DataFrame, output_dir: str | Path) -> list[Path]:
    """Create cohort profiles, distributions, and burden plots without trend assumptions."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    data = table[(table.scope == "subject") & (table["map"] != "geometry")].copy()
    if data.empty:
        raise ValueError("No subject-level physiology rows were found.")
    for map_name, group in data.groupby("map"):
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(map_name))
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        outer = group[group.region != "wmh_core"]
        for _, person in outer.groupby(["subject", "session"]):
            person = person.sort_values("region_order")
            axes[0].plot(person.region_order, person["median"], color="#4c78a8", alpha=0.25)
        med = outer.groupby("region_order")["median"].median()
        axes[0].plot(med.index, med, color="black", marker="o", linewidth=2, label="cohort median")
        axes[0].set_xticks(range(1, 6), ["0–2", "2–4", "4–6", "6–10", ">10"])
        axes[0].set_title("Profiles (no fitted direction)")
        axes[0].set_xlabel("Distance (mm)")
        axes[0].set_ylabel(f"Median {map_name}")
        axes[0].legend()
        values = [
            outer.loc[outer.region_order == i, "median"].dropna().to_numpy() for i in range(1, 6)
        ]
        axes[1].boxplot(values, tick_labels=["0–2", "2–4", "4–6", "6–10", ">10"])
        axes[1].set_title("Subject distributions")
        axes[1].set_xlabel("Distance (mm)")
        remote = outer[outer.region.str.startswith("remote_gt")]
        axes[2].scatter(remote.lesion_volume_ml, remote["median"], alpha=0.7)
        axes[2].set_xlabel("WMH burden (mL)")
        axes[2].set_ylabel(f"Remote median {map_name}")
        axes[2].set_title("Burden vs physiology")
        fig.suptitle(f"Cohort perilesional summaries: {map_name}")
        fig.tight_layout()
        path = root / f"cohort_{safe}_perilesional.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def analyze_files(
    wmh_file: str | Path,
    wm_file: str | Path,
    flair_file: str | Path,
    output_dir: str | Path,
    subject: str,
    session: str,
    csf_file: str | Path | None = None,
    physiology_files: Mapping[str, str | Path] | None = None,
    units: Mapping[str, str] | None = None,
    coverage_file: str | Path | None = None,
    edges_mm: Sequence[float] = DEFAULT_EDGES_MM,
    min_lesion_volume_mm3: float = 3.0,
    min_ring_voxels: int = 20,
    zero_is_missing: bool = True,
    parquet: bool = False,
    force: bool = False,
) -> dict:
    """File-oriented resumable analysis in one shared structural grid."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    prefix = f"{subject}_{session}"
    csv_file = root / f"{prefix}_desc-perilesional_features.csv"
    qc_file = root / f"{prefix}_desc-perilesional_qc.json"
    if not force and csv_file.exists() and qc_file.exists():
        record = json.loads(qc_file.read_text(encoding="utf-8"))
        record["resumed"] = True
        return record
    ref = nib.load(str(flair_file))
    flair = ref.get_fdata(dtype=np.float32)
    wmh = nib.load(str(wmh_file)).get_fdata() > 0
    wm = nib.load(str(wm_file)).get_fdata() > 0
    csf = nib.load(str(csf_file)).get_fdata() > 0 if csf_file else None
    _check_shapes(flair, wmh=wmh, wm=wm, csf=csf)
    maps: dict[str, np.ndarray] = {}
    for name, path in (physiology_files or {}).items():
        image = nib.load(str(path))
        array = image.get_fdata(dtype=np.float32)
        _check_shapes(flair, **{name: array})
        if not np.allclose(image.affine, ref.affine, atol=1e-3):
            raise ValueError(f"{name} affine does not match FLAIR space.")
        maps[name] = array
    if coverage_file:
        coverage = nib.load(str(coverage_file)).get_fdata() > 0
        _check_shapes(flair, coverage=coverage)
    elif maps:
        coverage = np.logical_and.reduce(
            [
                np.isfinite(value) & ((value != 0) if zero_is_missing else True)
                for value in maps.values()
            ]
        )
    else:
        coverage = np.ones(flair.shape, bool)
    geometry = build_ring_geometry(
        wmh, wm, ref.header.get_zooms()[:3], csf, coverage, edges_mm, min_lesion_volume_mm3
    )
    table = quantify_regions(
        geometry, maps, ref.header.get_zooms()[:3], subject, session, units, ref.affine, edges_mm
    )
    table.to_csv(csv_file, index=False)
    parquet_file = None
    if parquet:
        parquet_file = root / f"{prefix}_desc-perilesional_features.parquet"
        try:
            table.to_parquet(parquet_file, index=False)
        except ImportError as exc:
            raise RuntimeError(
                "Parquet export requires: pip install 'vascular-edge[parquet]'"
            ) from exc
    header = ref.header.copy()
    header.set_data_dtype(np.float32)
    distance_file = root / f"{prefix}_desc-wmh_distance-mm.nii.gz"
    nib.save(nib.Nifti1Image(geometry.distance_mm, ref.affine, header), distance_file)
    region_file = root / f"{prefix}_desc-perilesional_regions.nii.gz"
    nib.save(nib.Nifti1Image(geometry.regions, ref.affine), region_file)
    labels_file = root / f"{prefix}_desc-wmh_components.nii.gz"
    nib.save(nib.Nifti1Image(geometry.lesion_labels, ref.affine), labels_file)
    figure_file = plot_subject(
        flair, geometry, table, root / f"{prefix}_desc-perilesional_qc.png", maps, subject, session
    )
    qc = assess_qc(geometry, wmh, wm, csf, coverage, min_ring_voxels)
    record = {
        "subject": subject,
        "session": session,
        **qc,
        "parameters": {
            "edges_mm": list(edges_mm),
            "interval_convention": "(lower, upper] outside WMH; remote > final edge",
            "min_lesion_volume_mm3": min_lesion_volume_mm3,
            "zero_is_missing": zero_is_missing,
            "overlap_policy": "nearest retained lesion; deterministic EDT tie resolution",
        },
        "outputs": {
            "csv": str(csv_file),
            "parquet": str(parquet_file) if parquet_file else None,
            "distance_map": str(distance_file),
            "region_map": str(region_file),
            "lesion_labels": str(labels_file),
            "figure": str(figure_file),
        },
        "retained_lesions": list(geometry.retained_lesions),
        "excluded_lesions": list(geometry.excluded_lesions),
        "resumed": False,
    }
    qc_file.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record
