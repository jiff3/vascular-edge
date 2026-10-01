import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from vascular_edge.perilesional import (
    analyze_files,
    assess_qc,
    build_ring_geometry,
    make_rings,
    quantify_regions,
)


def test_rings_are_non_overlapping_and_exclude_lesion():
    lesion = np.zeros((9, 9, 9), dtype=bool)
    lesion[4, 4, 4] = True
    rings = make_rings(lesion, (1.0, 1.0, 1.0), ring_width_mm=1, max_distance_mm=2)
    assert rings[4, 4, 4] == 0
    assert rings[5, 4, 4] == 1
    assert rings[6, 4, 4] == 2
    assert rings.max() == 2


def test_anisotropic_physical_distance_and_exact_bins():
    lesion = np.zeros((15, 15, 15), bool)
    lesion[7, 7, 7] = True
    geometry = build_ring_geometry(
        lesion, np.ones_like(lesion), (1.0, 2.0, 3.0), min_lesion_volume_mm3=0
    )
    assert geometry.distance_mm[8, 7, 7] == 1
    assert geometry.distance_mm[7, 8, 7] == 2
    assert geometry.distance_mm[7, 7, 8] == 3
    assert geometry.regions[8, 7, 7] == 1  # (0, 2]
    assert geometry.regions[7, 8, 7] == 1
    assert geometry.regions[7, 7, 8] == 2  # (2, 4]
    assert geometry.regions[7, 7, 7] == 0


def test_masks_coverage_csf_and_nearest_lesion_partition():
    lesion = np.zeros((21, 9, 9), bool)
    lesion[5, 4, 4] = lesion[15, 4, 4] = True
    wm = np.ones_like(lesion)
    csf = np.zeros_like(lesion)
    coverage = np.ones_like(lesion)
    csf[6, 4, 4] = True
    coverage[14, 4, 4] = False
    geometry = build_ring_geometry(lesion, wm, (1, 1, 1), csf, coverage, min_lesion_volume_mm3=0)
    assert geometry.regions[6, 4, 4] == -1
    assert geometry.regions[14, 4, 4] == -1
    assert geometry.nearest_lesion[9, 4, 4] == 1
    assert geometry.nearest_lesion[11, 4, 4] == 2
    assert np.all(geometry.nearest_lesion[geometry.regions >= 0] > 0)


def test_tiny_components_lesion_rows_and_statistics():
    lesion = np.zeros((25, 25, 25), bool)
    lesion[11:14, 11:14, 11:14] = True
    lesion[2, 2, 2] = True
    geometry = build_ring_geometry(lesion, np.ones_like(lesion), (1, 1, 1), min_lesion_volume_mm3=3)
    assert len(geometry.retained_lesions) == 1
    assert len(geometry.excluded_lesions) == 1
    physiology = geometry.distance_mm.copy()
    table = quantify_regions(geometry, {"perfusion": physiology}, (1, 1, 1), "sub-test", "ses-1")
    row = table[(table.scope == "subject") & (table.region == "0_2_mm")].iloc[0]
    values = physiology[geometry.regions == 1]
    assert row.region_voxels == values.size
    assert np.isclose(row["median"], np.median(values))
    assert {"mean", "std", "iqr", "p05", "p95", "centroid_i"}.issubset(table.columns)


def test_qc_boundary_and_empty_coverage():
    lesion = np.zeros((9, 9, 9), bool)
    lesion[0, 4, 4] = True
    wm = np.ones_like(lesion)
    coverage = np.zeros_like(lesion)
    geometry = build_ring_geometry(
        lesion, wm, (1, 1, 1), coverage_mask=coverage, min_lesion_volume_mm3=0
    )
    qc = assess_qc(geometry, lesion, wm, None, coverage)
    assert qc["status"] == "fail"
    assert any("boundary" in warning for warning in qc["warnings"])
    assert any("no valid overlap" in warning for warning in qc["warnings"])


def test_file_pipeline_writes_tidy_outputs_and_figure(tmp_path):
    shape = (31, 31, 21)
    affine = np.diag([1.0, 1.0, 2.0, 1.0])
    grid = np.indices(shape)
    radius = np.sqrt(((grid - np.array([15, 15, 10])[:, None, None, None]) ** 2).sum(axis=0))
    flair = np.exp(-((radius / 12) ** 2)).astype("float32")
    wmh = radius <= 2.5
    wm = radius <= 13
    csf = radius < 1
    perfusion = (40 + 2 * radius).astype("float32")
    perfusion[~wm] = 0
    paths = {}
    for name, data in {
        "flair": flair,
        "wmh": wmh,
        "wm": wm,
        "csf": csf,
        "perfusion": perfusion,
    }.items():
        path = tmp_path / f"{name}.nii.gz"
        nib.save(nib.Nifti1Image(data.astype("float32"), affine), path)
        paths[name] = path
    record = analyze_files(
        paths["wmh"],
        paths["wm"],
        paths["flair"],
        tmp_path / "out",
        "sub-test",
        "ses-1",
        paths["csf"],
        {"perfusion": paths["perfusion"]},
        {"perfusion": "relative"},
        min_lesion_volume_mm3=1,
        force=True,
    )
    assert Path(record["outputs"]["figure"]).exists()
    frame = pd.read_csv(record["outputs"]["csv"])
    assert set(frame.scope) == {"subject", "lesion"}
    assert set(frame["map"]) == {"perfusion"}
    qc = json.loads((tmp_path / "out" / "sub-test_ses-1_desc-perilesional_qc.json").read_text())
    assert qc["parameters"]["overlap_policy"].startswith("nearest")
