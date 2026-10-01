import json

import numpy as np
import pandas as pd
import pytest

from vascular_edge.longitudinal import (
    analyze_longitudinal_files,
    choose_longitudinal_pair,
    classify_longitudinal_change,
    lesion_change,
    order_sessions,
    summarize_longitudinal,
)


def test_lesion_change_labels_new_voxels():
    initial = np.zeros((2, 2), bool)
    initial[0, 0] = True
    later = initial.copy()
    later[1, 1] = True
    result = lesion_change(initial, later)
    assert result["new"].sum() == 1
    assert result["persistent"].sum() == 1


def test_lesion_change_requires_matching_grids():
    with pytest.raises(ValueError):
        lesion_change(np.zeros((2, 2)), np.zeros((3, 3)))


def test_dlbs_wave_order_and_metadata_order():
    assert order_sessions(["ses-wave3", "ses-wave1", "ses-wave2"]) == [
        "ses-wave1",
        "ses-wave2",
        "ses-wave3",
    ]
    table = pd.DataFrame({"session_id": ["visit-b", "visit-a"], "days_since_baseline": [100, 0]})
    assert choose_longitudinal_pair(["visit-b", "visit-a"], table) == ("visit-a", "visit-b")
    with pytest.raises(ValueError, match="ambiguous"):
        order_sessions(["first", "second"])


def test_known_expansion_respects_boundary_margin_and_component_threshold():
    shape = (17, 17, 17)
    grid = np.indices(shape)
    radius = np.sqrt(((grid - 8) ** 2).sum(axis=0))
    baseline = radius <= 2
    followup = radius <= 4
    followup[1, 1, 1] = True  # one-voxel false positive
    classes = classify_longitudinal_change(
        baseline,
        followup,
        np.ones(shape, bool),
        np.ones(shape, bool),
        (1, 1, 2),
        boundary_uncertainty_mm=1,
        min_new_component_mm3=4,
    )
    expected = followup & (classes["baseline_distance_mm"] > 1) & ~baseline
    expected[1, 1, 1] = False
    assert np.array_equal(classes["converting"], expected)
    assert not classes["stable_nawm"][followup].any()
    assert classes["boundary_uncertainty"].any()


def test_subject_level_summary_does_not_emit_voxel_rows():
    shape = (15, 15, 15)
    baseline = np.zeros(shape, bool)
    baseline[7, 7, 7] = True
    followup = baseline.copy()
    followup[9:11, 7:9, 7:9] = True
    classes = classify_longitudinal_change(
        baseline,
        followup,
        np.ones(shape, bool),
        np.ones(shape, bool),
        (1, 1, 1),
        boundary_uncertainty_mm=0,
        min_new_component_mm3=1,
    )
    perfusion = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    physiology, distances, subject = summarize_longitudinal(
        classes, {"perfusion": perfusion}, (1, 1, 1), "sub-1", "ses-wave1", "ses-wave2"
    )
    assert len(physiology) == 4  # 2 tissues x (physiology + distance)
    assert len(distances) == 5
    assert subject["converting_voxels"] == int(classes["converting"].sum())
    assert "perfusion_converting_median" in subject


def test_anatomical_volumes_do_not_depend_on_physiology_coverage():
    shape = (9, 9, 9)
    baseline = np.zeros(shape, bool)
    baseline[3:5, 3:5, 3:5] = True
    followup = baseline.copy()
    followup[5:7, 3:5, 3:5] = True
    coverage = np.zeros(shape, bool)
    coverage[:5] = True
    classes = classify_longitudinal_change(
        baseline,
        followup,
        np.ones(shape, bool),
        coverage,
        (1, 1, 1),
        boundary_uncertainty_mm=0,
        min_new_component_mm3=1,
    )
    _, _, subject = summarize_longitudinal(
        classes, {}, (1, 1, 1), "sub-1", "ses-wave1", "ses-wave2"
    )
    assert np.isclose(subject["baseline_wmh_ml"], baseline.sum() / 1000)
    assert np.isclose(subject["followup_wmh_ml"], followup.sum() / 1000)


def test_missing_followup_is_a_qc_record(tmp_path):
    record = analyze_longitudinal_files(
        "missing-b.nii.gz",
        "missing-bf.nii.gz",
        "missing-bw.nii.gz",
        "missing-wm.nii.gz",
        "missing-f.nii.gz",
        "missing-ff.nii.gz",
        "missing-fw.nii.gz",
        tmp_path,
        "sub-1",
        "ses-wave1",
        "ses-wave2",
    )
    assert record["status"] == "fail"
    assert "missing required inputs" in record["warnings"][0]


def test_mismatched_wmh_provenance_fails_before_analysis(tmp_path):
    baseline = tmp_path / "baseline.json"
    followup = tmp_path / "followup.json"
    baseline.write_text(json.dumps({"backend": "truenet", "parameters": {"min_component_mm3": 3}}))
    followup.write_text(json.dumps({"backend": "truenet", "parameters": {"min_component_mm3": 5}}))
    record = analyze_longitudinal_files(
        "b1",
        "bf",
        "bw",
        "wm",
        "f1",
        "ff",
        "fw",
        tmp_path / "out",
        "sub-1",
        "ses-wave1",
        "ses-wave2",
        baseline_wmh_provenance=baseline,
        followup_wmh_provenance=followup,
    )
    assert record["status"] == "fail"
    assert "differ" in record["warnings"][0]
