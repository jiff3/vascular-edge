import json
from pathlib import Path

import nibabel as nib
import numpy as np

from vascular_edge.cvr import percent_bold_response, process_cvr
from vascular_edge.perfusion import (
    assess_asl,
    pair_control_label,
    process_perfusion,
    quantify_pcasl,
)
from vascular_edge.physiology_qc import qc_physiology


def _structural_fixture(root: Path, shape=(32, 32, 32)) -> tuple[Path, Path, Path, Path]:
    xyz = np.indices(shape)
    center = (np.asarray(shape) - 1)[:, None, None, None] / 2
    radius = np.sqrt(((xyz - center) ** 2).sum(0))
    brain = radius < 12
    t1 = brain.astype(np.float32) * 100
    flair = brain.astype(np.float32) * 60
    gm = (brain & (radius >= 7)).astype(np.uint8)
    wm = (radius < 7).astype(np.uint8)
    outputs = [root / name for name in ("t1.nii.gz", "flair.nii.gz", "gm.nii.gz", "wm.nii.gz")]
    for array, path in zip((t1, flair, gm, wm), outputs):
        nib.save(nib.Nifti1Image(array, np.eye(4)), path)
    return tuple(outputs)


def test_dlbs_style_asl_is_relative_when_m0_absent():
    metadata = {
        "ArterialSpinLabelingType": "PCASL",
        "PostLabelingDelay": 1.525,
        "LabelingDuration": 1.65,
        "M0Type": "Absent",
    }
    assessment = assess_asl((8, 8, 8, 4), metadata, ["label", "control", "label", "control"])
    assert not assessment.quantitative_cbf_supported
    assert assessment.units == "% mean control signal"


def test_derived_map_and_calibrated_pair_assessments():
    derived = assess_asl((8, 8, 8), {"Units": "mL/100g/min"}, None)
    assert derived.quantitative_cbf_supported and derived.organization == "derived_perfusion_map"
    calibrated = assess_asl(
        (8, 8, 8, 2),
        {
            "ArterialSpinLabelingType": "PCASL",
            "PostLabelingDelay": 1.5,
            "LabelingDuration": 1.8,
            "M0Type": "Included",
        },
        ["control", "label"],
    )
    assert calibrated.quantitative_cbf_supported
    unsupported = assess_asl((8, 8, 8, 3), {}, ["control"])
    assert unsupported.organization == "unsupported"


def test_control_label_pairing_handles_both_orders():
    data = np.zeros((2, 2, 2, 4), np.float32)
    data[..., 0] = 90
    data[..., 1] = 100
    data[..., 2] = 100
    data[..., 3] = 90
    differences, controls, skipped = pair_control_label(
        data, ["label", "control", "control", "label"]
    )
    assert np.allclose(differences, 10) and np.allclose(controls, 100) and not skipped


def test_pcasl_equation_returns_physical_units_scale():
    cbf = quantify_pcasl(np.array([10], np.float32), np.array([1000], np.float32), 1.65, 1.525)
    assert 20 < cbf[0] < 100


def test_relative_perfusion_pipeline_and_qc(tmp_path: Path):
    t1, flair, gm, wm = _structural_fixture(tmp_path)
    brain = nib.load(t1).get_fdata() > 0
    volumes = []
    for _ in range(4):
        volumes.extend([brain * 95, brain * 100])  # label then control
    asl = np.stack(volumes, axis=-1).astype(np.float32)
    asl_file = tmp_path / "asl.nii.gz"
    nib.save(nib.Nifti1Image(asl, np.eye(4)), asl_file)
    metadata_file = tmp_path / "asl.json"
    metadata_file.write_text(
        json.dumps(
            {
                "ArterialSpinLabelingType": "PCASL",
                "PostLabelingDelay": 1.525,
                "LabelingDuration": 1.65,
                "M0Type": "Absent",
            }
        ),
        encoding="utf-8",
    )
    context = tmp_path / "aslcontext.tsv"
    context.write_text("volume_type\n" + "\n".join(["label", "control"] * 4), encoding="utf-8")
    record = process_perfusion(
        asl_file,
        metadata_file,
        t1,
        gm,
        wm,
        tmp_path / "perf",
        "sub-syn",
        "ses-01",
        context,
        motion_correct=False,
    )
    assert record["assessment"]["units"] == "% mean control signal"
    perf = nib.load(record["outputs"]["t1_map"]).get_fdata()
    assert 4 < np.median(perf[brain]) < 6
    qc = qc_physiology(
        t1,
        flair,
        record["outputs"]["t1_map"],
        gm,
        wm,
        tmp_path / "qc",
        "sub-syn",
        "ses-01",
        record["assessment"]["units"],
        asl_file,
    )
    assert Path(qc["figure"]).exists()


def test_unsupported_asl_and_cvr_return_records(tmp_path: Path):
    asl = tmp_path / "asl.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((8, 8, 8, 3), np.float32), np.eye(4)), asl)
    metadata = tmp_path / "meta.json"
    metadata.write_text("{}", encoding="utf-8")
    record = process_perfusion(
        asl,
        metadata,
        "missing-t1",
        "missing-gm",
        "missing-wm",
        tmp_path / "perf",
        "sub-x",
        "ses-x",
        context_file=None,
        motion_correct=False,
    )
    assert record["status"] == "unsupported"
    cvr = process_cvr("missing-bold", metadata, "missing-t1", tmp_path / "cvr", "sub-x", "ses-x")
    assert cvr["status"] == "unsupported" and "cannot be estimated" in cvr["warnings"][0]


def test_percent_bold_response_known_effect():
    regressor = np.array([0, 0, 1, 1], np.float32)
    bold = np.ones((3, 3, 3, 4), np.float32) * 100
    bold[..., 2:] = 102
    response = percent_bold_response(bold, regressor)
    assert np.allclose(response, 2 / 101 * 100, atol=0.05)
