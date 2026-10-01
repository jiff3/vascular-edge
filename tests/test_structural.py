from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from nibabel.orientations import aff2axcodes

from vascular_edge.preprocessing import canonicalize_nifti, preprocess_structural, validate_image
from vascular_edge.qc import structural_qc
from vascular_edge.segmentation import (
    WMHBackendUnavailable,
    postprocess_wmh,
    run_truenet,
    segment_wmh_files,
)


def synthetic_pair(root: Path, shape=(40, 40, 40)) -> tuple[Path, Path]:
    coords = np.indices(shape, dtype=np.float32)
    center = (np.asarray(shape, np.float32) - 1)[:, None, None, None] / 2
    radius = np.sqrt(((coords - center) ** 2).sum(axis=0))
    brain = radius < 15
    t1 = np.zeros(shape, np.float32)
    t1[brain] = 45
    t1[radius < 12] = 80
    t1[radius < 8] = 120
    t1 *= np.linspace(0.85, 1.15, shape[0], dtype=np.float32)[:, None, None]
    flair = np.zeros(shape, np.float32)
    flair[brain] = 35
    flair[radius < 12] = 50
    lesion = ((coords[0] - 23) ** 2 + (coords[1] - 20) ** 2 + (coords[2] - 21) ** 2) < 2.5**2
    flair[lesion] = 180
    affine = np.diag([-1.0, 1.0, 1.2, 1.0])
    affine[0, 3] = 39
    t1_file, flair_file = root / "sub-syn_ses-01_T1w.nii.gz", root / "sub-syn_ses-01_FLAIR.nii.gz"
    nib.save(nib.Nifti1Image(t1, affine), t1_file)
    nib.save(nib.Nifti1Image(flair, affine), flair_file)
    return t1_file, flair_file


def test_canonicalization_preserves_world_and_fixes_orientation(tmp_path: Path):
    t1, _ = synthetic_pair(tmp_path)
    before = nib.load(t1)
    output = tmp_path / "canonical.nii.gz"
    canonicalize_nifti(t1, output)
    after = nib.load(output)
    assert aff2axcodes(after.affine) == ("R", "A", "S")
    assert sorted(np.round(before.get_fdata().ravel(), 3)) == sorted(
        np.round(after.get_fdata().ravel(), 3)
    )


def test_validation_rejects_non_3d(tmp_path: Path):
    malformed = tmp_path / "bad.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4, 2), np.float32), np.eye(4)), malformed)
    with pytest.raises(ValueError, match="3D"):
        validate_image(malformed)


def test_mismatched_input_orientations_canonicalize_to_same_codes(tmp_path: Path):
    t1, flair = synthetic_pair(tmp_path)
    ras_flair = tmp_path / "flair_ras.nii.gz"
    nib.save(nib.as_closest_canonical(nib.load(flair)), ras_flair)
    t1_out, flair_out = tmp_path / "t1_ras.nii.gz", tmp_path / "flair_out_ras.nii.gz"
    canonicalize_nifti(t1, t1_out)
    canonicalize_nifti(ras_flair, flair_out)
    assert (
        aff2axcodes(nib.load(t1_out).affine)
        == aff2axcodes(nib.load(flair_out).affine)
        == ("R", "A", "S")
    )


def test_missing_external_wmh_backend_fails_cleanly(tmp_path: Path):
    with pytest.raises(WMHBackendUnavailable, match="not found"):
        run_truenet(
            "t1.nii.gz", "flair.nii.gz", tmp_path / "out.nii.gz", tmp_path, "definitely-not-truenet"
        )


def test_empty_wmh_postprocessing():
    result, metrics = postprocess_wmh(np.zeros((8, 8, 8), bool), (1, 1, 1), 3)
    assert not result.any() and metrics.component_count == 0 and metrics.lesion_volume_ml == 0


def test_structural_pipeline_and_qc(tmp_path: Path):
    t1, flair = synthetic_pair(tmp_path)
    derivatives = tmp_path / "derivatives"
    pre = preprocess_structural(t1, flair, derivatives, "sub-syn", "ses-01", n4_shrink_factor=2)
    wmh = segment_wmh_files(
        pre["outputs"]["flair_registered"],
        pre["outputs"]["t1_n4"],
        pre["outputs"]["brain_mask"],
        pre["tissues"]["wm"],
        derivatives / "sub-syn" / "ses-01" / "anat",
        "sub-syn",
        "ses-01",
        z_threshold=2.0,
        min_component_mm3=2,
    )
    qc = structural_qc(
        pre["outputs"]["t1_n4"],
        pre["outputs"]["flair_registered"],
        pre["tissues"],
        wmh["final_mask"],
        derivatives / "sub-syn" / "ses-01" / "qc",
        "sub-syn",
        "ses-01",
        pre["registration"]["metric"],
    )
    assert Path(qc["figure"]).exists() and Path(wmh["raw_mask"]).exists()
    assert wmh["metrics"]["postprocessed_voxels"] > 0
    resumed = preprocess_structural(t1, flair, derivatives, "sub-syn", "ses-01")
    assert resumed["resumed"] is True
