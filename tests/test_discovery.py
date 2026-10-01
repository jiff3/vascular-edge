from pathlib import Path

from vascular_edge.discovery import discover_bids


def test_discovery_handles_optional_modalities(tmp_path: Path):
    anat = tmp_path / "sub-01" / "ses-02" / "anat"
    anat.mkdir(parents=True)
    (anat / "sub-01_ses-02_T1w.nii.gz").touch()
    (anat / "sub-01_ses-02_FLAIR.nii.gz").touch()
    frame = discover_bids(tmp_path)
    assert len(frame) == 1
    assert frame.loc[0, "t1w"] and frame.loc[0, "flair"]
    assert not frame.loc[0, "asl"]


def test_discovery_recognizes_dlbs_acq_flair_t2w(tmp_path: Path):
    anat = tmp_path / "sub-101" / "ses-wave1" / "anat"
    anat.mkdir(parents=True)
    (anat / "sub-101_ses-wave1_acq-MPRAGE_run-1_T1w.nii.gz").touch()
    (anat / "sub-101_ses-wave1_acq-FLAIR_run-1_T2w.nii.gz").touch()
    frame = discover_bids(tmp_path)
    assert len(frame) == 1 and frame.loc[0, "t1w"] and frame.loc[0, "flair"]
