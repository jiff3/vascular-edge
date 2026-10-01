from pathlib import Path

from vascular_edge.cohort import (
    SessionInventory,
    SubjectInventory,
    discover_openneuro,
    inventory_from_remote,
    select_cohort,
    write_manifest,
)
from vascular_edge.openneuro import RemoteFile


def _file(path: str, size: int = 100) -> RemoteFile:
    return RemoteFile(path, size, True)


def test_inventory_tracks_modalities_and_sessions():
    inventory = inventory_from_remote(
        {
            "sub-01": [
                _file("sub-01/ses-wave1/anat/sub-01_ses-wave1_T1w.nii.gz"),
                _file("sub-01/ses-wave1/anat/sub-01_ses-wave1_FLAIR.nii.gz"),
                _file("sub-01/ses-wave2/perf/sub-01_ses-wave2_asl.nii.gz"),
                _file("sub-01/ses-wave2/func/sub-01_ses-wave2_task-hypercapnia_bold.nii.gz"),
            ]
        },
        demographics_available=True,
        cognition_available=True,
    )
    person = inventory[0]
    assert person.longitudinal and person.demographics_available and person.cognition_available
    assert person.sessions["ses-wave1"].flair
    assert person.sessions["ses-wave2"].asl and person.sessions["ses-wave2"].bold_cvr


def test_inventory_recognizes_dlbs_flair_naming():
    person = inventory_from_remote(
        {
            "sub-101": [
                _file("sub-101/ses-wave1/anat/sub-101_ses-wave1_acq-MPRAGE_run-1_T1w.nii.gz"),
                _file("sub-101/ses-wave1/anat/sub-101_ses-wave1_acq-FLAIR_run-1_T2w.nii.gz"),
            ]
        }
    )[0]
    assert person.sessions["ses-wave1"].t1w and person.sessions["ses-wave1"].flair


def test_selection_is_deterministic_and_respects_cap(tmp_path: Path):
    cohort = []
    for name in ("sub-01", "sub-02", "sub-03"):
        subject = SubjectInventory(name)
        subject.sessions["ses-01"] = SessionInventory(
            "ses-01",
            t1w=True,
            flair=True,
            asl=True,
            files=[_file(f"{name}/ses-01/a_T1w.nii.gz", 100)],
        )
        cohort.append(subject)
    rows1 = select_cohort(cohort, target_size=2, max_download_gb=1, seed=7)
    rows2 = select_cohort(cohort, target_size=2, max_download_gb=1, seed=7)
    assert [r["subject"] for r in rows1 if r["included"]] == [
        r["subject"] for r in rows2 if r["included"]
    ]
    assert sum(r["included"] for r in rows1) == 2
    json_path, csv_path = write_manifest(rows1, tmp_path / "manifest")
    assert json_path.exists() and csv_path.exists()


def test_download_limit_can_exclude_all_candidates():
    subject = SubjectInventory("sub-01")
    subject.sessions["ses-01"] = SessionInventory(
        "ses-01",
        t1w=True,
        files=[_file("sub-01/ses-01/anat/sub-01_ses-01_T1w.nii.gz", 2 * 1024**3)],
    )
    rows = select_cohort([subject], target_size=1, max_download_gb=1)
    assert not rows[0]["included"]
    assert "cap exceeded" in rows[0]["inclusion_reason"]


def test_remote_discovery_uses_metadata_tree_without_image_bytes():
    class FakeClient:
        def files(self, tree=None, recursive=False):
            if tree is None:
                return [
                    {"filename": "participants.tsv", "directory": False},
                    {"filename": "derivatives", "directory": True, "id": "deriv"},
                    {"filename": "sub-01", "directory": True, "id": "subtree"},
                ]
            if tree == "deriv":
                return [{"filename": "cognition", "directory": True, "id": "cog"}]
            return []

        def subject_files(self, tree, name):
            assert tree == "subtree" and name == "sub-01"
            return [_file("sub-01/ses-wave1/anat/sub-01_ses-wave1_T1w.nii.gz")]

    result = discover_openneuro("ds004856", "1.3.0", client=FakeClient())
    assert result[0].demographics_available and result[0].cognition_available


def test_asl_selection_includes_context_and_calibration_sidecars():
    subject = inventory_from_remote(
        {
            "sub-01": [
                _file("sub-01/ses-01/anat/sub-01_ses-01_T1w.nii.gz"),
                _file("sub-01/ses-01/anat/sub-01_ses-01_FLAIR.nii.gz"),
                _file("sub-01/ses-01/perf/sub-01_ses-01_run-1_asl.nii.gz"),
                _file("sub-01/ses-01/perf/sub-01_ses-01_run-1_asl.json"),
                _file("sub-01/ses-01/perf/sub-01_ses-01_aslcontext.tsv"),
                _file("sub-01/ses-01/perf/sub-01_ses-01_m0scan.nii.gz"),
            ]
        }
    )[0]
    rows = select_cohort([subject], target_size=1, max_download_gb=1)
    paths = {item["path"] for item in rows[0]["files"]}
    assert any(path.endswith("aslcontext.tsv") for path in paths)
    assert any("m0scan" in path for path in paths)
