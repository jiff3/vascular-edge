"""Cohort inventory, deterministic tiering, manifests, and selective retrieval plans."""

from __future__ import annotations

import json
import random
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd

from .openneuro import OpenNeuroClient, RemoteFile, is_imaging_file


@dataclass
class SessionInventory:
    session: str
    t1w: bool = False
    flair: bool = False
    asl: bool = False
    bold: bool = False
    bold_cvr: bool = False
    files: list[RemoteFile] = field(default_factory=list)


@dataclass
class SubjectInventory:
    subject: str
    sessions: dict[str, SessionInventory] = field(default_factory=dict)
    demographics_available: bool = False
    cognition_available: bool = False

    @property
    def longitudinal(self) -> bool:
        return len(self.sessions) >= 2


def classify_file(path: str) -> tuple[str, str] | None:
    """Return BIDS subject/session inferred from a valid participant path."""
    pieces = Path(path).parts
    subject = next((p for p in pieces if p.startswith("sub-")), None)
    if not subject:
        return None
    session = next((p for p in pieces if p.startswith("ses-")), "ses-baseline")
    return subject, session


def modality_for_path(path: str) -> str | None:
    lower = path.lower()
    if not is_imaging_file(lower):
        return None
    if lower.endswith("_t1w.nii") or lower.endswith("_t1w.nii.gz"):
        return "t1w"
    if "_flair.nii" in lower or ("acq-flair" in lower and "_t2w.nii" in lower):
        return "flair"
    if lower.endswith("_asl.nii") or lower.endswith("_asl.nii.gz") or "cbf" in lower:
        return "asl"
    # CVR is acquisition/task-specific; retain BOLD only when labels signal it.
    if lower.endswith("_bold.nii") or lower.endswith("_bold.nii.gz"):
        return (
            "bold_cvr"
            if any(x in lower for x in ("cvr", "hypercap", "co2", "breath", "_hc_"))
            else None
        )
    return None


def inventory_from_remote(
    subject_files: dict[str, list[RemoteFile]],
    demographics_available: bool = False,
    cognition_available: bool = False,
) -> list[SubjectInventory]:
    """Create subject/session inventory without assuming a common acquisition protocol."""
    subjects: dict[str, SubjectInventory] = {}
    for expected_subject, files in subject_files.items():
        subject = subjects.setdefault(
            expected_subject,
            SubjectInventory(
                expected_subject,
                demographics_available=demographics_available,
                cognition_available=cognition_available,
            ),
        )
        for item in files:
            classified = classify_file(item.path)
            if not classified:
                continue
            _, session_name = classified
            session = subject.sessions.setdefault(session_name, SessionInventory(session_name))
            if item.path.lower().endswith(("_bold.nii", "_bold.nii.gz")):
                session.bold = True
            modality = modality_for_path(item.path)
            if modality:
                setattr(session, modality, True)
            session.files.append(item)
    return [subjects[key] for key in sorted(subjects)]


def discover_openneuro(
    dataset_id: str, version: str | None = None, client: OpenNeuroClient | None = None
) -> list[SubjectInventory]:
    """Walk only metadata trees, serially, to inventory a remote BIDS dataset."""
    client = client or OpenNeuroClient(dataset_id, version)
    root = client.files()
    demographics = any(entry["filename"] == "participants.tsv" for entry in root)
    cognition = any("cogn" in entry["filename"].lower() for entry in root)
    derivatives = next(
        (entry for entry in root if entry["directory"] and entry["filename"] == "derivatives"), None
    )
    if derivatives:
        cognition = cognition or any(
            "cogn" in entry["filename"].lower() for entry in client.files(derivatives["id"])
        )
    subject_trees = {
        entry["filename"]: entry["id"]
        for entry in root
        if entry["directory"] and entry["filename"].startswith("sub-")
    }
    remote = {
        subject: client.subject_files(tree, subject)
        for subject, tree in sorted(subject_trees.items())
    }
    return inventory_from_remote(remote, demographics, cognition)


def tier_for_subject(subject: SubjectInventory) -> tuple[str, str]:
    sessions = list(subject.sessions.values())
    structural = [s for s in sessions if s.t1w and s.flair]
    asl_structural = [s for s in structural if s.asl]
    if len(asl_structural) >= 2:
        return "A", "longitudinal T1w+FLAIR+ASL available"
    if asl_structural:
        return "B", "T1w+FLAIR+ASL available"
    if len(structural) >= 2:
        return "C", "longitudinal T1w+FLAIR available"
    if any(s.t1w for s in sessions):
        return "D", "T1w structural fallback only"
    return "excluded", "no T1w image available"


def selected_files(subject: SubjectInventory, include_cvr: bool = False) -> list[RemoteFile]:
    """Pick image files and BIDS sidecars relevant to the available modalities."""
    result: list[RemoteFile] = []
    for session in subject.sessions.values():
        for item in session.files:
            modality = modality_for_path(item.path)
            if modality in {"t1w", "flair", "asl"} or (include_cvr and modality == "bold_cvr"):
                result.append(item)
                sidecar = item.path.rsplit(".nii", 1)[0] + ".json"
                # Sidecars are selected later only if present in inventory.
                result.extend(x for x in session.files if x.path == sidecar)
                if modality == "asl":
                    result.extend(
                        x
                        for x in session.files
                        if x.path.lower().endswith("_aslcontext.tsv")
                        or "_m0scan.nii" in x.path.lower()
                        or "_m0scan.json" in x.path.lower()
                    )
    return list({item.path: item for item in result}.values())


def select_cohort(
    inventory: Iterable[SubjectInventory],
    target_size: int = 30,
    max_download_gb: float = 80,
    seed: int = 2026,
    include_cvr: bool = False,
) -> list[dict]:
    """Rank A→D then seeded-shuffle within tier; cap by count and estimated bytes."""
    if target_size < 1 or max_download_gb <= 0:
        raise ValueError("target_size and max_download_gb must be positive.")
    grouped: dict[str, list[SubjectInventory]] = {key: [] for key in ("A", "B", "C", "D")}
    excluded: list[dict] = []
    for subject in inventory:
        tier, reason = tier_for_subject(subject)
        if tier == "excluded":
            excluded.append(_manifest_row(subject, tier, reason, include_cvr, False))
        else:
            grouped[tier].append(subject)
    rng = random.Random(seed)
    selected: list[dict] = []
    used = 0
    cap = int(max_download_gb * 1024**3)
    for tier in ("A", "B", "C", "D"):
        group = sorted(grouped[tier], key=lambda s: s.subject)
        rng.shuffle(group)
        for subject in group:
            row = _manifest_row(subject, tier, tier_for_subject(subject)[1], include_cvr, False)
            size = sum(file["size_bytes"] for file in row["files"])
            if len(selected) >= target_size:
                row["inclusion_reason"] = "not selected: target cohort size reached"
                excluded.append(row)
                continue
            if used + size > cap:
                row["inclusion_reason"] = "not selected: estimated download cap exceeded"
                excluded.append(row)
                continue
            row["included"] = True
            selected.append(row)
            used += size
    return selected + excluded


def _manifest_row(
    subject: SubjectInventory, tier: str, reason: str, include_cvr: bool, included: bool
) -> dict:
    files = [item.to_dict() for item in selected_files(subject, include_cvr)]
    sessions = {
        name: {"t1w": s.t1w, "flair": s.flair, "asl": s.asl, "bold": s.bold, "bold_cvr": s.bold_cvr}
        for name, s in sorted(subject.sessions.items())
    }
    return {
        "subject": subject.subject,
        "tier": tier,
        "included": included,
        "inclusion_reason": reason,
        "longitudinal": subject.longitudinal,
        "bold_available": any(s.bold for s in subject.sessions.values()),
        "cvr_available": any(s.bold_cvr for s in subject.sessions.values()),
        "demographics_available": subject.demographics_available,
        "cognition_available": subject.cognition_available,
        "sessions": sessions,
        "files": files,
        "estimated_bytes": sum(item["size_bytes"] for item in files),
    }


def write_manifest(rows: list[dict], stem: str | Path) -> tuple[Path, Path]:
    """Write JSON detail plus a flat, inspection-friendly CSV."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    json_path, csv_path = stem.with_suffix(".json"), stem.with_suffix(".csv")
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    flat = [
        {
            **{k: v for k, v in row.items() if k not in {"files", "sessions"}},
            "sessions": json.dumps(row["sessions"], sort_keys=True),
            "file_count": len(row["files"]),
        }
        for row in rows
    ]
    pd.DataFrame(flat).to_csv(csv_path, index=False)
    return json_path, csv_path


def inventory_to_dicts(inventory: Iterable[SubjectInventory]) -> list[dict]:
    """Serialize remote/local inventory for a reviewable intermediate artifact."""
    return [
        {
            "subject": subject.subject,
            "demographics_available": subject.demographics_available,
            "cognition_available": subject.cognition_available,
            "sessions": {
                name: {
                    "session": s.session,
                    "t1w": s.t1w,
                    "flair": s.flair,
                    "asl": s.asl,
                    "bold": s.bold,
                    "bold_cvr": s.bold_cvr,
                    "files": [item.to_dict() for item in s.files],
                }
                for name, s in subject.sessions.items()
            },
        }
        for subject in inventory
    ]


def inventory_from_dicts(rows: Iterable[dict]) -> list[SubjectInventory]:
    """Load a saved inventory without contacting OpenNeuro again."""
    output: list[SubjectInventory] = []
    for row in rows:
        subject = SubjectInventory(
            row["subject"],
            demographics_available=bool(row.get("demographics_available")),
            cognition_available=bool(row.get("cognition_available")),
        )
        for name, session in row.get("sessions", {}).items():
            subject.sessions[name] = SessionInventory(
                name,
                **{
                    key: bool(session.get(key))
                    for key in ("t1w", "flair", "asl", "bold", "bold_cvr")
                },
                files=[RemoteFile(**file) for file in session.get("files", [])],
            )
        output.append(subject)
    return output


def download_with_datalad(
    rows: list[dict], dataset_root: str | Path, dry_run: bool = True
) -> list[str]:
    """Run `datalad get` only for included manifest paths in an existing clone."""
    paths = [f["path"] for row in rows if row["included"] for f in row["files"]]
    if not paths:
        return []
    if dry_run:
        return paths
    if not shutil.which("datalad"):
        raise RuntimeError(
            "DataLad is required for retrieval. Install/configure it, then retry; use --dry-run to inspect."
        )
    root = Path(dataset_root)
    if not (root / ".git").exists():
        raise RuntimeError(f"Expected an OpenNeuro/DataLad clone at {root}; clone metadata first.")
    subprocess.run(["datalad", "get", *paths], cwd=root, check=True)
    return paths
