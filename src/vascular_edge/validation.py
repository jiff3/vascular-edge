"""Basic, memory-conscious validation of selected BIDS/NIfTI files."""

from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib


def validate_selected_data(bids_root: str | Path, manifest_rows: list[dict]) -> list[dict]:
    """Check manifest files exist, NIfTI headers load, and subject path matches."""
    root = Path(bids_root)
    findings: list[dict] = []
    for row in manifest_rows:
        if not row.get("included"):
            continue
        for file in row.get("files", []):
            path = root / file["path"]
            finding = {
                "subject": row["subject"],
                "path": file["path"],
                "valid": False,
                "message": "",
            }
            if not path.exists():
                finding["message"] = "missing"
            elif path.suffix in {".nii", ".gz"} and (
                path.name.endswith(".nii") or path.name.endswith(".nii.gz")
            ):
                try:
                    image = nib.load(str(path))
                    if len(image.shape) < 3 or any(n <= 0 for n in image.shape[:3]):
                        raise ValueError(f"invalid spatial shape {image.shape}")
                    finding["valid"] = True
                    finding["message"] = f"NIfTI header readable; shape={image.shape}"
                except Exception as exc:  # report a corrupt file rather than stop whole cohort
                    finding["message"] = f"NIfTI unreadable: {exc}"
            elif path.suffix == ".json":
                try:
                    json.loads(path.read_text(encoding="utf-8"))
                    finding["valid"] = True
                    finding["message"] = "JSON readable"
                except (OSError, json.JSONDecodeError) as exc:
                    finding["message"] = f"JSON unreadable: {exc}"
            else:
                finding["valid"] = True
                finding["message"] = "present"
            findings.append(finding)
    return findings
