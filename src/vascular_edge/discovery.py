"""BIDS inventory utilities that do not require every modality."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

MODALITY_PATTERNS = {
    "t1w": ("*_T1w.nii*",),
    # DLBS ds004856 encodes FLAIR as acq-FLAIR with the BIDS T2w suffix.
    "flair": ("*_FLAIR.nii*", "*acq-FLAIR*_T2w.nii*"),
    "asl": ("*_asl.nii*",),
    "bold": ("*_bold.nii*",),
}


def _entity(path: Path, key: str) -> str | None:
    for part in path.name.split("_"):
        if part.startswith(f"{key}-"):
            return part.removeprefix(f"{key}-")
    return None


def discover_bids(bids_root: str | Path) -> pd.DataFrame:
    """Return one row per subject/session with observed imaging modalities."""
    root = Path(bids_root)
    if not root.is_dir():
        raise FileNotFoundError(f"BIDS root does not exist: {root}")
    records: dict[tuple[str, str], dict[str, object]] = {}
    for modality, patterns in MODALITY_PATTERNS.items():
        images = {image for pattern in patterns for image in root.rglob(pattern)}
        for image in sorted(images):
            subject = _entity(image, "sub")
            if not subject:
                continue
            session = _entity(image, "ses") or "baseline"
            row = records.setdefault((subject, session), {"subject": subject, "session": session})
            row[modality] = True
            row[f"{modality}_file"] = str(image.relative_to(root))
    columns = ["subject", "session", *MODALITY_PATTERNS]
    if not records:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(records.values())
    for modality in MODALITY_PATTERNS:
        if modality not in frame:
            frame[modality] = False
        else:
            frame[modality] = frame[modality].astype("boolean").fillna(False).astype(bool)
    return frame.sort_values(["subject", "session"], kind="stable").reset_index(drop=True)
