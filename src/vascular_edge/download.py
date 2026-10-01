"""Safe, manifest-based selective file downloading."""

from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path
from typing import Any

import yaml


def download_manifest(manifest_path: str | Path, output_root: str | Path) -> list[Path]:
    """Download declared files only, preventing manifest path traversal."""
    with Path(manifest_path).open(encoding="utf-8") as handle:
        manifest: dict[str, Any] = yaml.safe_load(handle) or {}
    files = manifest.get("download", {}).get("files", [])
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for item in files:
        relative = Path(item["path"])
        destination = (root / relative).resolve()
        if root not in destination.parents and destination != root:
            raise ValueError(f"Unsafe download destination: {relative}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(item["url"]) as response, destination.open("wb") as target:
            shutil.copyfileobj(response, target)
        written.append(destination)
    return written
