"""Portable YAML configuration handling."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path | None) -> dict[str, Any]:
    """Load a YAML config, returning an empty configuration when omitted."""
    if path is None:
        return {}
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a YAML mapping.")
    return config
