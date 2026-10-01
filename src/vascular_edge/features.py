"""Tabular feature output with stable column names."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def export_features(records: list[dict], output_file: str | Path) -> Path:
    """Write a tidy CSV, creating parent directories as needed."""
    output = Path(output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output, index=False)
    return output
