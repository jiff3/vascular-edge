"""Dataset-aware metadata and cognition harmonization.

The harmonizer discovers identifiers, waves, and candidate variables from file
contents.  Original names are retained in a data dictionary so that a canonical
column never silently changes the meaning or coding of a source variable.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd

MISSING = ["NA", "N/A", "n/a", "na", "xx", "", "."]
ID_NAMES = ("participant_id", "subject", "subject_id", "sub", "s#", "id")
SESSION_NAMES = ("session", "session_id", "timepoint", "wave", "visit")


def _clean(name: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def _subject(value: object) -> str:
    text = str(value).strip()
    if text.lower().startswith("sub-"):
        return "sub-" + text[4:]
    return "sub-" + re.sub(r"\.0$", "", text)


def _session(value: object) -> str:
    text = str(value).strip().lower()
    match = re.search(r"(?:wave|w)[-_ ]?([0-9]+)", text)
    if match:
        return f"ses-wave{match.group(1)}"
    if text.startswith("ses-"):
        return text
    return f"ses-{text}"


def _session_value(value: object, column: str) -> str:
    if _clean(column) == "wave" and pd.notna(value):
        wave = re.sub(r"\.0$", "", str(value).strip())
        return f"ses-wave{wave}"
    return _session(value)


def _read_table(path: Path, sheet_name: str | int | None = None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet_name, na_values=MISSING, keep_default_na=True)
    separator = "\t" if suffix == ".tsv" else ","
    return pd.read_csv(path, sep=separator, na_values=MISSING, keep_default_na=True)


def discover_metadata_files(bids_root: str | Path) -> dict[str, list[Path]]:
    """Find participant and behavioral tables without relying on exact filenames."""
    root = Path(bids_root)
    tables = sorted(
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in {".tsv", ".csv", ".xlsx", ".xls"}
    )
    participants, cognition, other = [], [], []
    for path in tables:
        low = str(path.relative_to(root)).lower()
        if path.name.lower().startswith("participants."):
            participants.append(path)
        elif any(
            token in low
            for token in ("cogn", "memory", "reason", "vocab", "fluency", "processing", "behavior")
        ):
            cognition.append(path)
        else:
            other.append(path)
    return {"participants": participants, "cognition": cognition, "other": other}


def _find_column(columns: Iterable[object], candidates: Iterable[str]) -> str | None:
    lookup = {_clean(column): str(column) for column in columns}
    return next((lookup[_clean(name)] for name in candidates if _clean(name) in lookup), None)


def _wave_suffix(column: object) -> int | None:
    match = re.search(r"(?:_|\b)(?:w|wave)[-_ ]?([0-9]+)$", str(column), re.I)
    return int(match.group(1)) if match else None


def _base_name(column: str) -> str:
    return _clean(re.sub(r"(?:_|\b)(?:w|wave)[-_ ]?[0-9]+$", "", column, flags=re.I))


def harmonize_participants(path: str | Path) -> tuple[pd.DataFrame, list[dict]]:
    """Convert wide or long participant metadata into subject-session rows."""
    path = Path(path)
    frame = _read_table(path)
    id_col = _find_column(frame.columns, ID_NAMES)
    if not id_col:
        raise ValueError(f"No participant identifier found in {path}")
    session_col = _find_column(frame.columns, SESSION_NAMES)
    records: list[dict] = []
    dictionary: list[dict] = []
    waves = sorted({wave for column in frame.columns if (wave := _wave_suffix(column)) is not None})
    invariant = [
        str(c)
        for c in frame.columns
        if c != id_col and _wave_suffix(c) is None and c != session_col
    ]
    if session_col:
        for _, source in frame.iterrows():
            row = {
                "subject": _subject(source[id_col]),
                "session": _session_value(source[session_col], session_col),
            }
            row.update({_clean(column): source[column] for column in invariant})
            records.append(row)
    elif waves:
        for _, source in frame.iterrows():
            for wave in waves:
                row = {"subject": _subject(source[id_col]), "session": f"ses-wave{wave}"}
                row.update({_clean(column): source[column] for column in invariant})
                for column in frame.columns:
                    if _wave_suffix(column) == wave:
                        row[_base_name(str(column))] = source[column]
                records.append(row)
    else:
        for _, source in frame.iterrows():
            row = {"subject": _subject(source[id_col]), "session": "ses-baseline"}
            row.update({_clean(column): source[column] for column in invariant})
            records.append(row)
    out = pd.DataFrame(records)
    aliases = {
        "age": ("agemri", "age_mri", "age", "agecog", "age_cog"),
        "sex": ("sex", "gender"),
        "education_years": ("eduyrsestcap", "education_years", "education", "edu_years"),
        "education_level": ("educomp", "education_level"),
        "bmi": ("bmi",),
        "mmse": ("mmse",),
        "race": ("race",),
        "ethnicity": ("ethnicity",),
    }
    for canonical, choices in aliases.items():
        selected = next((choice for choice in choices if choice in out.columns), None)
        if selected:
            out[canonical] = out[selected]
            dictionary.append(
                {
                    "analysis_column": canonical,
                    "source_file": str(path),
                    "source_column": selected,
                    "role": "demographic_or_covariate",
                    "transformation": "renamed; coding preserved",
                }
            )
    for column in out.columns:
        if column not in {"subject", "session"} and not any(
            x["analysis_column"] == column for x in dictionary
        ):
            dictionary.append(
                {
                    "analysis_column": column,
                    "source_file": str(path),
                    "source_column": column,
                    "role": "source_metadata",
                    "transformation": "name normalized; values preserved",
                }
            )
    return out, dictionary


ADMIN_PATTERNS = re.compile(
    r"^(age|sex|race|ethnicity|handed|mmse|cogw|mriw|pet|takehome|edu|construct|wave|hasdata|numtasks|task[0-9]*$)",
    re.I,
)


def _domain_from(path: Path, frame: pd.DataFrame, sheet: str) -> str:
    construct = _find_column(frame.columns, ("ConstructName", "domain", "construct"))
    if construct and frame[construct].notna().any():
        return _clean(frame[construct].dropna().astype(str).iloc[0])
    return _clean(re.sub(r"[-_ ]?(?:w|wave)[-_ ]?[0-9]+$", "", sheet, flags=re.I) or path.stem)


def harmonize_cognition(
    paths: Iterable[str | Path], composites: Mapping[str, object] | None = None
) -> tuple[pd.DataFrame, list[dict]]:
    """Preserve numeric cognitive variables and optionally build specified composites.

    A composite is only made when configuration lists its component columns and
    optional signs.  Automatic composites are deliberately avoided because raw,
    age-corrected, percentile, accuracy, and latency scores have incompatible scales.
    """
    tables: list[pd.DataFrame] = []
    dictionary: list[dict] = []
    for raw_path in paths:
        path = Path(raw_path)
        sheets = (
            pd.ExcelFile(path).sheet_names
            if path.suffix.lower() in {".xlsx", ".xls"}
            else [path.stem]
        )
        for sheet in sheets:
            frame = _read_table(path, sheet if path.suffix.lower() in {".xlsx", ".xls"} else None)
            if frame.empty:
                continue
            id_col = _find_column(frame.columns, ID_NAMES)
            session_col = _find_column(frame.columns, SESSION_NAMES)
            if not id_col:
                continue
            domain = _domain_from(path, frame, str(sheet))
            wave_from_sheet = re.search(r"(?:w|wave)[-_ ]?([0-9]+)", str(sheet), re.I)
            keys = pd.DataFrame({"subject": frame[id_col].map(_subject)})
            if session_col:
                keys["session"] = frame[session_col].map(
                    lambda value: _session_value(value, session_col)
                )
            elif wave_from_sheet:
                keys["session"] = f"ses-wave{wave_from_sheet.group(1)}"
            else:
                keys["session"] = "ses-baseline"
            values = keys.copy()
            for column in frame.columns:
                clean = _clean(column)
                if (
                    str(column) == id_col
                    or str(column) == session_col
                    or ADMIN_PATTERNS.search(clean)
                ):
                    continue
                numeric = pd.to_numeric(frame[column], errors="coerce")
                if numeric.notna().sum() < 3 or numeric.nunique(dropna=True) < 2:
                    continue
                target = f"cog_{domain}_{clean}"
                values[target] = numeric
                dictionary.append(
                    {
                        "analysis_column": target,
                        "source_file": str(path),
                        "source_sheet": str(sheet),
                        "source_column": str(column),
                        "role": "cognitive_measure",
                        "transformation": "numeric coercion; original scale and direction preserved",
                    }
                )
            if len(values.columns) == 2:
                continue
            tables.append(values)
    if tables:
        # Wave sheets stack vertically; construct workbooks contribute different
        # columns to the same subject-session. ``first`` coalesces those sparse
        # blocks without manufacturing values.
        result = (
            pd.concat(tables, ignore_index=True, sort=False)
            .groupby(["subject", "session"], as_index=False, dropna=False)
            .first()
        )
    else:
        result = pd.DataFrame(columns=["subject", "session"])
    for name, spec in (composites or {}).items():
        components = spec.get("components", []) if isinstance(spec, Mapping) else list(spec)
        signs = (
            spec.get("signs", [1] * len(components))
            if isinstance(spec, Mapping)
            else [1] * len(components)
        )
        present = [column for column in components if column in result]
        if len(present) < 2 or len(present) != len(components):
            continue
        z = []
        for column, sign in zip(components, signs):
            values = pd.to_numeric(result[column], errors="coerce")
            z.append(float(sign) * (values - values.mean()) / values.std(ddof=1))
        result[f"cog_domain_{_clean(name)}_z"] = pd.concat(z, axis=1).mean(axis=1, skipna=False)
        dictionary.append(
            {
                "analysis_column": f"cog_domain_{_clean(name)}_z",
                "source_file": "configured composite",
                "source_column": json.dumps(components),
                "role": "cognitive_domain_composite",
                "transformation": "mean of signed sample z-scores; complete components required",
            }
        )
    return result, dictionary


def write_data_dictionary(records: list[dict], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).drop_duplicates().to_csv(output, index=False)
    return output
