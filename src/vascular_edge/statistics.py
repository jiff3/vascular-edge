"""Cohort analysis-table assembly, exploratory summaries, and honest reporting."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import re
import sys
from pathlib import Path
from typing import Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests

from .metadata import (
    discover_metadata_files,
    harmonize_cognition,
    harmonize_participants,
    write_data_dictionary,
)


def _csvs(root: Path, pattern: str) -> list[Path]:
    return sorted(root.rglob(pattern)) if root.exists() else []


def _concat(paths: Iterable[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(path, na_values=["NA", "n/a", "xx"]) for path in paths]
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _norm_subject(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if "participant_id" in frame and "subject" not in frame:
        frame = frame.rename(columns={"participant_id": "subject"})
    if "subject" in frame:
        frame["subject"] = (
            frame["subject"].astype(str).map(lambda x: x if x.startswith("sub-") else f"sub-{x}")
        )
    return frame


def _pivot_perilesional(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if frame.empty:
        return pd.DataFrame(columns=["subject", "session"]), frame
    frame = _norm_subject(frame)
    frame = (
        frame[frame.get("scope", "subject").astype(str).eq("subject")].copy()
        if "scope" in frame
        else frame
    )
    needed = {"subject", "session", "map", "region", "mean"}
    if not needed.issubset(frame):
        return pd.DataFrame(columns=["subject", "session"]), frame
    frame["feature"] = (
        frame["map"].astype(str) + "__" + frame["region"].astype(str) + "__mean"
    ).map(lambda x: re.sub(r"[^a-zA-Z0-9_]+", "_", x).lower())
    wide = frame.pivot_table(
        index=["subject", "session"], columns="feature", values="mean", aggfunc="first"
    ).reset_index()
    lesion = (
        frame.groupby(["subject", "session"], as_index=False)["lesion_volume_ml"].first()
        if "lesion_volume_ml" in frame
        else None
    )
    if lesion is not None:
        wide = wide.merge(
            lesion.rename(columns={"lesion_volume_ml": "wmh_volume_ml"}),
            on=["subject", "session"],
            how="left",
        )
    return wide, frame.drop(columns="feature")


def build_analysis_table(
    bids_root: str | Path,
    derivatives_root: str | Path,
    output_dir: str | Path,
    config: Mapping[str, object] | None = None,
) -> dict:
    """Create documented subject-session, profile, and longitudinal analysis tables."""
    bids = Path(bids_root)
    derivatives = Path(derivatives_root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = dict(config or {})
    found = discover_metadata_files(bids)
    dictionary: list[dict] = []
    if found["participants"]:
        metadata, entries = harmonize_participants(found["participants"][0])
        dictionary.extend(entries)
    else:
        metadata = pd.DataFrame(columns=["subject", "session"])
    cognition, entries = harmonize_cognition(
        found["cognition"], config.get("cognitive_composites", {})
    )
    dictionary.extend(entries)
    profiles = _concat(_csvs(derivatives, "*desc-perilesional_features.csv"))
    perilesional, profiles = _pivot_perilesional(profiles)
    longitudinal = _norm_subject(_concat(_csvs(derivatives, "*desc-longitudinal_subject.csv")))
    if not longitudinal.empty and "baseline_session" in longitudinal:
        longitudinal["session"] = longitudinal["baseline_session"]
    imaging_keys = [
        frame[["subject", "session"]]
        for frame in (perilesional, longitudinal)
        if not frame.empty and {"subject", "session"}.issubset(frame)
    ]
    all_keys = [
        frame[["subject", "session"]]
        for frame in (metadata, cognition, perilesional, longitudinal)
        if not frame.empty and {"subject", "session"}.issubset(frame)
    ]
    keys = imaging_keys or all_keys
    table = (
        pd.concat(keys, ignore_index=True).drop_duplicates()
        if keys
        else pd.DataFrame(columns=["subject", "session"])
    )
    for frame in (metadata, cognition, perilesional, longitudinal):
        if not frame.empty and {"subject", "session"}.issubset(frame):
            extras = [
                column
                for column in frame.columns
                if column not in table.columns or column in {"subject", "session"}
            ]
            table = table.merge(
                frame[extras], on=["subject", "session"], how="left", validate="one_to_one"
            )
    if "baseline_wmh_ml" in table:
        table["baseline_wmh_volume_ml"] = table["baseline_wmh_ml"]
    elif "wmh_volume_ml" in table:
        table["baseline_wmh_volume_ml"] = table["wmh_volume_ml"]
    if "absolute_wmh_change_ml" in table:
        table["lesion_progression_ml"] = table["absolute_wmh_change_ml"]
    table = table.sort_values(["subject", "session"]).reset_index(drop=True)
    table_path = output / "analysis_table.csv"
    table.to_csv(table_path, index=False)
    profile_path = output / "perilesional_profiles.csv"
    profiles.to_csv(profile_path, index=False)
    long_path = output / "longitudinal_subjects.csv"
    longitudinal.to_csv(long_path, index=False)
    dictionary.extend(
        {
            "analysis_column": column,
            "source_file": "pipeline derivatives",
            "source_column": column,
            "role": "imaging_or_physiology",
            "transformation": "joined or subject-level pivot",
        }
        for column in table.columns
        if column not in {x["analysis_column"] for x in dictionary}
    )
    dictionary_path = write_data_dictionary(dictionary, output / "analysis_data_dictionary.csv")
    eligibility = (
        pd.concat(all_keys, ignore_index=True).drop_duplicates()
        if all_keys
        else table[["subject", "session"]]
    )
    imaging_index = (
        set(map(tuple, pd.concat(imaging_keys, ignore_index=True).drop_duplicates().to_numpy()))
        if imaging_keys
        else set(map(tuple, table[["subject", "session"]].to_numpy()))
    )
    eligibility["included"] = [
        tuple(row) in imaging_index for row in eligibility[["subject", "session"]].to_numpy()
    ]
    eligibility["reason"] = np.where(
        eligibility.included,
        "pipeline imaging feature row available",
        "excluded: no pipeline imaging feature row",
    )
    included = eligibility.sort_values(["subject", "session"])
    included_path = output / "included_subject_sessions.csv"
    included.to_csv(included_path, index=False)
    manifest = {
        "seed": int(config.get("random_seed", 2026)),
        "inputs": {
            "bids_root": str(bids.resolve()),
            "derivatives_root": str(derivatives.resolve()),
            "participant_files": [str(x) for x in found["participants"]],
            "cognition_files": [str(x) for x in found["cognition"]],
        },
        "rows": len(table),
        "subjects": int(table.subject.nunique()) if "subject" in table else 0,
        "cognitive_composites": config.get("cognitive_composites", {}),
        "note": "No cognitive composite is inferred automatically; configured components must all be present.",
    }
    manifest_path = output / "analysis_table_provenance.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "analysis_table": str(table_path),
        "perilesional_profiles": str(profile_path),
        "longitudinal_subjects": str(long_path),
        "data_dictionary": str(dictionary_path),
        "included": str(included_path),
        "provenance": str(manifest_path),
        **manifest,
    }


def _numeric(frame: pd.DataFrame, minimum: int = 3) -> list[str]:
    excluded = re.compile(r"(^conversion_centroid|_voxels$|^region_order$)")
    return [
        column
        for column in frame.select_dtypes(include=np.number).columns
        if frame[column].notna().sum() >= minimum
        and frame[column].nunique(dropna=True) > 1
        and not excluded.search(column)
    ]


def _bootstrap_pair(
    x: pd.Series, y: pd.Series, seed: int, repetitions: int = 2000
) -> tuple[float, float, int]:
    pair = pd.concat([x, y], axis=1).dropna().to_numpy(float)
    n = len(pair)
    if n < 4:
        return np.nan, np.nan, n
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repetitions):
        sample = pair[rng.integers(0, n, n)]
        if np.std(sample[:, 0]) and np.std(sample[:, 1]):
            values.append(np.corrcoef(sample[:, 0], sample[:, 1])[0, 1])
    low, high = np.percentile(values, [2.5, 97.5]) if values else (np.nan, np.nan)
    return float(low), float(high), n


def _correlations(
    frame: pd.DataFrame, columns: list[str], seed: int, repetitions: int
) -> pd.DataFrame:
    rows = []
    for i, left in enumerate(columns):
        for right in columns[i + 1 :]:
            pair = frame[[left, right]].dropna()
            if len(pair) < 3:
                continue
            pearson = stats.pearsonr(pair[left], pair[right])
            spearman = stats.spearmanr(pair[left], pair[right])
            low, high, n = _bootstrap_pair(pair[left], pair[right], seed, repetitions)
            rows.append(
                {
                    "variable_1": left,
                    "variable_2": right,
                    "n": n,
                    "pearson_r": pearson.statistic,
                    "pearson_p": pearson.pvalue,
                    "pearson_boot_ci_low": low,
                    "pearson_boot_ci_high": high,
                    "spearman_rho": spearman.statistic,
                    "spearman_p": spearman.pvalue,
                }
            )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["pearson_q_bh"] = multipletests(result.pearson_p, method="fdr_bh")[1]
        result["spearman_q_bh"] = multipletests(result.spearman_p, method="fdr_bh")[1]
    return result


def _relationships(
    frame: pd.DataFrame, predictor: str, outcomes: list[str], family: str
) -> pd.DataFrame:
    rows = []
    for outcome in outcomes:
        columns = [predictor, outcome] + [
            x for x in ("sex", "education_years") if x in frame and x not in {predictor, outcome}
        ]
        data = frame[columns].dropna().copy()
        if len(data) < max(8, len(columns) + 3):
            rows.append(
                {
                    "family": family,
                    "outcome": outcome,
                    "predictor": predictor,
                    "n": len(data),
                    "status": "not_estimable",
                    "reason": "insufficient complete cases",
                }
            )
            continue
        design = pd.get_dummies(data.drop(columns=outcome), drop_first=True, dtype=float)
        design = sm.add_constant(design, has_constant="add")
        try:
            fit = sm.OLS(pd.to_numeric(data[outcome]), design).fit()
            rows.append(
                {
                    "family": family,
                    "outcome": outcome,
                    "predictor": predictor,
                    "n": int(fit.nobs),
                    "status": "estimated",
                    "estimate": fit.params[predictor],
                    "std_error": fit.bse[predictor],
                    "ci_low": fit.conf_int().loc[predictor, 0],
                    "ci_high": fit.conf_int().loc[predictor, 1],
                    "p_value": fit.pvalues[predictor],
                    "r_squared": fit.rsquared,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "family": family,
                    "outcome": outcome,
                    "predictor": predictor,
                    "n": len(data),
                    "status": "not_estimable",
                    "reason": str(exc),
                }
            )
    result = pd.DataFrame(rows)
    ok = result.get("status", pd.Series(dtype=str)).eq("estimated")
    result["q_bh"] = np.nan
    if ok.any():
        result.loc[ok, "q_bh"] = multipletests(result.loc[ok, "p_value"], method="fdr_bh")[1]
    return result


def _save_figures(
    frame: pd.DataFrame, profiles: pd.DataFrame, output: Path, numeric: list[str]
) -> list[str]:
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    paths = []
    show = numeric[:12]
    if show:
        ncols = 3
        nrows = int(np.ceil(len(show) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.2 * nrows))
        for ax, column in zip(np.atleast_1d(axes).ravel(), show):
            ax.hist(frame[column].dropna(), bins="auto", color="#375a7f", alpha=0.85)
            ax.set_title(column)
            ax.set_ylabel("Count")
        for ax in np.atleast_1d(axes).ravel()[len(show) :]:
            ax.axis("off")
        fig.tight_layout()
        path = figures / "distributions.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        paths.append(str(path))
    needed = {"map", "region_order", "mean", "subject"}
    if not profiles.empty and needed.issubset(profiles):
        physiology = profiles[profiles["map"].ne("geometry")].copy()
        if "scope" in physiology:
            physiology = physiology[physiology["scope"].eq("subject")]
        summary = (
            physiology.groupby(["map", "units", "region", "region_order"], as_index=False)["mean"]
            .agg(["mean", "sem"])
            .reset_index()
        )
        maps = list(summary["map"].unique())
        fig, axes = plt.subplots(1, len(maps), figsize=(6.2 * len(maps), 4.8), squeeze=False)
        for ax, name in zip(axes.ravel(), maps):
            group = summary[summary["map"].eq(name)].sort_values("region_order")
            ax.errorbar(
                group.region_order,
                group["mean"],
                yerr=1.96 * group["sem"],
                marker="o",
                color="#2a9d8f",
                capsize=3,
            )
            ax.set_xticks(
                group.region_order, group.region.str.replace("_", " "), rotation=35, ha="right"
            )
            ax.set(
                xlabel="Distance region",
                ylabel=str(group.units.iloc[0]),
                title=name.replace("_", " ").title(),
            )
            ax.grid(axis="y", alpha=0.2)
        fig.suptitle("Cohort physiology by distance from WMH (mean and 95% CI)")
        fig.tight_layout()
        path = figures / "physiology_distance_profiles.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        paths.append(str(path))
    return paths


def software_versions() -> dict:
    packages = {}
    for name in ("vascular-edge", "numpy", "pandas", "scipy", "statsmodels", "matplotlib"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not installed as distribution"
    return {"python": sys.version, "platform": platform.platform(), "packages": packages}


def run_statistics(
    table_file: str | Path,
    output_dir: str | Path,
    profiles_file: str | Path | None = None,
    config: Mapping[str, object] | None = None,
) -> dict:
    """Generate descriptive/exploratory statistics with family-wise BH correction."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = dict(config or {})
    seed = int(config.get("random_seed", 2026))
    repetitions = int(config.get("bootstrap_repetitions", 2000))
    frame = pd.read_csv(table_file)
    profiles = (
        pd.read_csv(profiles_file)
        if profiles_file and Path(profiles_file).exists()
        else pd.DataFrame()
    )
    numeric = _numeric(frame)
    missing = pd.DataFrame(
        {
            "variable": frame.columns,
            "n_total": len(frame),
            "n_available": [frame[x].notna().sum() for x in frame],
            "n_missing": [frame[x].isna().sum() for x in frame],
            "percent_missing": [100 * frame[x].isna().mean() for x in frame],
        }
    )
    missing.to_csv(output / "sample_size_missingness.csv", index=False)
    distributions = (
        frame[numeric]
        .describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
        .T.reset_index(names="variable")
        if numeric
        else pd.DataFrame()
    )
    distributions.to_csv(output / "distributions.csv", index=False)
    selected = [
        x
        for x in numeric
        if any(
            token in x
            for token in ("age", "wmh", "perfusion", "cvr", "cog_", "physiology", "progression")
        )
    ]
    correlations = _correlations(frame, selected[:30], seed, repetitions)
    correlations.to_csv(output / "pairwise_correlations.csv", index=False)
    outcomes = [x for x in selected if x != "age"]
    age = (
        _relationships(frame, "age", outcomes, "exploratory_age")
        if "age" in frame
        else pd.DataFrame()
    )
    age.to_csv(output / "age_relationships.csv", index=False)
    burden_name = next(
        (x for x in ("baseline_wmh_volume_ml", "baseline_wmh_ml", "wmh_volume_ml") if x in frame),
        None,
    )
    burden = (
        _relationships(
            frame, burden_name, [x for x in outcomes if x != burden_name], "exploratory_wmh_burden"
        )
        if burden_name
        else pd.DataFrame()
    )
    burden.to_csv(output / "wmh_burden_relationships.csv", index=False)
    if not profiles.empty and {"map", "region", "mean"}.issubset(profiles):
        profile_summary = (
            profiles.groupby(["map", "units", "region", "region_order"], dropna=False)["mean"]
            .agg(n="count", mean="mean", sd="std", sem="sem")
            .reset_index()
        )
        profile_summary["ci_low"] = profile_summary["mean"] - 1.96 * profile_summary["sem"]
        profile_summary["ci_high"] = profile_summary["mean"] + 1.96 * profile_summary["sem"]
    else:
        profile_summary = pd.DataFrame()
    profile_summary.to_csv(output / "physiology_distance_profiles.csv", index=False)
    conversion_cols = [
        x
        for x in (
            "baseline_wmh_ml",
            "followup_wmh_ml",
            "absolute_wmh_change_ml",
            "percent_wmh_change",
            "newly_affected_ml",
        )
        if x in frame
    ]
    conversion = (
        frame[conversion_cols].describe().T.reset_index(names="measure")
        if conversion_cols
        else pd.DataFrame()
    )
    conversion.to_csv(output / "longitudinal_conversion_summary.csv", index=False)
    figures = _save_figures(frame, profiles, output, numeric)
    run = {
        "seed": seed,
        "bootstrap_repetitions": repetitions,
        "n_rows": len(frame),
        "n_subjects": int(frame.subject.nunique()) if "subject" in frame else None,
        "primary_analyses": config.get(
            "primary_analyses", ["R perilesional mixed model", "R longitudinal progression model"]
        ),
        "exploratory_analyses": [
            "pairwise correlations",
            "age relationships",
            "WMH-burden relationships",
        ],
        "multiple_comparisons": "Benjamini-Hochberg within each output family",
        "software": software_versions(),
        "figures": figures,
    }
    (output / "statistics_provenance.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    return run


def build_results(statistics_dir: str | Path, output_file: str | Path) -> Path:
    """Build a restrained Markdown report from generated estimates only."""
    root = Path(statistics_dir)
    output = Path(output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    provenance = json.loads((root / "statistics_provenance.json").read_text(encoding="utf-8"))
    lines = [
        "# Cohort-level statistical results",
        "",
        f"Analysis included {provenance.get('n_subjects', 'unknown')} subjects and {provenance['n_rows']} subject-session rows.",
        "",
        "## Prespecified analyses",
        "",
    ]
    status_path, estimates_path = root / "model_status.csv", root / "model_estimates.csv"
    if status_path.exists():
        statuses = pd.read_csv(status_path)
        primary_status = statuses[statuses.get("analysis_class", "").eq("prespecified_primary")]
        for _, row in primary_status.iterrows():
            if row.get("status") == "estimated":
                lines.append(
                    f"- {row['model']}: estimated with n = {int(row['n'])} complete observations; coefficients are in `model_estimates.csv`."
                )
            else:
                lines.append(
                    f"- {row['model']}: not estimable ({row.get('reason', 'reason unavailable')}; n = {int(row['n']) if pd.notna(row.get('n')) else 'unknown'})."
                )
    else:
        lines.append(
            "Prespecified R models have not been run. Their absence is not evidence of no association."
        )
    if estimates_path.exists():
        estimates = pd.read_csv(estimates_path)
        primary = estimates[
            (estimates.get("analysis_class", "") == "prespecified_primary")
            & (estimates.get("term", "") != "(Intercept)")
        ]
        adjusted = primary[primary.get("q_bh", pd.Series(np.nan, index=primary.index)) < 0.05]
        if adjusted.empty and not primary.empty:
            lines.append(
                f"- None of {len(primary)} non-intercept primary coefficients passed BH control at q < 0.05."
            )
        for _, row in adjusted.iterrows():
            direction = "higher" if row["estimate"] > 0 else "lower"
            lines.append(
                f"- {row['model']}, {row['term']}: {direction} outcome values were associated with the predictor "
                f"(estimate {row['estimate']:.3g}, 95% CI {row['conf.low']:.3g} to {row['conf.high']:.3g}, q = {row['q_bh']:.3g})."
            )
    lines += [
        "",
        "A missing table or a `not_estimable` status is not evidence of no association.",
        "",
        "## Exploratory analyses",
        "",
    ]
    correlation_path = root / "pairwise_correlations.csv"
    if correlation_path.exists():
        correlations = pd.read_csv(correlation_path)
        target = correlations[
            (
                correlations.variable_1.str.contains("change|progression", case=False, regex=True)
                & correlations.variable_2.str.contains(
                    "perfusion|cvr|physiology", case=False, regex=True
                )
            )
            | (
                correlations.variable_2.str.contains("change|progression", case=False, regex=True)
                & correlations.variable_1.str.contains(
                    "perfusion|cvr|physiology", case=False, regex=True
                )
            )
        ]
        if target.empty:
            lines.append("- Physiology–progression correlations were unavailable.")
        else:
            passing = target[target.pearson_q_bh < 0.05]
            if passing.empty:
                lines.append(
                    f"- None of {len(target)} physiology–progression Pearson correlations passed BH control at q < 0.05."
                )
            else:
                best = passing.sort_values("pearson_q_bh").iloc[0]
                lines.append(
                    f"- {len(passing)} of {len(target)} physiology–progression Pearson correlations passed BH control; "
                    f"the smallest-q pair was {best.variable_1} with {best.variable_2} "
                    f"(r = {best.pearson_r:.3g}, 95% bootstrap CI {best.pearson_boot_ci_low:.3g} to "
                    f"{best.pearson_boot_ci_high:.3g}, q = {best.pearson_q_bh:.3g}, n = {int(best.n)})."
                )
    for filename, label in (
        ("age_relationships.csv", "Age"),
        ("wmh_burden_relationships.csv", "Baseline WMH burden"),
    ):
        path = root / filename
        if not path.exists() or path.stat().st_size <= 2:
            lines.append(f"- {label}: no model was estimable with the available variables.")
            continue
        table = pd.read_csv(path)
        estimated = (
            table[table.get("status", "").eq("estimated")]
            if not table.empty and "status" in table
            else pd.DataFrame()
        )
        if estimated.empty:
            lines.append(f"- {label}: no model was estimable with the available complete cases.")
        else:
            supported = estimated[estimated.q_bh < 0.05]
            if supported.empty:
                lines.append(
                    f"- {label}: none of {len(estimated)} exploratory associations passed BH control at q < 0.05."
                )
            else:
                names = ", ".join(supported.outcome.astype(str))
                lines.append(
                    f"- {label}: {len(supported)} of {len(estimated)} associations passed BH control at q < 0.05 ({names}); see the table for directions and intervals."
                )
    lines += [
        "",
        "## Interpretation guardrails",
        "",
        "These observational associations do not establish causality. Effect estimates, confidence intervals, exact p values, adjusted q values, sample sizes, and non-estimable models should be interpreted together; null findings are retained.",
        "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")
    return output
