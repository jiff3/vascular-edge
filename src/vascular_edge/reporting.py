"""Publication-oriented showcase figures and evidence-bound Markdown reports."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd


def _image(path: str | Path) -> np.ndarray:
    return nib.load(str(path)).get_fdata(dtype=np.float32)


def _display(array: np.ndarray) -> np.ndarray:
    valid = array[np.isfinite(array) & (array != 0)]
    if not valid.size:
        return np.zeros_like(array)
    low, high = np.percentile(valid, (2, 98))
    return np.clip((array - low) / max(high - low, 1e-6), 0, 1)


def _masked_map(array: np.ndarray, brain_mask: np.ndarray) -> np.ndarray:
    """Mask physiology maps for display without altering analytic values."""
    return np.where(brain_mask, array, np.nan)


def _slice(array: np.ndarray, z: int) -> np.ndarray:
    return np.rot90(array[:, :, z])


def _copy(source: str | Path | None, destination: Path) -> str | None:
    if source is None or not Path(source).exists():
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return str(destination)


def build_showcase_figures(
    artifacts: Mapping[str, object], output_dir: str | Path
) -> dict[str, str]:
    """Create the eight documented showcase figures from completed outputs."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    figures: dict[str, str] = {}
    structural = artifacts.get("structural", {})
    perfusion = artifacts.get("perfusion", {})
    cvr = artifacts.get("cvr", {})
    perilesional = artifacts.get("perilesional", {})
    longitudinal = artifacts.get("longitudinal", {})
    try:
        t1 = _image(structural["outputs"]["t1_n4"])
        flair = _image(structural["outputs"]["flair_registered"])
        brain_mask = _image(structural["outputs"]["brain_mask"]) > 0
        perf = (
            _masked_map(_image(perfusion["outputs"]["t1_map"]), brain_mask)
            if perfusion.get("outputs")
            else None
        )
        cvr_map = (
            _masked_map(_image(cvr["outputs"]["t1_map"]), brain_mask)
            if cvr.get("outputs")
            else None
        )
        z = t1.shape[2] // 2
        panels = [
            ("T1-weighted", t1, "gray", "normalized signal"),
            ("FLAIR", flair, "gray", "normalized signal"),
        ]
        if perf is not None:
            panels.append(("Relative perfusion", perf, "magma", perfusion["assessment"]["units"]))
        if cvr_map is not None:
            panels.append(
                ("Hypercapnia response", cvr_map, "coolwarm", cvr.get("units", "unknown units"))
            )
        fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4))
        for ax, (title, data, cmap, units) in zip(np.atleast_1d(axes), panels):
            shown = _slice(_display(data), z) if cmap == "gray" else _slice(data, z)
            color_map = plt.get_cmap(cmap).copy()
            color_map.set_bad("black")
            image = ax.imshow(shown, cmap=color_map)
            ax.set_title(f"{title}\n{units}")
            ax.axis("off")
            if cmap != "gray":
                fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
        fig.suptitle("Synthetic multimodal MRI overview")
        fig.tight_layout()
        path = root / "01_multimodal_overview.png"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        figures["multimodal_overview"] = str(path)
    except (KeyError, FileNotFoundError):
        pass
    copied = _copy(
        artifacts.get("structural_qc", {}).get("figure"), root / "02_wmh_segmentation_qc.png"
    )
    if copied:
        figures["wmh_segmentation_qc"] = copied
    if perilesional.get("outputs") and structural.get("outputs"):
        regions = _image(perilesional["outputs"]["region_map"])
        flair = _image(structural["outputs"]["flair_registered"])
        lesion = regions == 0
        z_candidates = np.where(lesion.any(axis=(0, 1)))[0]
        z = (
            int(z_candidates[len(z_candidates) // 2])
            if len(z_candidates)
            else regions.shape[2] // 2
        )
        brain_mask = _image(structural["outputs"]["brain_mask"]) > 0
        for key, number, background, cmap, title, units in (
            (
                "flair_contours",
                "03",
                flair,
                "gray",
                "FLAIR with perilesional distance contours",
                "FLAIR signal",
            ),
            (
                "perfusion_contours",
                "04",
                _masked_map(_image(perfusion["outputs"]["t1_map"]), brain_mask),
                "magma",
                "Perfusion with perilesional distance contours",
                perfusion["assessment"]["units"],
            ),
        ):
            fig, ax = plt.subplots(figsize=(6.5, 5.5))
            shown = _slice(_display(background), z) if cmap == "gray" else _slice(background, z)
            color_map = plt.get_cmap(cmap).copy()
            color_map.set_bad("black")
            image = ax.imshow(shown, cmap=color_map)
            colors = ("cyan", "#ffd166", "#f8961e", "#ef476f", "#9b5de5")
            labels = ("WMH boundary", "2 mm", "4 mm", "6 mm", "10 mm")
            region_slice = _slice(regions, z)
            for threshold, color, label in zip((0, 1, 2, 3, 4), colors, labels):
                mask = (region_slice >= 0) & (region_slice <= threshold)
                if mask.any() and (~mask).any():
                    ax.contour(mask, [0.5], colors=[color], linewidths=1.1, linestyles="-")
                ax.plot([], [], color=color, label=label)
            if cmap != "gray":
                fig.colorbar(image, ax=ax, label=units)
            ax.set_title(title)
            ax.legend(loc="lower right", fontsize=8, framealpha=0.85)
            ax.axis("off")
            fig.tight_layout()
            path = root / f"{number}_{key}.png"
            fig.savefig(path, dpi=220, bbox_inches="tight")
            plt.close(fig)
            figures[key] = str(path)
    profile_source = (
        Path(str(artifacts.get("statistics_dir", "")))
        / "figures"
        / "physiology_distance_profiles.png"
    )
    copied = _copy(profile_source, root / "05_physiology_distance_profile.png")
    if copied:
        figures["physiology_distance_profile"] = copied
    copied = _copy(
        longitudinal.get("outputs", {}).get("figure"), root / "06_longitudinal_expansion.png"
    )
    if copied:
        figures["longitudinal_expansion"] = copied
    analysis_table = Path(str(artifacts.get("analysis_table", "")))
    if analysis_table.exists():
        table = pd.read_csv(analysis_table)
        columns = [
            x
            for x in ("relative_perfusion_converting_mean", "relative_perfusion_stable_nawm_mean")
            if x in table
        ]
        if len(columns) == 2:
            paired = table[columns].dropna()
            fig, ax = plt.subplots(figsize=(6.5, 5))
            for _, row in paired.iterrows():
                ax.plot([0, 1], row.values, color="#8da0cb", alpha=0.22, linewidth=0.8)
            ax.boxplot(
                [paired[columns[0]], paired[columns[1]]],
                tick_labels=["Later converting", "Stable NAWM"],
                showfliers=False,
            )
            ax.set_ylabel("Baseline relative perfusion (% mean control signal)")
            ax.set_title(
                f"Baseline physiology by future tissue status (synthetic; n={len(paired)})"
            )
            ax.grid(axis="y", alpha=0.2)
            fig.tight_layout()
            path = root / "07_converting_vs_stable.png"
            fig.savefig(path, dpi=220, bbox_inches="tight")
            plt.close(fig)
            figures["converting_vs_stable"] = str(path)
    status_file = Path(str(artifacts.get("status_file", "")))
    if status_file.exists():
        status = json.loads(status_file.read_text(encoding="utf-8"))
        steps = status.get("steps", {})
        colors = {
            "complete": "#2a9d8f",
            "resumed": "#457b9d",
            "skipped": "#e9c46a",
            "failed": "#e76f51",
        }
        fig, ax = plt.subplots(figsize=(10, max(4, 0.48 * len(steps))))
        names = list(steps)
        states = [steps[name].get("status", "skipped") for name in names]
        ax.barh(
            range(len(names)), np.ones(len(names)), color=[colors.get(x, "#adb5bd") for x in states]
        )
        ax.set_yticks(range(len(names)), [name.replace("_", " ") for name in names])
        ax.set_xlim(0, 1)
        ax.set_xticks([])
        ax.invert_yaxis()
        for i, state in enumerate(states):
            ax.text(
                0.5,
                i,
                state,
                ha="center",
                va="center",
                color="white" if state in {"complete", "resumed", "failed"} else "black",
                weight="bold",
            )
        ax.set_title("Pipeline and QC flow summary")
        [spine.set_visible(False) for spine in ax.spines.values()]
        fig.tight_layout()
        path = root / "08_cohort_qc_flow.png"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        figures["cohort_qc_flow"] = str(path)
    captions = {
        "multimodal_overview": "Co-registered structural and physiology images from the deterministic synthetic phantom.",
        "wmh_segmentation_qc": "Structural preprocessing and WMH segmentation quality-control montage.",
        "flair_contours": "Visible WMH and physical-distance neighborhoods over FLAIR.",
        "perfusion_contours": "The same physical-distance neighborhoods over relative perfusion.",
        "physiology_distance_profile": "Cohort physiology summaries by distance; intervals describe uncertainty and do not imply causality.",
        "longitudinal_expansion": "Follow-up lesion expansion registered to baseline with converting tissue highlighted.",
        "converting_vs_stable": "Baseline physiology in tissue classified by future lesion status in the injected-effect demo.",
        "cohort_qc_flow": "Completion state for each configured workflow component.",
    }
    (root / "captions.json").write_text(
        json.dumps({key: captions[key] for key in figures}, indent=2), encoding="utf-8"
    )
    return figures


def build_research_report(
    artifacts: Mapping[str, object], output_file: str | Path, demo: bool = False
) -> Path:
    """Render a cohort report from observed tables, QC records, and estimates."""
    output = Path(output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    table_path = Path(str(artifacts.get("analysis_table", "")))
    table = pd.read_csv(table_path) if table_path.exists() else pd.DataFrame()
    figures = artifacts.get("figures", {})
    captions_path = Path(str(artifacts.get("figures_dir", ""))) / "captions.json"
    captions = (
        json.loads(captions_path.read_text(encoding="utf-8")) if captions_path.exists() else {}
    )

    def rel(path: str | Path) -> str:
        """Return a portable report-relative asset path."""
        return Path(os.path.relpath(Path(path).resolve(), output.parent.resolve())).as_posix()

    subjects = table.subject.nunique() if "subject" in table else 0
    lines = [
        "# Mapping the Vascular Edge — Research Report",
        "",
        "> **Synthetic demonstration; not empirical evidence.**"
        if demo
        else "> Results generated from the configured research dataset.",
        "",
        f"Generated from {subjects} analyzed subjects and {len(table)} subject-session rows. All statements below are assembled from saved outputs; unavailable analyses remain explicitly unavailable.",
        "",
    ]

    def section(title: str, keys: tuple[str, ...], text: str) -> None:
        lines.extend([f"## {title}", "", text, ""])
        for key in keys:
            if key in figures:
                lines.extend(
                    [f"![{title}]({rel(figures[key])})", "", f"*{captions.get(key, '')}*", ""]
                )

    section(
        "Cohort flow and modality availability",
        ("cohort_qc_flow",),
        "The flow figure records completed, resumed, skipped, and failed components. A skipped modality is not counted as a negative biological finding.",
    )
    modality = []
    for name in ("structural", "perfusion", "cvr", "perilesional", "longitudinal"):
        value = artifacts.get(name, {})
        modality.append(
            f"- {name}: {'available' if value and value.get('status', 'pass') != 'unsupported' else 'unavailable'}"
        )
    lines.extend(modality + [""])
    section(
        "Multimodal overview",
        ("multimodal_overview",),
        "Structural, perfusion, and CVR panels are shown only when their processed maps exist.",
    )
    section(
        "Exclusions and QC failures",
        ("wmh_segmentation_qc",),
        "Warnings and exclusions are retained in JSON/CSV provenance and are not silently removed.",
    )
    if not table.empty:
        age = (
            pd.to_numeric(table.get("age"), errors="coerce")
            if "age" in table
            else pd.Series(dtype=float)
        )
        burden = next(
            (
                x
                for x in ("baseline_wmh_volume_ml", "baseline_wmh_ml", "wmh_volume_ml")
                if x in table
            ),
            None,
        )
        summary = (
            f"Age was available for {age.notna().sum()} rows"
            + (
                f" (median {age.median():.1f} years, range {age.min():.1f}–{age.max():.1f})"
                if age.notna().any()
                else ""
            )
            + "."
        )
        if burden:
            summary += f" Baseline WMH burden was available for {table[burden].notna().sum()} rows (median {table[burden].median():.3g} mL)."
    else:
        summary = "No harmonized participant table was available."
    section("Participant summary and WMH burden", (), summary)
    section(
        "Perilesional analysis",
        ("flair_contours", "perfusion_contours", "physiology_distance_profile"),
        "Distance regions are defined in physical millimeters outside visible WMH and restricted to valid white matter and physiology coverage.",
    )
    section(
        "Longitudinal analysis",
        ("longitudinal_expansion", "converting_vs_stable"),
        "Baseline physiology is summarized in normal-appearing white matter that later converts to WMH and tissue that remains stable. Registration uncertainty near the baseline lesion boundary is excluded.",
    )
    lines.extend(["## Perfusion and CVR results", ""])
    for name in ("perfusion", "cvr"):
        record = artifacts.get(name, {})
        if record and record.get("status") != "unsupported":
            units = (
                record.get("assessment", {}).get("units")
                if name == "perfusion"
                else record.get("units")
            )
            lines.append(
                f"- {name}: processed; units `{units}`; status `{record.get('status', 'unknown')}`. Interpret warnings in provenance."
            )
        else:
            lines.append(f"- {name}: unavailable or unsupported; no result inferred.")
    lines.extend(["", "## Cognition analyses", ""])
    cognition = [x for x in table if x.startswith("cog_")] if not table.empty else []
    lines.append(
        f"{len(cognition)} harmonized cognitive variables were available. See model tables for estimability and adjusted results."
        if cognition
        else "No usable cognition was available; cognitive associations were not estimated."
    )
    lines.extend(["", "## Statistical estimates and uncertainty", ""])
    generated = Path(str(artifacts.get("statistics_dir", ""))) / "statistical_results.md"
    if generated.exists():
        statistical_lines = generated.read_text(encoding="utf-8").splitlines()
        if statistical_lines and statistical_lines[0].startswith("# "):
            statistical_lines = statistical_lines[1:]
        statistical_text = "\n".join(statistical_lines).replace("\n## ", "\n### ")
        lines.append(statistical_text)
    else:
        lines.append("No cohort statistical result document was generated.")
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- This is observational research software and does not establish causality or provide clinical interpretation.",
            "- The fallback WMH segmenter is transparent testing/demo machinery, not a validated clinical model.",
            "- Relative perfusion is not absolute CBF when calibration is absent.",
            "- Hypercapnia percent-BOLD response is not calibrated CVR without end-tidal CO2 in mmHg.",
            "- Small or selected cohorts can yield unstable estimates; every model-specific sample size and non-estimable model is retained.",
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")
    return output
