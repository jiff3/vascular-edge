"""Resumable end-to-end orchestration for dataset and synthetic-demo analyses."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Callable, Mapping

import nibabel as nib
import pandas as pd

from .config import load_config
from .demo import create_demo_bids, create_demo_cohort_derivatives
from .discovery import discover_bids
from .segmentation import fallback_wmh_mask


def segment_flair_file(
    flair_file: str | Path, output_file: str | Path, z_threshold: float = 2.5, min_voxels: int = 3
) -> Path:
    """Create a uint8 fallback lesion mask from one FLAIR NIfTI."""
    image = nib.load(str(flair_file))
    mask = fallback_wmh_mask(
        image.get_fdata(dtype="float32"), z_threshold=z_threshold, min_voxels=min_voxels
    )
    output = Path(output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(mask.astype("uint8"), image.affine, image.header), str(output))
    return output


class PipelineRunner:
    """Persist step outcomes and continue past scientifically unavailable steps."""

    def __init__(self, output_root: Path, force: bool = False) -> None:
        self.output_root = output_root
        self.force = force
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.status_file = output_root / "pipeline_status.json"
        self.state: dict[str, Any] = (
            json.loads(self.status_file.read_text(encoding="utf-8"))
            if self.status_file.exists()
            else {"steps": {}}
        )
        self.log = logging.getLogger("vascular_edge.workflow")

    def _save(self) -> None:
        self.status_file.write_text(json.dumps(self.state, indent=2, default=str), encoding="utf-8")

    def step(self, name: str, action: Callable[[], Any], required: bool = True) -> Any:
        self.log.info("step=%s state=starting", name)
        try:
            result = action()
            resumed = isinstance(result, Mapping) and bool(result.get("resumed"))
            status = "resumed" if resumed else "complete"
            self.state["steps"][name] = {"status": status, "required": required}
            self._save()
            self.log.info("step=%s state=%s", name, status)
            return result
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            status = "failed" if required else "skipped"
            self.state["steps"][name] = {"status": status, "required": required, "reason": str(exc)}
            self._save()
            self.log.warning("step=%s state=%s reason=%s", name, status, exc)
            if required:
                raise
            return None


def _configure_logging(output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    log_file = output_root / "pipeline.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file, encoding="utf-8")],
        force=True,
    )


def _normalize_entities(row: pd.Series) -> tuple[str, str]:
    subject = str(row.subject)
    session = str(row.session)
    return (
        subject if subject.startswith("sub-") else f"sub-{subject}",
        session if session.startswith("ses-") else f"ses-{session}",
    )


def _select_local_cohort(inventory: pd.DataFrame, target_size: int) -> pd.DataFrame:
    candidates = inventory[inventory.t1w & inventory.flair].copy()
    subjects = sorted(candidates.subject.astype(str).unique())[:target_size]
    return candidates[candidates.subject.astype(str).isin(subjects)].sort_values(
        ["subject", "session"]
    )


def _local_manifest(selected: pd.DataFrame, root: Path) -> list[dict]:
    rows = []
    for subject, group in selected.groupby("subject"):
        files = []
        for _, row in group.iterrows():
            for modality in ("t1w", "flair", "asl", "bold"):
                name = f"{modality}_file"
                if pd.notna(row.get(name)):
                    path = str(row[name])
                    files.append({"path": path, "size_bytes": (root / path).stat().st_size})
        rows.append(
            {
                "subject": f"sub-{subject}"
                if not str(subject).startswith("sub-")
                else str(subject),
                "included": True,
                "inclusion_reason": "local structural inputs available",
                "files": files,
            }
        )
    return rows


def run_pipeline(
    config_file: str | Path, demo: bool = False, force: bool = False
) -> dict[str, Any]:
    """Execute the configured pipeline and return its complete artifact manifest.

    Unsupported optional modalities are recorded as skipped. Required structural,
    table, statistics, figure, and report failures stop the run with a clear error.
    """
    config_path = Path(config_file)
    config = load_config(config_path)
    paths = config.get("paths", {})
    seed = int(config.get("analysis", {}).get("random_seed", 2026))
    output_root = Path(
        paths.get("demo_root" if demo else "run_root", "results/demo" if demo else "results/run")
    )
    if not output_root.is_absolute():
        output_root = (config_path.parent.parent / output_root).resolve()
    _configure_logging(output_root)
    shutil.copy2(config_path, output_root / "analysis_config.yaml")
    runner = PipelineRunner(output_root, force)
    bids_root = Path(paths.get("bids_root", "data/ds004856"))
    if demo:
        bids_root = output_root / "synthetic_bids"
        runner.step("demo_input_generation", lambda: create_demo_bids(bids_root, seed))
    elif not bids_root.is_absolute():
        bids_root = (config_path.parent.parent / bids_root).resolve()
    derivatives = output_root / "derivatives"
    cohort_dir = output_root / "cohort"
    statistics_dir = output_root / "statistics"
    figures_dir = output_root / "figures"
    report_dir = output_root / "report"
    inventory = runner.step("cohort_discovery", lambda: discover_bids(bids_root))
    selected = runner.step(
        "cohort_selection",
        lambda: _select_local_cohort(
            inventory, int(config.get("cohort", {}).get("target_size", 30))
        ),
    )
    if selected is None or selected.empty:
        raise RuntimeError("No subject/session has both T1w and FLAIR inputs.")
    manifest = _local_manifest(selected, bids_root)
    (output_root / "cohort_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    from .validation import validate_selected_data

    validation = runner.step(
        "input_validation", lambda: validate_selected_data(bids_root, manifest)
    )
    pd.DataFrame(validation).to_csv(output_root / "input_validation.csv", index=False)
    if any(not row["valid"] for row in validation):
        raise RuntimeError("Input validation failed; inspect input_validation.csv.")
    structural: dict[tuple[str, str], dict] = {}
    segmentations: dict[tuple[str, str], dict] = {}
    from .preprocessing import preprocess_structural

    structural_settings = config.get("structural", {})

    def structural_action() -> dict:
        resumed = True
        for _, row in selected.iterrows():
            subject, session = _normalize_entities(row)
            record = preprocess_structural(
                bids_root / row.t1w_file,
                bids_root / row.flair_file,
                derivatives,
                subject,
                session,
                force,
                int(structural_settings.get("n4_shrink_factor", 2)),
            )
            structural[(subject, session)] = record
            resumed &= bool(record.get("resumed"))
        return {"resumed": resumed, "count": len(structural)}

    runner.step("structural_preprocessing", structural_action)
    from .segmentation import segment_wmh_files

    segmentation_settings = config.get("segmentation", {})

    def segmentation_action() -> dict:
        resumed = True
        for key, pre in structural.items():
            subject, session = key
            record = segment_wmh_files(
                pre["outputs"]["flair_registered"],
                pre["outputs"]["t1_n4"],
                pre["outputs"]["brain_mask"],
                pre["tissues"]["wm"],
                Path(pre["outputs"]["t1_n4"]).parent,
                subject,
                session,
                segmentation_settings.get("backend", "fallback"),
                force,
                float(segmentation_settings.get("lesion_z_threshold", 2.5)),
                float(segmentation_settings.get("min_component_mm3", 3.0)),
                segmentation_settings.get("model_dir"),
                segmentation_settings.get("executable", "truenet"),
            )
            segmentations[key] = record
            resumed &= bool(record.get("resumed"))
        return {"resumed": resumed, "count": len(segmentations)}

    runner.step("wmh_segmentation", segmentation_action)
    from .qc import structural_qc

    structural_qcs: dict[tuple[str, str], dict] = {}

    def structural_qc_action() -> dict:
        resumed = True
        for key, pre in structural.items():
            subject, session = key
            record = structural_qc(
                pre["outputs"]["t1_n4"],
                pre["outputs"]["flair_registered"],
                pre["tissues"],
                segmentations[key]["final_mask"],
                derivatives / subject / session / "qc",
                subject,
                session,
                pre["registration"].get("metric"),
                force,
                segmentations[key]["backend"],
            )
            structural_qcs[key] = record
            resumed &= bool(record.get("resumed"))
        return {"resumed": resumed, "count": len(structural_qcs)}

    runner.step("structural_qc", structural_qc_action)
    first_key = sorted(structural, key=lambda x: (x[0], x[1]))[0]
    first_subject, first_session = first_key
    pre = structural[first_key]
    selected_rows = {_normalize_entities(row): row for _, row in selected.iterrows()}
    perfusion_records: dict[tuple[str, str], dict] = {}
    cvr_records: dict[tuple[str, str], dict] = {}

    def perfusion_action() -> dict:
        from .perfusion import process_perfusion

        for key, row in selected_rows.items():
            if not bool(row.get("asl", False)):
                continue
            subject, session = key
            asl_path = bids_root / row.asl_file
            metadata = Path(str(asl_path).replace(".nii.gz", ".json").replace(".nii", ".json"))
            context = asl_path.parent / asl_path.name.replace(
                "_asl.nii.gz", "_aslcontext.tsv"
            ).replace("_asl.nii", "_aslcontext.tsv")
            item = structural[key]
            perfusion_records[key] = process_perfusion(
                asl_path,
                metadata,
                item["outputs"]["t1_n4"],
                item["tissues"]["gm"],
                item["tissues"]["wm"],
                derivatives / subject / session / "perf",
                subject,
                session,
                context if context.exists() else None,
                motion_correct=bool(config.get("perfusion", {}).get("motion_correction", True))
                and not demo,
                force=force,
            )
        if not perfusion_records:
            raise ValueError("ASL unavailable in the selected cohort")
        return {
            "count": len(perfusion_records),
            "resumed": all(x.get("resumed") for x in perfusion_records.values()),
        }

    runner.step("perfusion_processing", perfusion_action, required=False)

    def cvr_action() -> dict:
        from .cvr import process_cvr

        for key, row in selected_rows.items():
            bold_file = row.get("bold_file")
            if pd.isna(bold_file) or not any(
                token in str(bold_file).lower() for token in ("hypercap", "cvr", "co2")
            ):
                continue
            subject, session = key
            bold_path = bids_root / str(bold_file)
            metadata = Path(str(bold_path).replace(".nii.gz", ".json").replace(".nii", ".json"))
            events = bold_path.parent / bold_path.name.replace(
                "_bold.nii.gz", "_events.tsv"
            ).replace("_bold.nii", "_events.tsv")
            cvr_records[key] = process_cvr(
                bold_path,
                metadata,
                structural[key]["outputs"]["t1_n4"],
                derivatives / subject / session / "cvr",
                subject,
                session,
                events if events.exists() else None,
                motion_correct=False
                if demo
                else bool(config.get("cvr", {}).get("motion_correction", True)),
                force=force,
            )
        if not cvr_records:
            raise ValueError("documented CVR acquisition unavailable in the selected cohort")
        return {
            "count": len(cvr_records),
            "resumed": all(x.get("resumed") for x in cvr_records.values()),
        }

    runner.step("cvr_processing", cvr_action, required=False)

    from .physiology_qc import qc_physiology

    physiology_qcs: dict[tuple[str, str], dict] = {}

    def physiology_qc_action() -> dict:
        for key in sorted(set(perfusion_records) | set(cvr_records)):
            subject, session = key
            item = structural[key]
            perfusion_record = perfusion_records.get(key, {})
            cvr_record = cvr_records.get(key, {})
            row = selected_rows[key]
            physiology_qcs[key] = qc_physiology(
                item["outputs"]["t1_n4"],
                item["outputs"]["flair_registered"],
                perfusion_record.get("outputs", {}).get("t1_map"),
                item["tissues"]["gm"],
                item["tissues"]["wm"],
                derivatives / subject / session / "qc",
                subject,
                session,
                perfusion_record.get("assessment", {}).get("units"),
                bids_root / row.asl_file if bool(row.get("asl", False)) else None,
                cvr_record.get("outputs", {}).get("t1_map"),
                cvr_record.get("units"),
                force,
                perfusion_record.get("outputs", {}).get("registered_control"),
            )
        if not physiology_qcs:
            raise ValueError("no processed physiology maps were available for QC")
        return {
            "count": len(physiology_qcs),
            "resumed": all(x.get("resumed") for x in physiology_qcs.values()),
        }

    runner.step("physiology_qc", physiology_qc_action, required=False)

    from .perilesional import analyze_files

    perilesional_settings = config.get("perilesional", {})
    perilesional_records: dict[tuple[str, str], dict] = {}

    def perilesional_action() -> dict:
        for key, item in structural.items():
            subject, session = key
            perfusion_record = perfusion_records.get(key, {})
            cvr_record = cvr_records.get(key, {})
            maps: dict[str, str] = {}
            units: dict[str, str] = {}
            if perfusion_record.get("outputs"):
                maps["relative_perfusion"] = perfusion_record["outputs"]["t1_map"]
                units["relative_perfusion"] = perfusion_record["assessment"]["units"]
            if cvr_record.get("outputs"):
                maps["cvr_percent_bold"] = cvr_record["outputs"]["t1_map"]
                units["cvr_percent_bold"] = cvr_record["units"]
            perilesional_records[key] = analyze_files(
                segmentations[key]["final_mask"],
                item["tissues"]["wm"],
                item["outputs"]["flair_registered"],
                derivatives / subject / session / "perilesional",
                subject,
                session,
                item["tissues"]["csf"],
                maps,
                units,
                edges_mm=perilesional_settings.get("edges_mm", [0, 2, 4, 6, 10]),
                min_lesion_volume_mm3=float(perilesional_settings.get("min_lesion_volume_mm3", 3)),
                min_ring_voxels=int(perilesional_settings.get("min_ring_voxels", 20)),
                force=force,
            )
        return {
            "count": len(perilesional_records),
            "resumed": all(x.get("resumed") for x in perilesional_records.values()),
        }

    runner.step("perilesional_analysis", perilesional_action)

    from .longitudinal import analyze_longitudinal_files

    longitudinal_records: dict[str, dict] = {}

    def longitudinal_action() -> dict:
        long_settings = config.get("longitudinal", {})
        for subject in sorted({key[0] for key in structural}):
            subject_sessions = sorted(
                [key for key in structural if key[0] == subject], key=lambda x: x[1]
            )
            if len(subject_sessions) < 2:
                continue
            base_key, follow_key = subject_sessions[0], subject_sessions[-1]
            base, follow = structural[base_key], structural[follow_key]
            perfusion_record = perfusion_records.get(base_key, {})
            cvr_record = cvr_records.get(base_key, {})
            maps: dict[str, str] = {}
            units: dict[str, str] = {}
            if perfusion_record.get("outputs"):
                maps["relative_perfusion"] = perfusion_record["outputs"]["t1_map"]
                units["relative_perfusion"] = perfusion_record["assessment"]["units"]
            if cvr_record.get("outputs"):
                maps["cvr_percent_bold"] = cvr_record["outputs"]["t1_map"]
                units["cvr_percent_bold"] = cvr_record["units"]
            longitudinal_records[subject] = analyze_longitudinal_files(
                base["outputs"]["t1_n4"],
                base["outputs"]["flair_registered"],
                segmentations[base_key]["final_mask"],
                base["tissues"]["wm"],
                follow["outputs"]["t1_n4"],
                follow["outputs"]["flair_registered"],
                segmentations[follow_key]["final_mask"],
                derivatives / subject / "longitudinal",
                subject,
                base_key[1],
                follow_key[1],
                maps,
                units,
                base["tissues"]["csf"],
                boundary_uncertainty_mm=float(long_settings.get("boundary_uncertainty_mm", 1.0)),
                min_new_component_mm3=float(long_settings.get("min_new_component_mm3", 3.0)),
                min_conversion_voxels=int(long_settings.get("min_conversion_voxels", 20)),
                force=force,
            )
        if not longitudinal_records:
            raise ValueError("fewer than two structural sessions per subject")
        return {
            "count": len(longitudinal_records),
            "resumed": all(x.get("resumed") for x in longitudinal_records.values()),
        }

    runner.step("longitudinal_analysis", longitudinal_action, required=False)

    perfusion_record = perfusion_records.get(first_key, {})
    cvr_record = cvr_records.get(first_key, {})
    physiology_qc = physiology_qcs.get(first_key, {})
    perilesional_record = perilesional_records[first_key]
    longitudinal_record = longitudinal_records.get(first_subject, {})
    if demo:
        runner.step(
            "demo_cohort_generation",
            lambda: create_demo_cohort_derivatives(derivatives / "synthetic_cohort", seed),
        )
    from .statistics import build_analysis_table, build_results, run_statistics

    table_record = runner.step(
        "feature_table_creation",
        lambda: build_analysis_table(
            bids_root, derivatives, cohort_dir, config.get("analysis", {})
        ),
    )
    runner.step(
        "statistics",
        lambda: run_statistics(
            table_record["analysis_table"],
            statistics_dir,
            table_record["perilesional_profiles"],
            config.get("analysis", {}),
        ),
    )
    statistical_report = runner.step(
        "statistical_results",
        lambda: {
            "path": str(build_results(statistics_dir, statistics_dir / "statistical_results.md"))
        },
    )
    artifacts: dict[str, Any] = {
        "demo": demo,
        "bids_root": str(bids_root),
        "output_root": str(output_root),
        "status_file": str(runner.status_file),
        "structural": pre,
        "structural_qc": structural_qcs[first_key],
        "perfusion": perfusion_record or {},
        "cvr": cvr_record or {},
        "physiology_qc": physiology_qc or {},
        "perilesional": perilesional_record,
        "longitudinal": longitudinal_record or {},
        "analysis_table": table_record["analysis_table"],
        "statistics_dir": str(statistics_dir),
        "figures_dir": str(figures_dir),
        "statistical_report": statistical_report["path"],
        "seed": seed,
        "software_config": str(config_path.resolve()),
    }
    (output_root / "artifacts.json").write_text(
        json.dumps(artifacts, indent=2, default=str), encoding="utf-8"
    )
    from .reporting import build_research_report, build_showcase_figures

    figures = runner.step(
        "showcase_figures", lambda: build_showcase_figures(artifacts, figures_dir)
    )
    artifacts["figures"] = figures
    report = runner.step(
        "research_report",
        lambda: {
            "path": str(build_research_report(artifacts, report_dir / "research_report.md", demo))
        },
    )
    artifacts["report"] = report["path"]
    (output_root / "artifacts.json").write_text(
        json.dumps(artifacts, indent=2, default=str), encoding="utf-8"
    )
    summary = {name: item["status"] for name, item in runner.state["steps"].items()}
    logging.getLogger("vascular_edge.workflow").info("pipeline_summary=%s", summary)
    return {
        "status": "complete",
        "demo": demo,
        "output_root": str(output_root),
        "report": report["path"],
        "status_file": str(runner.status_file),
        "steps": summary,
        "artifacts": str(output_root / "artifacts.json"),
    }
