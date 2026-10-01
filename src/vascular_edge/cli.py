"""Command-line entry point for the Vascular Edge workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .cohort import (
    discover_openneuro,
    download_with_datalad,
    inventory_from_dicts,
    inventory_to_dicts,
    select_cohort,
    write_manifest,
)
from .config import load_config
from .discovery import discover_bids
from .features import export_features
from .perilesional import make_rings, summarize_rings


def _write_inventory(frame: pd.DataFrame, output: str | None) -> None:
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output, index=False)
    print(frame.to_string(index=False) if not frame.empty else "No BIDS imaging files found.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vascular-edge", description=__doc__)
    parser.add_argument("--version", action="version", version="vascular-edge 0.1.0")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("discover", help="Inventory local BIDS or remote OpenNeuro metadata")
    p.add_argument("--bids-root")
    p.add_argument("--remote", action="store_true")
    p.add_argument("--config")
    p.add_argument("--output")
    p = sub.add_parser("select-cohort", help="Deterministically tier and cap a saved inventory")
    p.add_argument("--inventory", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--output-stem")
    p = sub.add_parser("download", help="Selectively retrieve included manifest paths with DataLad")
    p.add_argument("--manifest", required=True)
    p.add_argument("--bids-root", required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-subjects", type=int)
    p.add_argument("--backend", choices=["datalad"], default="datalad")
    p = sub.add_parser(
        "validate-data", help="Validate downloaded manifest files without loading full arrays"
    )
    p.add_argument("--manifest", required=True)
    p.add_argument("--bids-root", required=True)
    p.add_argument("--output")
    p = sub.add_parser(
        "preprocess-structural", help="Canonicalize, N4-correct, register, and segment tissues"
    )
    p.add_argument("--t1", required=True)
    p.add_argument("--flair", required=True)
    p.add_argument("--derivatives-root", required=True)
    p.add_argument("--subject", required=True)
    p.add_argument("--session", required=True)
    p.add_argument("--config")
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("segment-wmh", help="Run configured WMH inference and postprocessing")
    p.add_argument("--t1", required=True)
    p.add_argument("--flair", required=True)
    p.add_argument("--brain-mask", required=True)
    p.add_argument("--wm-mask", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--subject", required=True)
    p.add_argument("--session", required=True)
    p.add_argument("--config")
    p.add_argument("--backend", choices=["fallback", "truenet"])
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("qc-structural", help="Create structural montage and QC JSON")
    p.add_argument("--t1", required=True)
    p.add_argument("--flair", required=True)
    p.add_argument("--wmh", required=True)
    p.add_argument("--csf", required=True)
    p.add_argument("--gm", required=True)
    p.add_argument("--wm", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--subject", required=True)
    p.add_argument("--session", required=True)
    p.add_argument("--registration-metric", type=float)
    p.add_argument("--wmh-backend", choices=["fallback", "truenet"])
    p.add_argument("--force", action="store_true")
    p = sub.add_parser(
        "process-perfusion", help="Metadata-gated ASL differencing and quantification"
    )
    p.add_argument("--asl", required=True)
    p.add_argument("--metadata", required=True)
    p.add_argument("--context")
    p.add_argument("--m0")
    p.add_argument("--t1", required=True)
    p.add_argument("--gm", required=True)
    p.add_argument("--wm", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--subject", required=True)
    p.add_argument("--session", required=True)
    p.add_argument("--config")
    p.add_argument("--no-motion-correction", action="store_true")
    p.add_argument("--force", action="store_true")
    p = sub.add_parser(
        "process-cvr", help="Analyze hypercapnia BOLD only with an explicit regressor"
    )
    p.add_argument("--bold", required=True)
    p.add_argument("--metadata", required=True)
    p.add_argument("--t1", required=True)
    p.add_argument("--regressor")
    p.add_argument("--regressor-column", default="stimulus")
    p.add_argument("--trace-units")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--subject", required=True)
    p.add_argument("--session", required=True)
    p.add_argument("--no-motion-correction", action="store_true")
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("qc-physiology", help="Create perfusion/CVR QC montage and metrics")
    p.add_argument("--t1", required=True)
    p.add_argument("--flair", required=True)
    p.add_argument("--gm", required=True)
    p.add_argument("--wm", required=True)
    p.add_argument("--perfusion")
    p.add_argument("--perfusion-units")
    p.add_argument("--raw-asl")
    p.add_argument("--registered-reference")
    p.add_argument("--cvr")
    p.add_argument("--cvr-units")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--subject", required=True)
    p.add_argument("--session", required=True)
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("perilesional", help="Create rings and optional image summaries")
    p.add_argument("--lesion-mask", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--ring-width-mm", type=float, default=2.0)
    p.add_argument("--max-distance-mm", type=float, default=10.0)
    p.add_argument("--image")
    p.add_argument("--summary")
    p = sub.add_parser("analyze-perilesional", help="Quantify physical-distance WMH neighborhoods")
    p.add_argument("--wmh", required=True)
    p.add_argument("--wm", required=True)
    p.add_argument("--flair", required=True)
    p.add_argument("--csf")
    p.add_argument("--perfusion")
    p.add_argument("--perfusion-units", default="unknown")
    p.add_argument("--cvr")
    p.add_argument("--cvr-units", default="unknown")
    p.add_argument("--coverage-mask")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--subject", required=True)
    p.add_argument("--session", required=True)
    p.add_argument("--config")
    p.add_argument("--parquet", action="store_true")
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("plot-perilesional", help="Plot cohort perilesional feature tables")
    p.add_argument("--table", required=True)
    p.add_argument("--output-dir", required=True)
    p = sub.add_parser(
        "analyze-longitudinal", help="Register follow-up and analyze future WMH tissue"
    )
    p.add_argument("--baseline-t1", required=True)
    p.add_argument("--baseline-flair", required=True)
    p.add_argument("--baseline-wmh", required=True)
    p.add_argument("--baseline-wm", required=True)
    p.add_argument("--followup-t1", required=True)
    p.add_argument("--followup-flair", required=True)
    p.add_argument("--followup-wmh", required=True)
    p.add_argument("--baseline-csf")
    p.add_argument("--baseline-coverage")
    p.add_argument("--baseline-wmh-provenance")
    p.add_argument("--followup-wmh-provenance")
    p.add_argument("--baseline-perfusion")
    p.add_argument("--perfusion-units", default="unknown")
    p.add_argument("--baseline-cvr")
    p.add_argument("--cvr-units", default="unknown")
    p.add_argument("--demographics")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--subject", required=True)
    p.add_argument("--baseline-session", required=True)
    p.add_argument("--followup-session", required=True)
    p.add_argument("--config")
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("qc-longitudinal", help="Verify longitudinal outputs and stored QC")
    p.add_argument("--provenance", required=True)
    p.add_argument("--output")
    p = sub.add_parser("qc", help="Create a lesion-mask overlay PNG")
    p.add_argument("--image", required=True)
    p.add_argument("--mask", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("export-features", help="Copy a tidy feature CSV to a requested path")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser(
        "build-analysis-table", help="Harmonize metadata, cognition, and imaging features"
    )
    p.add_argument("--bids-root", required=True)
    p.add_argument("--derivatives-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--config")
    p = sub.add_parser(
        "run-statistics", help="Generate cohort descriptive and exploratory statistics"
    )
    p.add_argument("--table", required=True)
    p.add_argument("--profiles")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--config")
    p = sub.add_parser("build-results", help="Build an evidence-bound Markdown results section")
    p.add_argument("--statistics-dir", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("run", help="Run the complete resumable analysis workflow")
    p.add_argument("--config", required=True)
    p.add_argument("--demo", action="store_true")
    p.add_argument(
        "--force", action="store_true", help="Recompute resumable subject-level derivatives"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "discover":
        if args.remote:
            config = load_config(args.config)
            dataset = config.get("dataset", {})
            inventory = inventory_to_dicts(
                discover_openneuro(dataset.get("id", "ds004856"), dataset.get("version"))
            )
            if args.output:
                Path(args.output).parent.mkdir(parents=True, exist_ok=True)
                Path(args.output).write_text(json.dumps(inventory, indent=2), encoding="utf-8")
            print(
                f"Discovered {len(inventory)} remote subjects; metadata only (no image bytes transferred)."
            )
        elif args.bids_root:
            _write_inventory(discover_bids(args.bids_root), args.output)
        else:
            raise SystemExit("discover requires --bids-root or --remote.")
    elif args.command == "select-cohort":
        config = load_config(args.config)
        cohort = config.get("cohort", {})
        inventory = inventory_from_dicts(
            json.loads(Path(args.inventory).read_text(encoding="utf-8"))
        )
        rows = select_cohort(
            inventory,
            int(cohort.get("target_size", 30)),
            float(cohort.get("max_download_gb", 80)),
            int(cohort.get("random_seed", 2026)),
            bool(cohort.get("include_cvr", False)),
        )
        output = args.output_stem or config.get("paths", {}).get(
            "manifest_stem", "results/cohort_manifest"
        )
        paths = write_manifest(rows, output)
        print(
            f"Selected {sum(r['included'] for r in rows)} subjects; wrote {paths[0]} and {paths[1]}."
        )
    elif args.command == "download":
        rows = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        if args.max_subjects is not None:
            kept = 0
            for row in rows:
                if row.get("included"):
                    kept += 1
                    if kept > args.max_subjects:
                        row["included"] = False
        paths = download_with_datalad(rows, args.bids_root, args.dry_run)
        bytes_total = sum(
            f["size_bytes"] for r in rows if r.get("included") for f in r.get("files", [])
        )
        print(
            f"{'Would retrieve' if args.dry_run else 'Retrieved'} {len(paths)} files ({bytes_total / 1024**3:.2f} GiB estimated)."
        )
        if args.dry_run:
            print("\n".join(paths))
    elif args.command == "validate-data":
        from .validation import validate_selected_data

        findings = validate_selected_data(
            args.bids_root, json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        )
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(findings).to_csv(args.output, index=False)
        invalid = sum(not item["valid"] for item in findings)
        print(f"Validated {len(findings)} files; {invalid} invalid.")
        if invalid:
            return 1
    elif args.command == "preprocess-structural":
        from .preprocessing import preprocess_structural

        config = load_config(args.config)
        structural = config.get("structural", {})
        record = preprocess_structural(
            args.t1,
            args.flair,
            args.derivatives_root,
            args.subject,
            args.session,
            args.force,
            int(structural.get("n4_shrink_factor", 2)),
        )
        print(json.dumps(record, indent=2))
    elif args.command == "segment-wmh":
        from .segmentation import segment_wmh_files

        config = load_config(args.config)
        segmentation = config.get("segmentation", {})
        backend = args.backend or segmentation.get("backend", "fallback")
        record = segment_wmh_files(
            args.flair,
            args.t1,
            args.brain_mask,
            args.wm_mask,
            args.output_dir,
            args.subject,
            args.session,
            backend,
            args.force,
            float(segmentation.get("lesion_z_threshold", 2.5)),
            float(segmentation.get("min_component_mm3", 3.0)),
            segmentation.get("model_dir"),
            segmentation.get("executable", "truenet"),
        )
        print(json.dumps(record, indent=2))
    elif args.command == "qc-structural":
        from .qc import structural_qc

        record = structural_qc(
            args.t1,
            args.flair,
            {"csf": args.csf, "gm": args.gm, "wm": args.wm},
            args.wmh,
            args.output_dir,
            args.subject,
            args.session,
            args.registration_metric,
            args.force,
            args.wmh_backend,
        )
        print(json.dumps(record, indent=2))
    elif args.command == "process-perfusion":
        from .perfusion import process_perfusion

        config = load_config(args.config)
        physiology = config.get("perfusion", {})
        parameters = {
            key: float(physiology[key])
            for key in ("labeling_efficiency", "blood_t1_s", "blood_brain_partition")
            if key in physiology
        }
        record = process_perfusion(
            args.asl,
            args.metadata,
            args.t1,
            args.gm,
            args.wm,
            args.output_dir,
            args.subject,
            args.session,
            args.context,
            args.m0,
            not args.no_motion_correction,
            args.force,
            parameters,
        )
        print(json.dumps(record, indent=2))
    elif args.command == "process-cvr":
        from .cvr import process_cvr

        record = process_cvr(
            args.bold,
            args.metadata,
            args.t1,
            args.output_dir,
            args.subject,
            args.session,
            args.regressor,
            args.regressor_column,
            args.trace_units,
            not args.no_motion_correction,
            args.force,
        )
        print(json.dumps(record, indent=2))
    elif args.command == "qc-physiology":
        from .physiology_qc import qc_physiology

        record = qc_physiology(
            args.t1,
            args.flair,
            args.perfusion,
            args.gm,
            args.wm,
            args.output_dir,
            args.subject,
            args.session,
            args.perfusion_units,
            args.raw_asl,
            args.cvr,
            args.cvr_units,
            args.force,
            args.registered_reference,
        )
        print(json.dumps(record, indent=2))
    elif args.command == "perilesional":
        import nibabel as nib

        lesion = nib.load(args.lesion_mask)
        rings = make_rings(
            lesion.get_fdata() > 0,
            lesion.header.get_zooms()[:3],
            args.ring_width_mm,
            args.max_distance_mm,
        )
        nib.save(nib.Nifti1Image(rings, lesion.affine, lesion.header), args.output)
        if args.image and args.summary:
            records = summarize_rings(nib.load(args.image).get_fdata(dtype="float32"), rings)
            Path(args.summary).write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(args.output)
    elif args.command == "analyze-perilesional":
        from .perilesional import analyze_files

        config = load_config(args.config)
        settings = config.get("perilesional", {})
        maps = {}
        units = {}
        if args.perfusion:
            maps["perfusion"] = args.perfusion
            units["perfusion"] = args.perfusion_units
        if args.cvr:
            maps["cvr"] = args.cvr
            units["cvr"] = args.cvr_units
        record = analyze_files(
            args.wmh,
            args.wm,
            args.flair,
            args.output_dir,
            args.subject,
            args.session,
            args.csf,
            maps,
            units,
            args.coverage_mask,
            settings.get("edges_mm", [0, 2, 4, 6, 10]),
            float(settings.get("min_lesion_volume_mm3", 3.0)),
            int(settings.get("min_ring_voxels", 20)),
            bool(settings.get("zero_is_missing", True)),
            args.parquet,
            args.force,
        )
        print(json.dumps(record, indent=2))
    elif args.command == "plot-perilesional":
        from .perilesional import plot_cohort

        frame = (
            pd.read_parquet(args.table)
            if str(args.table).lower().endswith(".parquet")
            else pd.read_csv(args.table)
        )
        print("\n".join(str(path) for path in plot_cohort(frame, args.output_dir)))
    elif args.command == "analyze-longitudinal":
        from .longitudinal import analyze_longitudinal_files

        settings = load_config(args.config).get("longitudinal", {})
        maps = {}
        units = {}
        if args.baseline_perfusion:
            maps["perfusion"] = args.baseline_perfusion
            units["perfusion"] = args.perfusion_units
        if args.baseline_cvr:
            maps["cvr"] = args.baseline_cvr
            units["cvr"] = args.cvr_units
        record = analyze_longitudinal_files(
            args.baseline_t1,
            args.baseline_flair,
            args.baseline_wmh,
            args.baseline_wm,
            args.followup_t1,
            args.followup_flair,
            args.followup_wmh,
            args.output_dir,
            args.subject,
            args.baseline_session,
            args.followup_session,
            maps,
            units,
            args.baseline_csf,
            args.baseline_coverage,
            args.demographics,
            float(settings.get("boundary_uncertainty_mm", 1.0)),
            float(settings.get("min_new_component_mm3", 3.0)),
            int(settings.get("min_conversion_voxels", 20)),
            float(settings.get("huge_volume_jump_percent", 200.0)),
            args.force,
            args.baseline_wmh_provenance,
            args.followup_wmh_provenance,
        )
        print(json.dumps(record, indent=2))
    elif args.command == "qc-longitudinal":
        from .longitudinal import qc_longitudinal

        record = qc_longitudinal(args.provenance)
        if args.output:
            Path(args.output).write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(json.dumps(record, indent=2))
        if record["status"] == "fail":
            return 1
    elif args.command == "qc":
        import nibabel as nib

        from .qc import save_mask_overlay

        print(
            save_mask_overlay(
                nib.load(args.image).get_fdata(dtype="float32"),
                nib.load(args.mask).get_fdata() > 0,
                args.output,
            )
        )
    elif args.command == "export-features":
        print(export_features(pd.read_csv(args.input).to_dict("records"), args.output))
    elif args.command == "build-analysis-table":
        from .statistics import build_analysis_table

        settings = load_config(args.config).get("analysis", {})
        print(
            json.dumps(
                build_analysis_table(
                    args.bids_root, args.derivatives_root, args.output_dir, settings
                ),
                indent=2,
            )
        )
    elif args.command == "run-statistics":
        from .statistics import run_statistics

        settings = load_config(args.config).get("analysis", {})
        print(
            json.dumps(
                run_statistics(args.table, args.output_dir, args.profiles, settings), indent=2
            )
        )
    elif args.command == "build-results":
        from .statistics import build_results

        print(build_results(args.statistics_dir, args.output))
    elif args.command == "run":
        from .workflow import run_pipeline

        print(json.dumps(run_pipeline(args.config, args.demo, args.force), indent=2))
    else:
        raise SystemExit(f"Unsupported command: {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
