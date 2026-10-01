from vascular_edge.cli import build_parser, main


def test_cli_version(capsys):
    try:
        main(["--version"])
    except SystemExit as exc:
        assert exc.code == 0
    assert "vascular-edge" in capsys.readouterr().out


def test_perilesional_commands_are_registered():
    parser = build_parser()
    analyze = parser.parse_args(
        [
            "analyze-perilesional",
            "--wmh",
            "w.nii.gz",
            "--wm",
            "wm.nii.gz",
            "--flair",
            "f.nii.gz",
            "--output-dir",
            "out",
            "--subject",
            "sub-1",
            "--session",
            "ses-1",
        ]
    )
    plot = parser.parse_args(
        ["plot-perilesional", "--table", "features.csv", "--output-dir", "figures"]
    )
    assert analyze.command == "analyze-perilesional"
    assert plot.command == "plot-perilesional"


def test_longitudinal_commands_are_registered():
    parser = build_parser()
    args = parser.parse_args(
        [
            "analyze-longitudinal",
            "--baseline-t1",
            "b1.nii.gz",
            "--baseline-flair",
            "bf.nii.gz",
            "--baseline-wmh",
            "bw.nii.gz",
            "--baseline-wm",
            "wm.nii.gz",
            "--followup-t1",
            "f1.nii.gz",
            "--followup-flair",
            "ff.nii.gz",
            "--followup-wmh",
            "fw.nii.gz",
            "--output-dir",
            "out",
            "--subject",
            "sub-1",
            "--baseline-session",
            "ses-wave1",
            "--followup-session",
            "ses-wave2",
        ]
    )
    qc = parser.parse_args(["qc-longitudinal", "--provenance", "record.json"])
    assert args.command == "analyze-longitudinal" and qc.command == "qc-longitudinal"


def test_cohort_statistics_commands_are_registered():
    parser = build_parser()
    build = parser.parse_args(
        [
            "build-analysis-table",
            "--bids-root",
            "bids",
            "--derivatives-root",
            "deriv",
            "--output-dir",
            "out",
        ]
    )
    stats = parser.parse_args(["run-statistics", "--table", "table.csv", "--output-dir", "stats"])
    results = parser.parse_args(
        ["build-results", "--statistics-dir", "stats", "--output", "results.md"]
    )
    assert (build.command, stats.command, results.command) == (
        "build-analysis-table",
        "run-statistics",
        "build-results",
    )


def test_end_to_end_run_command_is_registered():
    args = build_parser().parse_args(
        ["run", "--config", "configs/analysis.yaml", "--demo", "--force"]
    )
    assert args.command == "run"
    assert args.demo and args.force
