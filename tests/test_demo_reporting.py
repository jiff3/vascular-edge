import json

import pandas as pd

from vascular_edge.demo import create_demo_bids, create_demo_cohort_derivatives
from vascular_edge.discovery import discover_bids
from vascular_edge.reporting import build_research_report


def test_demo_inputs_are_deterministic_and_discoverable(tmp_path):
    root = tmp_path / "bids"
    first = create_demo_bids(root, seed=17)
    second = create_demo_bids(root, seed=17)
    inventory = discover_bids(root)

    assert not first["resumed"]
    assert second["resumed"]
    assert len(inventory) == 2
    assert inventory.t1w.all() and inventory.flair.all()
    assert inventory.asl.sum() == 1 and inventory.bold.sum() == 1


def test_demo_cohort_contains_injected_effect(tmp_path):
    outputs = create_demo_cohort_derivatives(tmp_path, seed=2026)
    table = pd.read_csv(outputs["longitudinal"])

    effect = table["relative_perfusion_stable_nawm_mean"].corr(table["absolute_wmh_change_ml"])
    assert effect < -0.8
    assert len(table) == 48


def test_report_uses_relative_links_and_preserves_unavailable_results(tmp_path):
    figures = tmp_path / "figures"
    figures.mkdir()
    (figures / "flow.png").write_bytes(b"not-read-by-report")
    status = tmp_path / "pipeline_status.json"
    status.write_text(json.dumps({"steps": {"cvr_processing": {"status": "skipped"}}}))
    table = tmp_path / "table.csv"
    pd.DataFrame({"subject": ["sub-1"], "age": [60]}).to_csv(table, index=False)
    output = tmp_path / "report" / "report.md"

    build_research_report(
        {
            "analysis_table": str(table),
            "status_file": str(status),
            "figures_dir": str(figures),
            "statistics_dir": str(tmp_path / "statistics"),
            "figures": {"cohort_qc_flow": str(figures / "flow.png")},
        },
        output,
        demo=True,
    )
    text = output.read_text(encoding="utf-8")
    assert "../figures/flow.png" in text
    assert str(tmp_path.resolve()) not in text
    assert "cvr: unavailable or unsupported; no result inferred" in text
