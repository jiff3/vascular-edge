from pathlib import Path

import numpy as np
import pandas as pd

from vascular_edge.metadata import harmonize_cognition, harmonize_participants
from vascular_edge.statistics import build_analysis_table, build_results, run_statistics


def _fixture(root: Path, effect: bool, seed: int = 9, n: int = 70):
    rng = np.random.default_rng(seed)
    bids = root / "bids"
    deriv = root / "deriv"
    (bids / "derivatives" / "cognition").mkdir(parents=True)
    deriv.mkdir()
    ids = [f"sub-{i:03d}" for i in range(n)]
    age = rng.uniform(40, 85, n)
    perf = rng.normal(50, 5, n)
    change = -0.12 * perf + rng.normal(0, 0.25, n) if effect else rng.normal(0, 1, n)
    pd.DataFrame(
        {
            "participant_id": ids,
            "AgeMRI_W1": age,
            "Sex": rng.choice(["f", "m"], n),
            "EduYrsEstCap": rng.integers(12, 21, n),
        }
    ).to_csv(bids / "participants.tsv", sep="\t", index=False)
    pd.DataFrame(
        {
            "participant_id": ids,
            "Wave": 1,
            "memory_total": rng.normal(size=n),
            "memory_delayed": rng.normal(size=n),
        }
    ).to_csv(bids / "derivatives" / "cognition" / "memory.csv", index=False)
    pd.DataFrame(
        {
            "subject": ids,
            "baseline_session": "ses-wave1",
            "followup_session": "ses-wave2",
            "baseline_wmh_ml": rng.lognormal(0, 0.4, n),
            "absolute_wmh_change_ml": change,
            "relative_perfusion_stable_nawm_mean": perf,
        }
    ).to_csv(deriv / "x_desc-longitudinal_subject.csv", index=False)
    profile = [
        {
            "subject": sid,
            "session": "ses-wave1",
            "scope": "subject",
            "region": region,
            "region_order": order,
            "map": "relative_perfusion",
            "units": "relative",
            "mean": p + order,
            "lesion_volume_ml": 1.0,
        }
        for sid, p in zip(ids, perf)
        for order, region in enumerate(("wmh_core", "0_2_mm", "2_4_mm"))
    ]
    pd.DataFrame(profile).to_csv(deriv / "x_desc-perilesional_features.csv", index=False)
    return bids, deriv


def test_wide_dlbs_participants_and_cognition_are_discovered(tmp_path):
    bids, _ = _fixture(tmp_path, True, n=12)
    metadata, dictionary = harmonize_participants(bids / "participants.tsv")
    cognition, _ = harmonize_cognition([bids / "derivatives" / "cognition" / "memory.csv"])
    assert {"subject", "session", "age", "sex", "education_years"}.issubset(metadata)
    assert len([x for x in cognition if x.startswith("cog_")]) == 2
    assert any(row["analysis_column"] == "age" for row in dictionary)


def test_effect_recovered_and_null_not_manufactured(tmp_path):
    results = []
    for label, effect, seed in (("effect", True, 11), ("null", False, 23)):
        bids, deriv = _fixture(tmp_path / label, effect, seed)
        built = build_analysis_table(bids, deriv, tmp_path / label / "tables")
        run_statistics(
            built["analysis_table"],
            tmp_path / label / "stats",
            built["perilesional_profiles"],
            {"random_seed": 42, "bootstrap_repetitions": 300},
        )
        corr = pd.read_csv(tmp_path / label / "stats" / "pairwise_correlations.csv")
        hit = corr[
            (
                (corr.variable_1 == "absolute_wmh_change_ml")
                & (corr.variable_2 == "relative_perfusion_stable_nawm_mean")
            )
            | (
                (corr.variable_2 == "absolute_wmh_change_ml")
                & (corr.variable_1 == "relative_perfusion_stable_nawm_mean")
            )
        ]
        results.append(hit.iloc[0])
    assert results[0].pearson_r < -0.8 and results[0].pearson_q_bh < 0.05
    assert results[1].pearson_q_bh >= 0.05
    report = build_results(tmp_path / "null" / "stats", tmp_path / "null" / "results.md")
    assert "significant decrease" not in report.read_text(encoding="utf-8").lower()
