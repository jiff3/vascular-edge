#!/usr/bin/env python
"""Generate effect and null cohort fixtures for end-to-end statistical validation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate(root: Path, effect: bool, seed: int = 2026, n: int = 80) -> None:
    rng = np.random.default_rng(seed)
    bids = root / "bids"
    derivatives = root / "derivatives"
    cognition = bids / "derivatives" / "cognition"
    cognition.mkdir(parents=True, exist_ok=True)
    derivatives.mkdir(parents=True, exist_ok=True)
    subjects = [f"sub-{i:03d}" for i in range(1, n + 1)]
    age = rng.uniform(45, 85, n)
    sex = rng.choice(["f", "m"], n)
    education = rng.integers(12, 21, n)
    burden = np.exp(rng.normal(-0.1 + 0.025 * (age - 65), 0.45, n))
    perfusion = 55 - 0.17 * (age - 65) - 1.2 * burden + rng.normal(0, 2.2, n)
    progression = (
        (-0.09 * perfusion + 0.06 * burden + rng.normal(0, 0.35, n))
        if effect
        else rng.normal(0, 0.55, n)
    )
    participants = pd.DataFrame(
        {
            "participant_id": subjects,
            "AgeMRI_W1": age,
            "Sex": sex,
            "EduYrsEstCap": education,
            "BMI_W1": rng.normal(26, 3, n),
            "MMSE_W1": rng.integers(26, 31, n),
        }
    )
    participants.to_csv(bids / "participants.tsv", sep="\t", index=False)
    cognition_score = (
        0.25 * perfusion - 0.18 * age + 0.22 * education + rng.normal(0, 3, n)
        if effect
        else rng.normal(0, 5, n)
    )
    pd.DataFrame(
        {
            "participant_id": subjects,
            "session": "ses-wave1",
            "memory_total": cognition_score,
            "memory_delayed": cognition_score + rng.normal(0, 2, n),
        }
    ).to_csv(cognition / "episodic_memory.csv", index=False)
    long = pd.DataFrame(
        {
            "subject": subjects,
            "baseline_session": "ses-wave1",
            "followup_session": "ses-wave2",
            "baseline_wmh_ml": burden,
            "followup_wmh_ml": burden + progression,
            "absolute_wmh_change_ml": progression,
            "newly_affected_ml": np.maximum(progression, 0),
            "relative_perfusion_stable_nawm_mean": perfusion,
        }
    )
    long.to_csv(derivatives / "cohort_desc-longitudinal_subject.csv", index=False)
    rows = []
    for subject, base, perf in zip(subjects, burden, perfusion):
        for order, region in enumerate(("wmh_core", "0_2_mm", "2_4_mm", "4_6_mm", "6_10_mm")):
            rows.append(
                {
                    "subject": subject,
                    "session": "ses-wave1",
                    "scope": "subject",
                    "region": region,
                    "region_order": order,
                    "map": "relative_perfusion",
                    "units": "relative",
                    "mean": perf + order * (1.5 if effect else 0) + rng.normal(0, 0.7),
                    "lesion_volume_ml": base,
                }
            )
    pd.DataFrame(rows).to_csv(derivatives / "cohort_desc-perilesional_features.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    generate(args.output / "effect", True, args.seed)
    generate(args.output / "null", False, args.seed + 1)


if __name__ == "__main__":
    main()
