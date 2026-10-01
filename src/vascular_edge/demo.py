"""Small deterministic, redistributable inputs for the end-to-end demonstration."""

from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


def _save(array: np.ndarray, path: Path, affine: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.asarray(array, np.float32), affine), path)
    return path


def create_demo_bids(root: str | Path, seed: int = 2026) -> dict[str, object]:
    """Create two-session synthetic BIDS-like MRI plus cohort metadata.

    All arrays are mathematical phantoms and contain no participant data. The
    injected physiology/progression relationship exists solely to test recovery.
    """
    root = Path(root)
    marker = root / "dataset_description.json"
    if marker.exists() and (root / "participants.tsv").exists():
        return {"root": str(root), "resumed": True, "seed": seed}
    rng = np.random.default_rng(seed)
    shape = (48, 48, 36)
    grid = np.indices(shape, dtype=np.float32)
    center = (np.asarray(shape, np.float32) - 1)[:, None, None, None] / 2
    radius = np.sqrt(((grid - center) ** 2).sum(axis=0))
    brain = radius < 17
    affine = np.diag((1.0, 1.0, 1.4, 1.0))
    landmark = 18 * np.exp(
        -(((grid[0] - 31) / 4) ** 2 + ((grid[1] - 19) / 3) ** 2 + ((grid[2] - 19) / 3) ** 2)
    )
    t1 = np.where(brain, 45 + 40 * (radius < 15) + 45 * (radius < 11) + landmark, 0).astype(
        np.float32
    )
    t1[brain] += rng.normal(0, 0.6, brain.sum())
    lesion_distance = np.sqrt(
        ((grid - np.array((17, 25, 18))[:, None, None, None]) ** 2).sum(axis=0)
    )
    baseline_lesion = lesion_distance <= 2.8
    followup_lesion = lesion_distance <= 4.1
    base_flair = np.where(brain, 35 + 12 * np.exp(-((radius / 15) ** 2)), 0).astype(np.float32)
    for wave, lesion in ((1, baseline_lesion), (2, followup_lesion)):
        session = root / "sub-demo01" / f"ses-wave{wave}" / "anat"
        flair = base_flair.copy()
        flair[lesion] += 85
        _save(t1, session / f"sub-demo01_ses-wave{wave}_T1w.nii.gz", affine)
        _save(flair, session / f"sub-demo01_ses-wave{wave}_FLAIR.nii.gz", affine)
    perf_dir = root / "sub-demo01" / "ses-wave1" / "perf"
    frames = []
    control = np.where(brain, 100 + 3 * np.sin(grid[1] / 5), 0)
    delta = np.where(brain, 4.0 + 0.12 * lesion_distance, 0)
    for _ in range(5):
        noise_a = rng.normal(0, 0.15, shape) * brain
        noise_b = rng.normal(0, 0.15, shape) * brain
        frames.extend([control - delta + noise_a, control + noise_b])
    asl = _save(np.stack(frames, axis=-1), perf_dir / "sub-demo01_ses-wave1_asl.nii.gz", affine)
    (perf_dir / "sub-demo01_ses-wave1_asl.json").write_text(
        json.dumps(
            {
                "ArterialSpinLabelingType": "PCASL",
                "PostLabelingDelay": 1.525,
                "LabelingDuration": 1.65,
                "M0Type": "Absent",
                "TotalAcquiredPairs": 5,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (perf_dir / "sub-demo01_ses-wave1_aslcontext.tsv").write_text(
        "volume_type\n" + "\n".join(["label", "control"] * 5), encoding="utf-8"
    )
    func_dir = root / "sub-demo01" / "ses-wave1" / "func"
    regressor = np.array([0] * 4 + [1] * 4 + [0] * 4, np.float32)
    bold = np.stack(
        [
            np.where(brain, 100 + value * (1.2 + 0.025 * lesion_distance), 0)
            + rng.normal(0, 0.08, shape) * brain
            for value in regressor
        ],
        axis=-1,
    )
    _save(bold, func_dir / "sub-demo01_ses-wave1_task-hypercapnia_bold.nii.gz", affine)
    (func_dir / "sub-demo01_ses-wave1_task-hypercapnia_bold.json").write_text(
        json.dumps({"TaskName": "Hypercapnia", "RepetitionTime": 2.0}, indent=2), encoding="utf-8"
    )
    (func_dir / "sub-demo01_ses-wave1_task-hypercapnia_events.tsv").write_text(
        "stimulus\n" + "\n".join(map(str, regressor.astype(int))), encoding="utf-8"
    )
    # A tabular cohort provides adequate n for the cohort statistics while the
    # image phantom above exercises every image-processing component.
    n = 48
    subjects = [f"sub-sim{i:03d}" for i in range(n)]
    age = rng.uniform(45, 85, n)
    sex = rng.choice(["f", "m"], n)
    education = rng.integers(12, 21, n)
    burden = np.exp(rng.normal(-0.2 + 0.022 * (age - 65), 0.38, n))
    perfusion = 54 - 0.15 * (age - 65) - 0.9 * burden + rng.normal(0, 2.0, n)
    participants = pd.DataFrame(
        {
            "participant_id": subjects + ["sub-demo01"],
            "AgeMRI_W1": np.r_[age, 64],
            "Sex": np.concatenate([sex, ["f"]]),
            "EduYrsEstCap": np.r_[education, 16],
            "BMI_W1": np.r_[rng.normal(26, 3, n), 25],
            "MMSE_W1": np.r_[rng.integers(26, 31, n), 29],
        }
    )
    participants.to_csv(root / "participants.tsv", sep="\t", index=False)
    cognition = root / "derivatives" / "cognition"
    cognition.mkdir(parents=True, exist_ok=True)
    memory = 0.18 * perfusion - 0.12 * age + 0.2 * education + rng.normal(0, 3, n)
    pd.DataFrame(
        {
            "participant_id": subjects,
            "session": "ses-wave1",
            "memory_total": memory,
            "memory_delayed": memory + rng.normal(0, 1.5, n),
        }
    ).to_csv(cognition / "episodic_memory.csv", index=False)
    marker.write_text(
        json.dumps(
            {
                "Name": "Vascular Edge deterministic synthetic demo",
                "BIDSVersion": "1.9.0",
                "DatasetType": "raw",
                "GeneratedBy": [{"Name": "vascular-edge"}],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"root": str(root), "resumed": False, "seed": seed, "asl": str(asl)}


def create_demo_cohort_derivatives(root: str | Path, seed: int = 2026) -> dict[str, str]:
    """Write deterministic cohort-level derivative tables with a known effect."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    longitudinal = root / "synthetic_cohort_desc-longitudinal_subject.csv"
    profiles = root / "synthetic_cohort_desc-perilesional_features.csv"
    if longitudinal.exists() and profiles.exists():
        return {"longitudinal": str(longitudinal), "profiles": str(profiles)}
    rng = np.random.default_rng(seed)
    n = 48
    subjects = [f"sub-sim{i:03d}" for i in range(n)]
    age = rng.uniform(45, 85, n)
    burden = np.exp(rng.normal(-0.2 + 0.022 * (age - 65), 0.38, n))
    stable = 5.4 - 0.015 * (age - 65) - 0.09 * burden + rng.normal(0, 0.2, n)
    converting = stable - 0.32 + rng.normal(0, 0.1, n)
    raw_change = 0.8 - 2.0 * (stable - stable.mean()) + rng.normal(0, 0.2, n)
    followup = np.maximum(0.05, burden + raw_change)
    change = followup - burden
    pd.DataFrame(
        {
            "subject": subjects,
            "baseline_session": "ses-wave1",
            "followup_session": "ses-wave2",
            "baseline_wmh_ml": burden,
            "followup_wmh_ml": followup,
            "absolute_wmh_change_ml": change,
            "newly_affected_ml": np.maximum(change, 0),
            "relative_perfusion_stable_nawm_mean": stable,
            "relative_perfusion_converting_mean": converting,
        }
    ).to_csv(longitudinal, index=False)
    rows = []
    regions = ("wmh_core", "0_2_mm", "2_4_mm", "4_6_mm", "6_10_mm", "remote_gt_10_mm")
    for subject, base, value in zip(subjects, burden, stable):
        for order, region in enumerate(regions):
            rows.append(
                {
                    "subject": subject,
                    "session": "ses-wave1",
                    "scope": "subject",
                    "region": region,
                    "region_order": order,
                    "map": "relative_perfusion",
                    "units": "% mean control signal",
                    "mean": value - 0.4 + 0.115 * order + rng.normal(0, 0.055),
                    "median": value - 0.4 + 0.115 * order,
                    "lesion_volume_ml": base,
                }
            )
    pd.DataFrame(rows).to_csv(profiles, index=False)
    return {"longitudinal": str(longitudinal), "profiles": str(profiles)}
