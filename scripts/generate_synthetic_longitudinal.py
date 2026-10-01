"""Generate a complete, explicitly synthetic longitudinal example."""

from pathlib import Path

import nibabel as nib
import numpy as np

from vascular_edge.longitudinal import analyze_longitudinal_files


def _shift(array: np.ndarray, amount: int) -> np.ndarray:
    shifted = np.zeros_like(array)
    shifted[amount:, :, :] = array[:-amount, :, :]
    return shifted


def main() -> None:
    root = Path("results/synthetic_longitudinal")
    inputs = root / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    shape = (56, 56, 36)
    affine = np.diag((1.0, 1.0, 1.5, 1.0))
    grid = np.indices(shape).astype(np.float32)
    center = np.array((28, 28, 18))[:, None, None, None]
    radius = np.sqrt(((grid - center) ** 2).sum(axis=0))
    brain = radius < 24
    wm = brain & (radius > 7)
    landmark = 25 * np.exp(
        -(((grid[0] - 37) / 5) ** 2 + ((grid[1] - 21) / 4) ** 2 + ((grid[2] - 20) / 3) ** 2)
    )
    texture = (
        75
        + 25 * np.exp(-((radius / 18) ** 2))
        + 5 * np.sin(grid[0] / 4)
        + 3 * np.cos(grid[1] / 5)
        + landmark
    )
    t1_baseline = np.where(brain, texture, 0).astype(np.float32)
    lesion_center = np.array((18, 28, 18))[:, None, None, None]
    lesion_radius = np.sqrt(((grid - lesion_center) ** 2).sum(axis=0))
    baseline_wmh = lesion_radius <= 3.0
    followup_wmh_native = lesion_radius <= 4.2
    baseline_flair = np.where(brain, 35 + 15 * np.exp(-((radius / 20) ** 2)), 0).astype(np.float32)
    baseline_flair[baseline_wmh] += 50
    followup_flair = baseline_flair.copy()
    followup_flair[followup_wmh_native] += 45
    # Follow-up acquisition is deliberately translated two voxels; all its arrays
    # move together so affine registration must return them to baseline space.
    t1_followup = _shift(t1_baseline, 2)
    followup_flair = _shift(followup_flair, 2)
    followup_wmh = _shift(followup_wmh_native.astype(np.uint8), 2)
    distance = np.sqrt(((grid - lesion_center) ** 2).sum(axis=0))
    perfusion = np.where(brain, 42 + 0.35 * distance + 2 * np.sin(grid[1] / 6), 0).astype(
        np.float32
    )
    cvr = np.where(brain, 0.65 + 0.01 * distance + 0.04 * np.cos(grid[0] / 7), 0).astype(np.float32)
    paths = {}
    for name, data in {
        "baseline_t1": t1_baseline,
        "baseline_flair": baseline_flair,
        "baseline_wmh": baseline_wmh,
        "baseline_wm": wm,
        "followup_t1": t1_followup,
        "followup_flair": followup_flair,
        "followup_wmh": followup_wmh,
        "perfusion": perfusion,
        "cvr": cvr,
    }.items():
        path = inputs / f"{name}.nii.gz"
        nib.save(nib.Nifti1Image(data.astype(np.float32), affine), path)
        paths[name] = path
    record = analyze_longitudinal_files(
        paths["baseline_t1"],
        paths["baseline_flair"],
        paths["baseline_wmh"],
        paths["baseline_wm"],
        paths["followup_t1"],
        paths["followup_flair"],
        paths["followup_wmh"],
        root / "derivatives",
        "sub-synthetic",
        "ses-wave1",
        "ses-wave2",
        {"relative_perfusion": paths["perfusion"], "cvr_percent_bold": paths["cvr"]},
        {"relative_perfusion": "relative", "cvr_percent_bold": "% BOLD"},
        boundary_uncertainty_mm=1.0,
        min_new_component_mm3=4.0,
        force=True,
    )
    print(record["outputs"]["figure"])


if __name__ == "__main__":
    main()
