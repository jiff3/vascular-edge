"""Generate explicitly synthetic perilesional outputs for visual QA."""

from pathlib import Path

import nibabel as nib
import numpy as np

from vascular_edge.perilesional import analyze_files, plot_cohort


def main() -> None:
    root = Path("results/synthetic_perilesional")
    inputs = root / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    shape = (64, 64, 40)
    affine = np.diag((1.0, 1.0, 2.0, 1.0))
    xyz = np.indices(shape).astype(np.float32)
    center = np.array((32, 32, 20), np.float32)[:, None, None, None]
    radius = np.sqrt(((xyz - center) ** 2).sum(axis=0))
    brain = radius < 27
    flair = (20 + 70 * np.exp(-((radius / 24) ** 2)) + 2 * np.sin(xyz[0] / 5)).astype(np.float32)
    wmh = (((xyz[0] - 27) / 5) ** 2 + ((xyz[1] - 32) / 4) ** 2 + ((xyz[2] - 20) / 2.5) ** 2) <= 1
    wmh |= (((xyz[0] - 42) / 3) ** 2 + ((xyz[1] - 38) / 3) ** 2 + ((xyz[2] - 21) / 2) ** 2) <= 1
    ventricle_radius = np.sqrt((xyz[0] - 32) ** 2 + (xyz[1] - 20) ** 2 + (xyz[2] - 20) ** 2)
    csf = ventricle_radius <= 4
    wm = brain & ~csf
    distance_hint = np.sqrt((xyz[0] - 27) ** 2 + (xyz[1] - 32) ** 2 + (2 * (xyz[2] - 20)) ** 2)
    perfusion = np.where(brain, 38 + 0.35 * distance_hint + 2 * np.sin(xyz[1] / 7), 0).astype(
        np.float32
    )
    cvr = np.where(brain, 0.7 + 0.008 * distance_hint + 0.05 * np.cos(xyz[0] / 6), 0).astype(
        np.float32
    )
    paths = {}
    for name, data in {
        "flair": flair,
        "wmh": wmh,
        "wm": wm,
        "csf": csf,
        "perfusion": perfusion,
        "cvr": cvr,
    }.items():
        path = inputs / f"{name}.nii.gz"
        nib.save(nib.Nifti1Image(data.astype(np.float32), affine), path)
        paths[name] = path
    record = analyze_files(
        paths["wmh"],
        paths["wm"],
        paths["flair"],
        root / "derivatives",
        "sub-synthetic",
        "ses-01",
        paths["csf"],
        {"relative_perfusion": paths["perfusion"], "cvr_percent_bold": paths["cvr"]},
        {"relative_perfusion": "relative", "cvr_percent_bold": "% BOLD"},
        min_lesion_volume_mm3=3,
        force=True,
    )
    import pandas as pd

    table = pd.read_csv(record["outputs"]["csv"])
    plot_cohort(table, root / "cohort_figures")
    print(record["outputs"]["figure"])


if __name__ == "__main__":
    main()
