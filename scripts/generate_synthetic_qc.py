"""Generate a small, clearly labelled synthetic structural QC example."""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np

from vascular_edge.preprocessing import preprocess_structural
from vascular_edge.qc import structural_qc
from vascular_edge.segmentation import segment_wmh_files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/synthetic_qc")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    root = Path(args.output)
    inputs = root / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    shape = (64, 64, 56)
    coords = np.indices(shape, dtype=np.float32)
    center = (np.asarray(shape, np.float32) - 1)[:, None, None, None] / 2
    radius = np.sqrt(((coords - center) ** 2).sum(axis=0))
    brain = radius < 23
    t1 = np.zeros(shape, np.float32)
    t1[brain] = 40
    t1[radius < 19] = 75
    t1[radius < 13] = 115
    t1 *= np.linspace(0.8, 1.2, shape[0], dtype=np.float32)[:, None, None]
    flair = np.zeros(shape, np.float32)
    flair[brain] = 30
    flair[radius < 19] = 48
    for point, lesion_radius in (((38, 32, 29), 3.5), ((25, 39, 25), 2.7), ((31, 25, 35), 2.2)):
        lesion = sum((coords[i] - point[i]) ** 2 for i in range(3)) < lesion_radius**2
        flair[lesion] = 175
    rng = np.random.default_rng(2026)
    t1[brain] += rng.normal(0, 1.2, brain.sum())
    flair[brain] += rng.normal(0, 1, brain.sum())
    affine = np.diag([-1.0, 1.0, 1.2, 1.0])
    affine[0, 3] = shape[0] - 1
    t1_file, flair_file = (
        inputs / "sub-synthetic_ses-01_T1w.nii.gz",
        inputs / "sub-synthetic_ses-01_FLAIR.nii.gz",
    )
    nib.save(nib.Nifti1Image(t1, affine), t1_file)
    nib.save(nib.Nifti1Image(flair, affine), flair_file)
    derivatives = root / "derivatives"
    pre = preprocess_structural(
        t1_file, flair_file, derivatives, "sub-synthetic", "ses-01", args.force
    )
    anatomy = derivatives / "sub-synthetic" / "ses-01" / "anat"
    lesions = segment_wmh_files(
        pre["outputs"]["flair_registered"],
        pre["outputs"]["t1_n4"],
        pre["outputs"]["brain_mask"],
        pre["tissues"]["wm"],
        anatomy,
        "sub-synthetic",
        "ses-01",
        "fallback",
        args.force,
        3.0,
        3.0,
    )
    qc = structural_qc(
        pre["outputs"]["t1_n4"],
        pre["outputs"]["flair_registered"],
        pre["tissues"],
        lesions["final_mask"],
        derivatives / "sub-synthetic" / "ses-01" / "qc",
        "sub-synthetic",
        "ses-01",
        pre["registration"]["metric"],
        args.force,
        "fallback",
    )
    print(qc["figure"])


if __name__ == "__main__":
    main()
