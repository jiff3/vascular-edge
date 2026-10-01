"""Generate a labelled synthetic ASL/CVR example and physiology QC montage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from vascular_edge.cvr import process_cvr
from vascular_edge.perfusion import process_perfusion
from vascular_edge.physiology_qc import qc_physiology


def save(array: np.ndarray, path: Path) -> Path:
    nib.save(nib.Nifti1Image(array.astype(np.float32), np.eye(4)), path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/synthetic_physiology")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    root = Path(args.output)
    inputs = root / "inputs"
    outputs = root / "derivatives"
    inputs.mkdir(parents=True, exist_ok=True)
    shape = (48, 48, 40)
    xyz = np.indices(shape)
    center = (np.asarray(shape) - 1)[:, None, None, None] / 2
    radius = np.sqrt(((xyz - center) ** 2).sum(0))
    brain = radius < 17
    wm = radius < 10
    gm = brain & ~wm
    t1 = save(brain * 100 + wm * 25, inputs / "T1w.nii.gz")
    flair = save(brain * 55, inputs / "FLAIR.nii.gz")
    gm_file = save(gm, inputs / "GM.nii.gz")
    wm_file = save(wm, inputs / "WM.nii.gz")
    rng = np.random.default_rng(2026)
    asl_frames = []
    for _ in range(6):
        asl_frames.extend(
            [brain * 95 + rng.normal(0, 0.3, shape), brain * 100 + rng.normal(0, 0.3, shape)]
        )
    asl = save(np.stack(asl_frames, axis=-1), inputs / "asl.nii.gz")
    asl_json = inputs / "asl.json"
    asl_json.write_text(
        json.dumps(
            {
                "ArterialSpinLabelingType": "PCASL",
                "PostLabelingDelay": 1.525,
                "LabelingDuration": 1.65,
                "M0Type": "Absent",
                "TotalAcquiredPairs": 6,
            }
        ),
        encoding="utf-8",
    )
    context = inputs / "aslcontext.tsv"
    context.write_text("volume_type\n" + "\n".join(["label", "control"] * 6), encoding="utf-8")
    perf = process_perfusion(
        asl,
        asl_json,
        t1,
        gm_file,
        wm_file,
        outputs / "perf",
        "sub-synthetic",
        "ses-01",
        context,
        motion_correct=False,
        force=args.force,
    )
    regressor = np.array([0] * 4 + [1] * 4 + [0] * 4, np.float32)
    bold_frames = []
    for value in regressor:
        bold_frames.append(brain * (100 + 2 * value) + rng.normal(0, 0.15, shape))
    bold = save(np.stack(bold_frames, axis=-1), inputs / "hypercapnia_bold.nii.gz")
    bold_json = inputs / "hypercapnia_bold.json"
    bold_json.write_text(
        json.dumps({"TaskName": "Hypercapnia", "RepetitionTime": 2.0}), encoding="utf-8"
    )
    regressor_file = inputs / "documented_regressor.tsv"
    regressor_file.write_text("stimulus\n" + "\n".join(map(str, regressor)), encoding="utf-8")
    cvr = process_cvr(
        bold,
        bold_json,
        t1,
        outputs / "cvr",
        "sub-synthetic",
        "ses-01",
        regressor_file,
        motion_correct=False,
        force=args.force,
    )
    qc = qc_physiology(
        t1,
        flair,
        perf["outputs"]["t1_map"],
        gm_file,
        wm_file,
        outputs / "qc",
        "sub-synthetic",
        "ses-01",
        perf["assessment"]["units"],
        asl,
        cvr["outputs"]["t1_map"],
        cvr["units"],
        args.force,
        perf["outputs"]["registered_control"],
    )
    print(qc["figure"])


if __name__ == "__main__":
    main()
