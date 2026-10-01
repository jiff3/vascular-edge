"""CPU-conscious structural MRI preprocessing and validation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import SimpleITK as sitk
from nibabel.orientations import aff2axcodes
from scipy import ndimage
from skimage.filters import threshold_multiotsu, threshold_otsu


@dataclass
class ImageValidation:
    path: str
    shape: tuple[int, ...]
    voxel_sizes_mm: tuple[float, ...]
    orientation: tuple[str, ...]
    finite_fraction: float
    nonzero_fraction: float
    affine_determinant: float
    status: str
    warnings: list[str]


def load_float32(image_file: str | Path) -> tuple[np.ndarray, nib.Nifti1Image]:
    image = nib.load(str(image_file))
    if len(image.shape) != 3:
        raise ValueError(f"Expected a 3D NIfTI, got shape {image.shape}: {image_file}")
    if not np.isfinite(image.affine).all() or abs(np.linalg.det(image.affine[:3, :3])) < 1e-8:
        raise ValueError(f"Invalid or singular NIfTI affine: {image_file}")
    return image.get_fdata(dtype=np.float32), image


def validate_image(image_file: str | Path) -> ImageValidation:
    """Validate header, data coverage, finite values, and plausible voxels."""
    data, image = load_float32(image_file)
    zooms = tuple(float(v) for v in image.header.get_zooms()[:3])
    warnings: list[str] = []
    finite = np.isfinite(data)
    finite_fraction = float(finite.mean())
    if finite_fraction < 1:
        warnings.append(f"non-finite voxels: {(~finite).sum()}")
    if any(v < 0.3 or v > 5.0 for v in zooms):
        warnings.append(f"unusual voxel sizes: {zooms}")
    if any(n < 32 for n in data.shape) or any(n > 1024 for n in data.shape):
        warnings.append(f"suspicious dimensions: {data.shape}")
    nonzero_fraction = float(np.count_nonzero(np.nan_to_num(data)) / data.size)
    if nonzero_fraction < 0.05:
        warnings.append(f"low nonzero image coverage: {nonzero_fraction:.3f}")
    determinant = float(np.linalg.det(image.affine[:3, :3]))
    orientation = tuple(str(x) for x in aff2axcodes(image.affine))
    if orientation != ("R", "A", "S"):
        warnings.append(f"noncanonical input orientation {orientation}; will reorient to RAS+")
    return ImageValidation(
        str(image_file),
        tuple(data.shape),
        zooms,
        orientation,
        finite_fraction,
        nonzero_fraction,
        determinant,
        "warn" if warnings else "pass",
        warnings,
    )


def canonicalize_nifti(input_file: str | Path, output_file: str | Path) -> Path:
    """Reorder to closest RAS+ while preserving voxel-to-world geometry."""
    image = nib.load(str(input_file))
    canonical = nib.as_closest_canonical(image, enforce_diag=False)
    data = canonical.get_fdata(dtype=np.float32)
    data[~np.isfinite(data)] = 0
    output = Path(output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    header = canonical.header.copy()
    header.set_data_dtype(np.float32)
    nib.save(nib.Nifti1Image(data, canonical.affine, header), str(output))
    return output


def largest_component(mask: np.ndarray) -> np.ndarray:
    labels, count = ndimage.label(mask)
    if count == 0:
        return np.zeros(mask.shape, dtype=bool)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == int(np.argmax(sizes))


def extract_brain_mask(data: np.ndarray, closing_iterations: int = 2) -> np.ndarray:
    """Lightweight Otsu/morphology extraction; not a clinical brain mask."""
    array = np.nan_to_num(np.asarray(data, np.float32), copy=True)
    positive = array[array > 0]
    if positive.size < 32:
        return np.zeros(array.shape, dtype=bool)
    upper = np.percentile(positive, 99.5)
    threshold = threshold_otsu(np.clip(positive, None, upper))
    mask = array > threshold * 0.35
    mask = ndimage.binary_closing(mask, iterations=closing_iterations)
    mask = ndimage.binary_fill_holes(mask)
    mask = largest_component(mask)
    return ndimage.binary_opening(mask, iterations=1).astype(bool)


def save_mask(mask: np.ndarray, reference: nib.Nifti1Image, output_file: str | Path) -> Path:
    output = Path(output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    header = reference.header.copy()
    header.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(np.asarray(mask, np.uint8), reference.affine, header), str(output))
    return output


def apply_mask(image_file: str | Path, mask_file: str | Path, output_file: str | Path) -> Path:
    data, image = load_float32(image_file)
    mask = nib.load(str(mask_file)).get_fdata() > 0
    if data.shape != mask.shape:
        raise ValueError("Image and brain mask grids do not match.")
    output = Path(output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    header = image.header.copy()
    header.set_data_dtype(np.float32)
    nib.save(
        nib.Nifti1Image(np.where(mask, data, 0).astype(np.float32), image.affine, header),
        str(output),
    )
    return output


def n4_bias_correct(
    input_file: str | Path,
    mask_file: str | Path,
    output_file: str | Path,
    shrink_factor: int = 2,
    iterations: tuple[int, ...] = (20, 10, 5),
) -> Path:
    image = sitk.Cast(sitk.ReadImage(str(input_file)), sitk.sitkFloat32)
    mask = sitk.Cast(sitk.ReadImage(str(mask_file)), sitk.sitkUInt8)
    if shrink_factor > 1:
        factors = [shrink_factor] * image.GetDimension()
        work_image, work_mask = sitk.Shrink(image, factors), sitk.Shrink(mask, factors)
    else:
        work_image, work_mask = image, mask
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations(list(iterations))
    corrector.Execute(work_image, work_mask)
    corrected = image / sitk.Exp(corrector.GetLogBiasFieldAsImage(image))
    output = Path(output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(sitk.Cast(corrected, sitk.sitkFloat32), str(output))
    return output


def register_flair_to_t1(
    flair_file: str | Path,
    t1_file: str | Path,
    output_file: str | Path,
    transform_file: str | Path,
    sampling_percentage: float = 0.15,
) -> dict[str, Any]:
    """Rigid then affine Mattes-MI registration and linear resampling."""
    fixed = sitk.Cast(sitk.ReadImage(str(t1_file)), sitk.sitkFloat32)
    moving = sitk.Cast(sitk.ReadImage(str(flair_file)), sitk.sitkFloat32)
    initial = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY
    )

    def method(iterations: int) -> sitk.ImageRegistrationMethod:
        reg = sitk.ImageRegistrationMethod()
        reg.SetMetricAsMattesMutualInformation(32)
        reg.SetMetricSamplingStrategy(reg.RANDOM)
        reg.SetMetricSamplingPercentage(sampling_percentage, seed=2026)
        reg.SetInterpolator(sitk.sitkLinear)
        reg.SetOptimizerAsGradientDescent(1.0, iterations, 1e-6, 10)
        reg.SetOptimizerScalesFromPhysicalShift()
        reg.SetShrinkFactorsPerLevel([4, 2, 1])
        reg.SetSmoothingSigmasPerLevel([2, 1, 0])
        reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        return reg

    rigid_method = method(60)
    rigid_method.SetInitialTransform(initial, inPlace=False)
    rigid = rigid_method.Execute(fixed, moving)
    rigid_base = rigid.GetBackTransform() if isinstance(rigid, sitk.CompositeTransform) else rigid
    affine = sitk.AffineTransform(3)
    affine.SetCenter(rigid_base.GetCenter())
    affine.SetMatrix(rigid_base.GetMatrix())
    affine.SetTranslation(rigid_base.GetTranslation())
    affine_method = method(40)
    affine_method.SetInitialTransform(affine, inPlace=False)
    affine_error = None
    try:
        final = affine_method.Execute(fixed, moving)
        candidates = {"affine": final, "rigid": rigid, "initialized": initial}
    except RuntimeError as exc:
        affine_error = str(exc).splitlines()[-1]
        candidates = {"rigid": rigid, "initialized": initial}
    fixed_array = sitk.GetArrayViewFromImage(fixed)
    fixed_foreground = fixed_array > 0

    def candidate_score(transform: sitk.Transform) -> tuple[float, sitk.Image]:
        candidate = sitk.Resample(moving, fixed, transform, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
        foreground = sitk.GetArrayViewFromImage(candidate) > 0
        denominator = min(int(fixed_foreground.sum()), int(foreground.sum()))
        score = float((fixed_foreground & foreground).sum() / denominator) if denominator else 0.0
        return score, candidate

    evaluated = {name: candidate_score(transform) for name, transform in candidates.items()}
    selected_stage = max(evaluated, key=lambda name: evaluated[name][0])
    selected_transform = candidates[selected_stage]
    overlap_score, resampled = evaluated[selected_stage]
    output = Path(output_file)
    transform = Path(transform_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    transform.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(resampled, str(output))
    sitk.WriteTransform(selected_transform, str(transform))
    return {
        "metric": float(affine_method.GetMetricValue()),
        "optimizer_stop": affine_method.GetOptimizerStopConditionDescription(),
        "transform": str(transform),
        "selected_stage": selected_stage,
        "foreground_overlap": overlap_score,
        "affine_error": affine_error,
    }


def segment_tissues(
    t1_file: str | Path, brain_mask_file: str | Path, output_dir: str | Path, prefix: str
) -> dict[str, Path]:
    """Three-class T1 intensity segmentation (CSF/GM/WM), backend-ready."""
    data, image = load_float32(t1_file)
    mask = nib.load(str(brain_mask_file)).get_fdata() > 0
    values = data[mask & np.isfinite(data)]
    if values.size < 32:
        raise ValueError("Brain mask is too small for tissue segmentation.")
    q1, q2 = threshold_multiotsu(values, classes=3)
    classes = {
        "csf": mask & (data <= q1),
        "gm": mask & (data > q1) & (data <= q2),
        "wm": mask & (data > q2),
    }
    outputs: dict[str, Path] = {}
    root = Path(output_dir)
    for name, tissue in classes.items():
        tissue = ndimage.binary_opening(tissue, iterations=1)
        outputs[name] = save_mask(
            tissue, image, root / f"{prefix}_label-{name.upper()}_probseg.nii.gz"
        )
    return outputs


def structural_paths(root: str | Path, subject: str, session: str) -> dict[str, Path]:
    base = Path(root) / subject / session / "anat"
    prefix = f"{subject}_{session}"
    return {
        "dir": base,
        "prefix": Path(prefix),
        "t1_canonical": base / f"{prefix}_desc-canonical_T1w.nii.gz",
        "flair_canonical": base / f"{prefix}_desc-canonical_FLAIR.nii.gz",
        "brain_mask": base / f"{prefix}_desc-brain_mask.nii.gz",
        "t1_n4": base / f"{prefix}_desc-N4_T1w.nii.gz",
        "t1_brain": base / f"{prefix}_desc-N4brain_T1w.nii.gz",
        "flair_registered_raw": base / f"{prefix}_space-T1w_desc-reg_FLAIR.nii.gz",
        "flair_registered": base / f"{prefix}_space-T1w_desc-N4_FLAIR.nii.gz",
        "flair_brain": base / f"{prefix}_space-T1w_desc-N4brain_FLAIR.nii.gz",
        "transform": base / f"{prefix}_from-FLAIR_to-T1w_mode-image_xfm.tfm",
        "provenance": base / f"{prefix}_desc-structural_provenance.json",
    }


def preprocess_structural(
    t1_file: str | Path,
    flair_file: str | Path,
    derivatives_root: str | Path,
    subject: str,
    session: str,
    force: bool = False,
    n4_shrink_factor: int = 2,
) -> dict[str, Any]:
    """Canonicalize, extract, N4-correct, register, and segment one session."""
    paths = structural_paths(derivatives_root, subject, session)
    expected = [paths[k] for k in ("t1_n4", "flair_registered", "brain_mask", "provenance")]
    if not force and all(path.exists() for path in expected):
        record = json.loads(paths["provenance"].read_text(encoding="utf-8"))
        record["resumed"] = True
        return record
    paths["dir"].mkdir(parents=True, exist_ok=True)
    validation = {
        "t1": asdict(validate_image(t1_file)),
        "flair": asdict(validate_image(flair_file)),
    }
    canonicalize_nifti(t1_file, paths["t1_canonical"])
    canonicalize_nifti(flair_file, paths["flair_canonical"])
    t1_data, t1_image = load_float32(paths["t1_canonical"])
    brain = extract_brain_mask(t1_data)
    save_mask(brain, t1_image, paths["brain_mask"])
    if brain.sum() < 100:
        raise ValueError("Brain extraction produced an implausibly small mask.")
    n4_bias_correct(paths["t1_canonical"], paths["brain_mask"], paths["t1_n4"], n4_shrink_factor)
    apply_mask(paths["t1_n4"], paths["brain_mask"], paths["t1_brain"])
    registration = register_flair_to_t1(
        paths["flair_canonical"], paths["t1_n4"], paths["flair_registered_raw"], paths["transform"]
    )
    n4_bias_correct(
        paths["flair_registered_raw"],
        paths["brain_mask"],
        paths["flair_registered"],
        n4_shrink_factor,
    )
    apply_mask(paths["flair_registered"], paths["brain_mask"], paths["flair_brain"])
    tissues = segment_tissues(
        paths["t1_n4"], paths["brain_mask"], paths["dir"], str(paths["prefix"])
    )
    record = {
        "subject": subject,
        "session": session,
        "inputs": {"t1": str(t1_file), "flair": str(flair_file)},
        "validation": validation,
        "registration": registration,
        "algorithms": {
            "orientation": "nibabel closest canonical RAS+",
            "bias": "SimpleITK N4",
            "brain_extraction": "Otsu+morphology largest component",
            "registration": "SimpleITK rigid then affine Mattes MI",
            "tissues": "three-class T1 intensity segmentation",
        },
        "software": {
            "nibabel": nib.__version__,
            "numpy": np.__version__,
            "SimpleITK": sitk.Version_VersionString(),
        },
        "parameters": {"n4_shrink_factor": n4_shrink_factor, "processing_dtype": "float32"},
        "outputs": {
            key: str(value) for key, value in paths.items() if key not in {"dir", "prefix"}
        },
        "tissues": {key: str(value) for key, value in tissues.items()},
        "resumed": False,
    }
    paths["provenance"].write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def resample_to_reference(
    moving_file: str | Path, reference_file: str | Path, output_file: str | Path
) -> Path:
    moving = sitk.ReadImage(str(moving_file))
    reference = sitk.ReadImage(str(reference_file))
    result = sitk.Resample(
        moving, reference, sitk.Transform(), sitk.sitkLinear, 0.0, sitk.sitkFloat32
    )
    output = Path(output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(result, str(output))
    return output
