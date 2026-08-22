#!/usr/bin/env python3
"""LISA 2026 OS50+AURA Docker inference entry point."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage


OS50_MODEL = "Dataset001_LISA/nnUNetTrainerDiceTopK10_OS50_LS005__nnUNetPlans__3d_fullres"
AURA_MODEL = "Dataset002_LISA_VCF/nnUNetTrainerAURA_v0__nnUNetPlans__3d_fullres"
MODEL_ROOT = Path(os.environ.get("nnUNet_results", "/opt/nnunet_results"))
FILE_ENDING = ".nii.gz"


def case_id_from_name(name: str) -> str:
    if not name.endswith(FILE_ENDING):
        raise ValueError(f"Expected a {FILE_ENDING} image, got {name}")
    stem = name[: -len(FILE_ENDING)]
    for suffix in ("_0000", "_ciso"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    if not stem:
        raise ValueError(f"Could not derive a case ID from {name}")
    return stem


def collect_inputs(input_dir: Path) -> list[tuple[str, Path]]:
    paths = sorted(p for p in input_dir.rglob(f"*{FILE_ENDING}") if p.is_file())
    if not paths:
        raise RuntimeError(f"No {FILE_ENDING} images found below {input_dir}")
    cases: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for path in paths:
        case_id = case_id_from_name(path.name)
        if case_id in seen:
            raise RuntimeError(f"Duplicate case ID {case_id!r} derived from input files")
        seen.add(case_id)
        cases.append((case_id, path))
    return cases


def run_predict(dataset_id: int, trainer: str, source: Path, destination: Path) -> None:
    command = [
        "nnUNetv2_predict",
        "-d", str(dataset_id),
        "-i", str(source),
        "-o", str(destination),
        "-tr", trainer,
        "-c", "3d_fullres",
        "-f", "0",
        "-chk", "checkpoint_best.pth",
        "--save_probabilities",
        "-npp", "2",
        "-nps", "2",
        "-device", "cuda",
    ]
    subprocess.run(command, check=True)


def load_probabilities(path: Path) -> np.ndarray:
    with np.load(path) as archive:
        probabilities = archive["probabilities"].astype(np.float32, copy=False)
    if probabilities.ndim != 4 or probabilities.shape[0] != 12:
        raise RuntimeError(f"Unexpected probability shape in {path}: {probabilities.shape}")
    return probabilities


def keep_largest_components(segmentation: np.ndarray) -> np.ndarray:
    output = segmentation.copy()
    connectivity_26 = np.ones((3, 3, 3), dtype=np.uint8)
    for label_id in range(1, 12):
        components, count = ndimage.label(output == label_id, structure=connectivity_26)
        if count <= 1:
            continue
        sizes = np.bincount(components.ravel())
        sizes[0] = 0
        output[(components != 0) & (components != sizes.argmax())] = 0
    return output


def write_prediction(segmentation: np.ndarray, reference: Path, destination: Path) -> None:
    reference_image = sitk.ReadImage(str(reference))
    if tuple(segmentation.shape) != tuple(reversed(reference_image.GetSize())):
        raise RuntimeError(
            f"Shape mismatch for {reference.name}: prediction {segmentation.shape}, "
            f"reference ZYX {tuple(reversed(reference_image.GetSize()))}"
        )
    prediction = sitk.GetImageFromArray(segmentation.astype(np.uint8, copy=False))
    prediction.CopyInformation(reference_image)
    sitk.WriteImage(prediction, str(destination), useCompression=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("/input"))
    parser.add_argument("--output-dir", type=Path, default=Path("/output"))
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = collect_inputs(input_dir)

    for model in (OS50_MODEL, AURA_MODEL):
        if not (MODEL_ROOT / model / "fold_0" / "checkpoint_best.pth").is_file():
            raise FileNotFoundError(f"Packaged model is incomplete: {MODEL_ROOT / model}")

    with tempfile.TemporaryDirectory(prefix="lisa-os50-aura-") as temporary:
        work = Path(temporary)
        staged = work / "images"
        os50_probs = work / "os50"
        aura_probs = work / "aura"
        staged.mkdir()
        for case_id, source in cases:
            shutil.copy2(source, staged / f"{case_id}_0000{FILE_ENDING}")

        run_predict(1, "nnUNetTrainerDiceTopK10_OS50_LS005", staged, os50_probs)
        run_predict(2, "nnUNetTrainerAURA_v0", staged, aura_probs)

        for case_id, reference in cases:
            base = load_probabilities(os50_probs / f"{case_id}.npz")
            aura = load_probabilities(aura_probs / f"{case_id}.npz")
            if base.shape != aura.shape:
                raise RuntimeError(f"Probability shape mismatch for {case_id}: {base.shape} vs {aura.shape}")
            fused = 0.6 * base + 0.4 * aura
            segmentation = keep_largest_components(np.argmax(fused, axis=0).astype(np.uint8))
            destination = output_dir / f"{case_id}_seg_prediction{FILE_ENDING}"
            write_prediction(segmentation, reference, destination)
            print(f"Wrote {destination}", flush=True)


if __name__ == "__main__":
    main()
