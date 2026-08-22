#!/usr/bin/env python3
"""Screen the OS50/AURA soft-fusion weight on the development split.

For each weight ``w`` the script forms ``w * p(OS50) + (1 - w) * p(AURA)``,
takes the argmax, applies per-class largest-connected-component filtering, and
scores mean Dice against the HF reference labels. This is the procedure that
selected the 0.6 / 0.4 weight reported in the paper.

The selection is made on the same 16 development cases the paper reports, so
the resulting ensemble score is validation-tuned and is not an unbiased
estimate of generalization.

Usage:
    python scripts/sweep_ensemble_weight.py \\
        --os50-probs  predictions/os50_fold0_probs \\
        --aura-probs  predictions/aura_fold0_probs \\
        --out results/metrics/ensemble_weight_sweep.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

N_CLASSES = 11
HIPPOCAMPI = (1, 2)


def default_gt_dir() -> Path | None:
    raw = os.environ.get("nnUNet_raw")
    return Path(raw) / "Dataset001_LISA" / "labelsTr" if raw else None


def load_probabilities(path: Path) -> np.ndarray:
    with np.load(path) as archive:
        probabilities = archive["probabilities"].astype(np.float32, copy=False)
    if probabilities.ndim != 4 or probabilities.shape[0] != N_CLASSES + 1:
        raise ValueError(f"{path}: expected (12, Z, Y, X) probabilities, got {probabilities.shape}")
    return probabilities


def keep_largest_components(segmentation: np.ndarray) -> np.ndarray:
    output = segmentation.copy()
    connectivity_26 = np.ones((3, 3, 3), dtype=np.uint8)
    for label_id in range(1, N_CLASSES + 1):
        components, count = ndimage.label(output == label_id, structure=connectivity_26)
        if count <= 1:
            continue
        sizes = np.bincount(components.ravel())
        sizes[0] = 0
        output[(components != 0) & (components != sizes.argmax())] = 0
    return output


def dice_per_class(prediction: np.ndarray, reference: np.ndarray) -> np.ndarray:
    scores = np.full(N_CLASSES, np.nan)
    for label_id in range(1, N_CLASSES + 1):
        p = prediction == label_id
        r = reference == label_id
        denominator = p.sum() + r.sum()
        if denominator == 0:
            continue
        scores[label_id - 1] = 2.0 * np.logical_and(p, r).sum() / denominator
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--os50-probs", type=Path, required=True,
                        help="Directory of OS50 <case>.npz probability maps")
    parser.add_argument("--aura-probs", type=Path, required=True,
                        help="Directory of AURA <case>.npz probability maps")
    parser.add_argument("--gt-dir", type=Path, default=default_gt_dir(),
                        help="Reference labels (default: $nnUNet_raw/Dataset001_LISA/labelsTr)")
    parser.add_argument("--weights", default=",".join(f"{w / 10:.1f}" for w in range(11)),
                        help="Comma-separated OS50 weights to screen")
    parser.add_argument("--out", type=Path, default=None, help="Output JSON path")
    args = parser.parse_args()

    if args.gt_dir is None:
        raise SystemExit("--gt-dir is required when nnUNet_raw is not set")

    cases = sorted(p.stem for p in args.os50_probs.glob("*.npz"))
    if not cases:
        raise SystemExit(f"No .npz probability maps found in {args.os50_probs}")
    aura_cases = sorted(p.stem for p in args.aura_probs.glob("*.npz"))
    if aura_cases != cases:
        raise SystemExit("The two probability directories cover different cases")

    weights = [float(w) for w in args.weights.split(",") if w.strip()]
    accumulator = {w: [] for w in weights}

    for case in cases:
        os50 = load_probabilities(args.os50_probs / f"{case}.npz")
        aura = load_probabilities(args.aura_probs / f"{case}.npz")
        if os50.shape != aura.shape:
            raise SystemExit(f"{case}: probability shape mismatch {os50.shape} vs {aura.shape}")
        reference = np.asanyarray(nib.load(str(args.gt_dir / f"{case}.nii.gz")).dataobj).astype(np.uint8)

        for weight in weights:
            fused = weight * os50 + (1.0 - weight) * aura
            segmentation = keep_largest_components(np.argmax(fused, axis=0).astype(np.uint8))
            accumulator[weight].append(dice_per_class(segmentation, reference))
        print(f"[{case}] scored", flush=True)

    results = []
    for weight in weights:
        per_class = np.nanmean(np.stack(accumulator[weight]), axis=0)
        results.append({
            "w_os50": weight,
            "mean_overall": float(np.nanmean(per_class)),
            "hippo_mean": float(np.nanmean([per_class[c - 1] for c in HIPPOCAMPI])),
            "mean_per_class": [float(x) for x in per_class],
        })

    best = max(results, key=lambda row: row["mean_overall"])
    for row in results:
        marker = "  <- best" if row is best else ""
        print(f"w_os50={row['w_os50']:.1f}  DSC={row['mean_overall']:.6f}"
              f"  hippocampi={row['hippo_mean']:.6f}{marker}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=1))
        print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
