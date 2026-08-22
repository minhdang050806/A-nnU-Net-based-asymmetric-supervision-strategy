#!/usr/bin/env python3
"""Generate Figure 3: qualitative OS50 vs. OS50+AURA comparison.

Writes ``paper/figures/qualitative_os50_aura.{pdf,png}``. The three displayed
cases follow a pre-specified selection rule rather than visual choice:
LISA_0041 is nearest the median ensemble DSC, LISA_0043 has the largest
positive DSC change over OS50, and LISA_0021 has the lowest ensemble DSC. For
each case the axial slice with the most reference foreground voxels is shown.
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import nibabel as nib
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CASES = (
    ("LISA_0041", "Representative"),
    ("LISA_0043", "Largest ensemble gain"),
    ("LISA_0021", "Failure case"),
)
DEFAULT_OUT_DIR = ROOT / "paper/figures"
TITLE_FONTSIZE = 15
ROW_LABEL_FONTSIZE = 13


def default_gt_dir() -> Path | None:
    raw = os.environ.get("nnUNet_raw")
    return Path(raw) / "Dataset001_LISA" / "labelsTr" if raw else None


def load(path: Path) -> np.ndarray:
    return np.asanyarray(nib.load(str(path)).dataobj)


def display_slice(volume: np.ndarray, z: int) -> np.ndarray:
    return np.rot90(volume[:, :, z])


def normalize(image: np.ndarray) -> np.ndarray:
    values = image[np.isfinite(image)]
    lo, hi = np.percentile(values, (1, 99))
    return np.clip((image - lo) / max(hi - lo, 1e-6), 0, 1)


def content_crop(
    image: np.ndarray,
    *segmentations: np.ndarray,
    threshold: float = 0.08,
    margin_fraction: float = 0.06,
) -> tuple[slice, slice]:
    """Crop black background while retaining the full brain and all labels."""
    height, width = image.shape
    brain = image > threshold
    rows = np.flatnonzero(brain.sum(axis=1) >= max(3, round(0.03 * width)))
    cols = np.flatnonzero(brain.sum(axis=0) >= max(3, round(0.03 * height)))
    if rows.size == 0 or cols.size == 0:
        return slice(0, height), slice(0, width)

    y0, y1 = int(rows[0]), int(rows[-1])
    x0, x1 = int(cols[0]), int(cols[-1])
    for segmentation in segmentations:
        foreground_y, foreground_x = np.nonzero(segmentation)
        if foreground_y.size:
            y0 = min(y0, int(foreground_y.min()))
            y1 = max(y1, int(foreground_y.max()))
            x0 = min(x0, int(foreground_x.min()))
            x1 = max(x1, int(foreground_x.max()))

    margin_y = max(4, round((y1 - y0 + 1) * margin_fraction))
    margin_x = max(4, round((x1 - x0 + 1) * margin_fraction))
    return (
        slice(max(0, y0 - margin_y), min(height, y1 + margin_y + 1)),
        slice(max(0, x0 - margin_x), min(width, x1 + margin_x + 1)),
    )


def label_cmap() -> ListedColormap:
    colors = [(0, 0, 0, 0)] + [plt.cm.tab20(i) for i in range(11)]
    return ListedColormap(colors)


def overlay_labels(ax, image: np.ndarray, labels: np.ndarray) -> None:
    ax.imshow(image, cmap="gray", vmin=0, vmax=1)
    masked = np.ma.masked_where(labels == 0, labels)
    ax.imshow(masked, cmap=label_cmap(), vmin=0, vmax=11, alpha=0.62, interpolation="nearest")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--image-dir", type=Path, default=ROOT / "Dataset",
                    help="Directory holding <case>_ciso.nii.gz")
    ap.add_argument("--gt-dir", type=Path, default=default_gt_dir(),
                    help="Reference labels (default: "
                         "$nnUNet_raw/Dataset001_LISA/labelsTr)")
    ap.add_argument("--os50-dir", type=Path, required=True,
                    help="OS50 fold-0 predictions as <case>.nii.gz")
    ap.add_argument("--ensemble-dir", type=Path, required=True,
                    help="OS50+AURA fold-0 predictions as <case>.nii.gz")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = ap.parse_args()
    if args.gt_dir is None:
        raise SystemExit("--gt-dir is required when nnUNet_raw is not set")
    IMAGE_DIR, GT_DIR = args.image_dir, args.gt_dir
    OS50_DIR, ENSEMBLE_DIR, OUT_DIR = args.os50_dir, args.ensemble_dir, args.out_dir

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(CASES), 5, figsize=(12.4, 7.2), constrained_layout=True)
    titles = ("ULF MRI", "Ground truth", "OS50", "OS50+AURA", "Ensemble errors")
    for ax, title in zip(axes[0], titles):
        ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight="bold")

    for row, (case_id, role) in enumerate(CASES):
        image = load(IMAGE_DIR / f"{case_id}_ciso.nii.gz")
        gt = load(GT_DIR / f"{case_id}.nii.gz").astype(np.uint8)
        os50 = load(OS50_DIR / f"{case_id}.nii.gz").astype(np.uint8)
        ensemble = load(ENSEMBLE_DIR / f"{case_id}.nii.gz").astype(np.uint8)
        z = int(np.argmax(np.count_nonzero(gt, axis=(0, 1))))

        image_s = normalize(display_slice(image, z))
        gt_s = display_slice(gt, z)
        os50_s = display_slice(os50, z)
        ensemble_s = display_slice(ensemble, z)
        crop = content_crop(image_s, gt_s, os50_s, ensemble_s)
        image_s = image_s[crop]
        gt_s = gt_s[crop]
        os50_s = os50_s[crop]
        ensemble_s = ensemble_s[crop]

        axes[row, 0].imshow(image_s, cmap="gray", vmin=0, vmax=1)
        overlay_labels(axes[row, 1], image_s, gt_s)
        overlay_labels(axes[row, 2], image_s, os50_s)
        overlay_labels(axes[row, 3], image_s, ensemble_s)

        axes[row, 4].imshow(image_s, cmap="gray", vmin=0, vmax=1)
        fn = (gt_s > 0) & (ensemble_s != gt_s)
        fp = (ensemble_s > 0) & (ensemble_s != gt_s)
        axes[row, 4].imshow(np.ma.masked_where(~fn, fn), cmap=ListedColormap(["red"]), alpha=0.72)
        axes[row, 4].imshow(np.ma.masked_where(~fp, fp), cmap=ListedColormap(["cyan"]), alpha=0.72)

        axes[row, 0].set_xlabel(
            f"{role}\n{case_id}, axial {z}",
            fontsize=ROW_LABEL_FONTSIZE,
            fontweight="semibold",
            linespacing=1.15,
            labelpad=8,
            loc="left",
        )
        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

    for suffix in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"qualitative_os50_aura.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
