#!/usr/bin/env python3
"""Generate Figure 1: representative HF/LF annotation disagreement examples.

Reads the released LISA volumes (``<case>_ciso.nii.gz``, ``<case>_seg.nii.gz``,
``<case>_LF_seg.nii.gz``) from a challenge data directory and writes
``paper/figures/dataset_hf_lf_disagreement.{pdf,png}``.

The two displayed cases were selected quantitatively, not visually:
LISA_0028 is the median training case by macro HF/LF Dice over the 11 labels,
and LISA_0044 has both the lowest macro-Dice and the largest disagreement
fraction. For each case the axial slice with the most disagreeing voxels is
shown.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import nibabel as nib
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "Dataset"
DEFAULT_OUT = ROOT / "paper/figures/dataset_hf_lf_disagreement.pdf"
CASES = (
    ("LISA_0028", "Representative", 0.8616, 0.2407),
    ("LISA_0044", "High mismatch", 0.6616, 0.3807),
)
TITLE_FONTSIZE = 14
ROW_LABEL_FONTSIZE = 12


def load(path: Path) -> np.ndarray:
    return np.asanyarray(nib.load(str(path)).dataobj)


def norm(image: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(image[np.isfinite(image)], (1, 99))
    return np.clip((image - lo) / max(hi - lo, 1e-6), 0, 1)


def sl(volume: np.ndarray, z: int) -> np.ndarray:
    return np.rot90(volume[:, :, z])


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


def labels(ax, image, seg):
    colors = [(0, 0, 0, 0)] + [plt.cm.tab20(i) for i in range(11)]
    ax.imshow(image, cmap="gray", vmin=0, vmax=1)
    ax.imshow(np.ma.masked_where(seg == 0, seg), cmap=ListedColormap(colors),
              vmin=0, vmax=11, alpha=0.62, interpolation="nearest")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA,
                    help="Directory holding the released LISA NIfTI volumes")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help="Output PDF path; a PNG is written alongside it")
    args = ap.parse_args()
    DATA, OUT = args.data_dir, args.out

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 5, figsize=(12.2, 5.1), constrained_layout=True)
    for ax, title in zip(
        axes[0],
        (
            "ULF MRI",
            "HF-derived\ntarget",
            "LF-edited\nmask",
            "HF/LF foreground\ncontours",
            "Annotation\ndisagreement",
        ),
    ):
        ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight="bold", linespacing=0.95)

    for row, (case_id, role, macro_dsc, disagreement_rate) in enumerate(CASES):
        image = load(DATA / f"{case_id}_ciso.nii.gz")
        hf = load(DATA / f"{case_id}_seg.nii.gz").astype(np.uint8)
        lf = load(DATA / f"{case_id}_LF_seg.nii.gz").astype(np.uint8)
        union = (hf > 0) | (lf > 0)
        disagreement = (hf != lf) & union
        z = int(np.argmax(np.count_nonzero(disagreement, axis=(0, 1))))

        image_s, hf_s, lf_s = norm(sl(image, z)), sl(hf, z), sl(lf, z)
        crop = content_crop(image_s, hf_s, lf_s)
        image_s, hf_s, lf_s = image_s[crop], hf_s[crop], lf_s[crop]
        axes[row, 0].imshow(image_s, cmap="gray", vmin=0, vmax=1)
        labels(axes[row, 1], image_s, hf_s)
        labels(axes[row, 2], image_s, lf_s)

        axes[row, 3].imshow(image_s, cmap="gray", vmin=0, vmax=1)
        axes[row, 3].contour(hf_s > 0, levels=[0.5], colors=["#228833"], linewidths=1.5)
        axes[row, 3].contour(lf_s > 0, levels=[0.5], colors=["#AA3377"], linewidths=1.3,
                             linestyles="--")

        hf_only = (hf_s > 0) & (lf_s == 0)
        lf_only = (lf_s > 0) & (hf_s == 0)
        class_shift = (hf_s > 0) & (lf_s > 0) & (hf_s != lf_s)
        disagreement_rgb = np.zeros((*hf_s.shape, 4), dtype=float)
        disagreement_rgb[hf_only] = (0.84, 0.15, 0.16, 0.78)
        disagreement_rgb[lf_only] = (0.0, 0.62, 0.72, 0.78)
        disagreement_rgb[class_shift] = (0.95, 0.65, 0.08, 0.82)
        axes[row, 4].imshow(image_s, cmap="gray", vmin=0, vmax=1)
        axes[row, 4].imshow(disagreement_rgb, interpolation="nearest")

        axes[row, 0].set_xlabel(
            f"{role}\n{case_id}, axial {z}\nmacro-DSC={macro_dsc:.4f}, diff={100*disagreement_rate:.1f}%",
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

    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
