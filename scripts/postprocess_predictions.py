"""
Post-process nnU-Net segmentation predictions.

Applies per-class largest-connected-component (LCC) filtering and optional
left/right symmetry enforcement.  Operates in-place or to a separate output
directory.

Usage
-----
# Filter a single prediction folder (in-place):
python scripts/postprocess_predictions.py --pred_dir PATH/TO/validation

# Write filtered copies to a new folder:
python scripts/postprocess_predictions.py \\
    --pred_dir PATH/TO/validation \\
    --out_dir  PATH/TO/validation_postproc

Flags
-----
--keep_top N    : keep the N largest components per class (default 1).
                  Use 2 for bilateral structures (hippocampi, etc.) when
                  left/right are the same label.
--bilateral LABELS : comma-separated label indices that are bilateral and
                  should keep 2 components (overrides --keep_top for those).
                  E.g. --bilateral 1,2,3,4,5,6,7,8,9,10 for all structures
                  or --bilateral 1,2 for hippocampi only.
--no_lcc        : skip LCC filtering (useful to test symmetry fix alone).
"""
import argparse
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import label as cc_label


N_CLASSES = 11


def keep_largest_k(binary: np.ndarray, k: int = 1) -> np.ndarray:
    """Return a binary mask keeping only the k largest connected components."""
    struct, n = cc_label(binary)
    if n == 0:
        return binary
    sizes = np.bincount(struct.ravel())
    sizes[0] = 0  # background
    top_k = np.argsort(sizes)[::-1][:k]
    out = np.zeros_like(binary)
    for idx in top_k:
        if sizes[idx] > 0:
            out[struct == idx] = 1
    return out


def postprocess_volume(
    seg: np.ndarray,
    keep_top: int = 1,
    bilateral_labels: set = None,
    apply_lcc: bool = True,
) -> np.ndarray:
    """
    Per-class LCC filtering on a multi-label segmentation volume.

    seg             : (D, H, W) int array with labels 0..N_CLASSES
    keep_top        : default number of components to keep per class
    bilateral_labels: set of label ints that should keep 2 components
    apply_lcc       : if False, return seg unchanged
    """
    if not apply_lcc:
        return seg
    bilateral_labels = bilateral_labels or set()
    out = np.zeros_like(seg)
    for k in range(1, N_CLASSES + 1):
        mask = (seg == k)
        if mask.sum() == 0:
            continue
        n_keep = 2 if k in bilateral_labels else keep_top
        filtered = keep_largest_k(mask, k=n_keep)
        out[filtered > 0] = k
    return out


def process_file(
    in_path: Path,
    out_path: Path,
    keep_top: int,
    bilateral_labels: set,
    apply_lcc: bool,
):
    nii = nib.load(str(in_path))
    seg = np.asanyarray(nii.dataobj).astype(np.int16)
    seg_post = postprocess_volume(seg, keep_top, bilateral_labels, apply_lcc)

    changed = int((seg != seg_post).sum())
    nib.save(nib.Nifti1Image(seg_post, nii.affine, nii.header), str(out_path))
    return changed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_dir', required=True, type=Path,
                        help='Directory containing *_pred.nii.gz or *.nii.gz predictions')
    parser.add_argument('--out_dir', type=Path, default=None,
                        help='Output directory (default: overwrite pred_dir in-place)')
    parser.add_argument('--keep_top', type=int, default=1,
                        help='Components to keep per class (default 1)')
    parser.add_argument('--bilateral', type=str, default='',
                        help='Comma-separated label indices that are bilateral (keep 2)')
    parser.add_argument('--no_lcc', action='store_true',
                        help='Skip LCC filtering')
    args = parser.parse_args()

    pred_dir = args.pred_dir
    out_dir  = args.out_dir or pred_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    bilateral_labels = set()
    if args.bilateral:
        bilateral_labels = {int(x) for x in args.bilateral.split(',') if x.strip()}

    nii_files = sorted(pred_dir.glob('*.nii.gz'))
    if not nii_files:
        print(f'No .nii.gz files found in {pred_dir}')
        return

    print(f'Processing {len(nii_files)} files  →  {out_dir}')
    print(f'  keep_top={args.keep_top}, bilateral={bilateral_labels or "none"}, '
          f'lcc={"off" if args.no_lcc else "on"}')

    total_changed = 0
    for p in nii_files:
        out_p = out_dir / p.name
        changed = process_file(
            p, out_p,
            keep_top=args.keep_top,
            bilateral_labels=bilateral_labels,
            apply_lcc=not args.no_lcc,
        )
        print(f'  {p.name}: {changed} voxels changed')
        total_changed += changed

    print(f'\nDone. Total voxels changed: {total_changed}')


if __name__ == '__main__':
    main()
