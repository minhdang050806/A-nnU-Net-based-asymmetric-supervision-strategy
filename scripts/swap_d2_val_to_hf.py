"""Swap Dataset002 (HF+LF 2-channel seg) val cases so channel 1 = channel 0 = HF.

Effect: nnU-Net's online validation Dice — which scatters all target channels
into y_onehot — becomes HF-only, because both channels carry the same HF
labels at the validation cases. Training cases keep their original (HF, LF)
pair, so the AURA loss still sees both observations. The EMA pseudo-Dice and
`checkpoint_best.pth` selection therefore track the scored HF target rather
than a mixture of the two annotations.

Reversible: original `_seg.b2nd` files for val cases are moved out of the
preprocessed-config folder to `Dataset002_LISA_VCF/_d2_val_2ch_backup/`.

Requires the nnUNet_preprocessed environment variable.

Usage:
    python scripts/swap_d2_val_to_hf.py            # do swap
    python scripts/swap_d2_val_to_hf.py --restore  # undo
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import blosc2
import numpy as np

BACKUP_NAME = '_d2_val_2ch_backup'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--restore', action='store_true')
    ap.add_argument('--dataset', default='Dataset002_LISA_VCF',
                    help='Preprocessed dataset folder name')
    ap.add_argument('--configuration', default='nnUNetPlans_3d_fullres',
                    help='Preprocessed configuration folder name')
    ap.add_argument('--fold', type=int, default=0)
    args = ap.parse_args()

    preprocessed = os.environ.get('nnUNet_preprocessed')
    if not preprocessed:
        raise SystemExit('nnUNet_preprocessed is not set')
    dataset_dir = Path(preprocessed) / args.dataset
    D2_PREP = dataset_dir / args.configuration
    BACKUP_DIR = dataset_dir / BACKUP_NAME
    SPLITS = dataset_dir / 'splits_final.json'

    splits = json.loads(SPLITS.read_text())
    val_keys = splits[args.fold]['val']
    print(f'fold {args.fold} val ({len(val_keys)}): {val_keys}')

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    if args.restore:
        for k in val_keys:
            d2_seg = D2_PREP / f'{k}_seg.b2nd'
            backup = BACKUP_DIR / f'{k}_seg.b2nd'
            if not backup.exists():
                print(f'  [skip] {k}')
                continue
            if d2_seg.exists():
                d2_seg.unlink()
            shutil.move(str(backup), str(d2_seg))
            print(f'  [restored] {k}')
        return

    n_swapped, n_already = 0, 0
    for k in val_keys:
        d2_seg = D2_PREP / f'{k}_seg.b2nd'
        backup = BACKUP_DIR / f'{k}_seg.b2nd'
        assert d2_seg.exists(), f'missing seg: {d2_seg}'

        if backup.exists():
            print(f'  [already swapped] {k}')
            n_already += 1
            continue

        # Read original 2-channel seg
        seg = blosc2.open(urlpath=str(d2_seg))[:]
        assert seg.ndim == 4 and seg.shape[0] == 2, f'unexpected shape: {seg.shape}'

        # Build new seg where ch1 := ch0 (HF copied into LF position)
        new_seg = np.stack([seg[0], seg[0]], axis=0)

        # Backup original first (move outside preprocessed-config folder)
        shutil.move(str(d2_seg), str(backup))

        # Save new seg with same blosc2 codec
        # Match the chunk + cparams of the original via blosc2.asarray on the array
        b = blosc2.asarray(new_seg, urlpath=str(d2_seg), mode='w')
        del b
        n_swapped += 1
        print(f'  [swapped] {k}')

    print(f'\nDone: {n_swapped} swapped / {n_already} already / {len(val_keys)} total')
    print('Resume training with --c. Per-epoch validation Dice is now the HF metric.')
    print('Reverse later: python scripts/swap_d2_val_to_hf.py --restore')


if __name__ == '__main__':
    main()
