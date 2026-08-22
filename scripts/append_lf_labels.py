"""
Append low-field (_LF_seg) labels as a SECOND channel of each preprocessed
*_seg.b2nd file produced by nnUNetv2.

After running nnUNetv2_plan_and_preprocess the preprocessed folder contains:
    LISA_XXXX.b2nd              (image, float32)
    LISA_XXXX_seg.b2nd          (HF seg, shape (1, Z, Y, X), int)
    LISA_XXXX.pkl               (properties)

This script re-runs the default preprocessor on each case using the *LF* label
as the seg input (with the same image so cropping/resampling are identical),
extracts the processed LF seg and concatenates it so the result becomes:
    LISA_XXXX_seg.b2nd          shape (2, Z, Y, X)
        channel 0 = HF  (y_HF  = main target)
        channel 1 = LF  (y_LF  = visible anatomy)

Usage:
    python scripts/append_lf_labels.py \
        --dataset_id 2 \
        --configuration 3d_fullres \
        --lf_label_dir /path/to/LISA/Dataset \
        --lf_suffix _LF_seg.nii.gz
"""
import argparse
import os
import shutil
from os.path import join, isfile

import blosc2
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import load_json, load_pickle

from nnunetv2.paths import nnUNet_preprocessed, nnUNet_raw
from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.utilities.utils import get_filenames_of_train_images_and_targets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_id', type=str, required=True)
    ap.add_argument('--configuration', type=str, default='3d_fullres')
    ap.add_argument('--plans_identifier', type=str, default='nnUNetPlans')
    ap.add_argument('--lf_label_dir', type=str, required=True,
                    help='Directory that holds the LF nifti labels (one per case).')
    ap.add_argument('--lf_suffix', type=str, default='_LF_seg.nii.gz',
                    help='Suffix applied to the case identifier to locate the LF file.')
    args = ap.parse_args()

    ds_name = maybe_convert_to_dataset_name(args.dataset_id)
    preproc_root = join(nnUNet_preprocessed, ds_name)
    plans = load_json(join(preproc_root, args.plans_identifier + '.json'))
    dataset_json = load_json(join(preproc_root, 'dataset.json'))

    pm = PlansManager(plans)
    cm = pm.get_configuration(args.configuration)
    out_dir = join(preproc_root, cm.data_identifier)
    assert os.path.isdir(out_dir), f'Preprocessed folder missing: {out_dir}'

    raw_ds = get_filenames_of_train_images_and_targets(join(nnUNet_raw, ds_name), dataset_json)

    pp = DefaultPreprocessor(verbose=False)

    for case_id in sorted(raw_ds.keys()):
        image_files = raw_ds[case_id]['images']
        lf_file = join(args.lf_label_dir, case_id + args.lf_suffix)
        assert isfile(lf_file), f'Missing LF label for {case_id}: {lf_file}'

        seg_b2nd = join(out_dir, case_id + '_seg.b2nd')
        pkl = join(out_dir, case_id + '.pkl')
        assert isfile(seg_b2nd) and isfile(pkl), f'Preprocessed case missing: {case_id}'

        # already stacked? skip
        existing = blosc2.open(urlpath=seg_b2nd, mode='r')[:]
        if existing.shape[0] >= 2:
            print(f'[{case_id}] seg already has {existing.shape[0]} channels, skipping')
            continue

        # Run the default preprocessor with the IMAGE + LF seg. This reproduces
        # the same crop/resample pipeline so LF ends up spatially aligned with HF.
        _, lf_seg, _ = pp.run_case(image_files, lf_file, pm, cm, dataset_json)
        # lf_seg: (1, Z, Y, X)

        if lf_seg.shape != existing.shape:
            raise RuntimeError(f'[{case_id}] shape mismatch HF {existing.shape} vs LF {lf_seg.shape}')

        stacked = np.concatenate([existing.astype(lf_seg.dtype), lf_seg], axis=0)
        # channel 0 = HF, channel 1 = LF

        # compute chunk/block params on the stacked shape
        blocks, chunks = nnUNetDatasetBlosc2.comp_blosc2_params(
            stacked.shape, tuple(cm.patch_size), stacked.itemsize)

        tmp = seg_b2nd + '.tmp'
        if os.path.exists(tmp):
            shutil.rmtree(tmp) if os.path.isdir(tmp) else os.remove(tmp)
        blosc2.asarray(np.ascontiguousarray(stacked), urlpath=tmp,
                       chunks=chunks, blocks=blocks,
                       cparams={'codec': blosc2.Codec.LZ4HC, 'clevel': 8})
        os.replace(tmp, seg_b2nd)
        print(f'[{case_id}] stacked HF+LF seg -> {stacked.shape}, dtype {stacked.dtype}')


if __name__ == '__main__':
    main()
