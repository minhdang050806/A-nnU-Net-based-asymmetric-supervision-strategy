#!/usr/bin/env python3
"""Build the nnU-Net raw datasets used by the OS50 and AURA branches.

The released LISA Task 2 training data is a flat directory of

    LISA_XXXX_ciso.nii.gz      combined isotropic ULF T2 image
    LISA_XXXX_seg.nii.gz       HF-derived annotation (the scored target)
    LISA_XXXX_LF_seg.nii.gz    LF-edited annotation (optional, image-aligned)

Both branches consume the same images and the same HF labels in raw form:

    Dataset001_LISA      OS50 anchor
    Dataset002_LISA_VCF  AURA, identical raw content

The two datasets diverge only after preprocessing, where
``scripts/append_lf_labels.py`` stacks the LF annotation as a second
segmentation channel of Dataset002. Keeping them as separate nnU-Net dataset
IDs lets both models be trained, validated and predicted independently with the
stock nnU-Net CLI.

Usage:
    python scripts/prepare_lisa_datasets.py --data-dir /path/to/LISA/Dataset
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path

CASE_PATTERN = re.compile(r"^(LISA_\d+)_ciso\.nii\.gz$")

DATASETS = (
    ("Dataset001_LISA", "OS50 anchor: HF-supervised"),
    ("Dataset002_LISA_VCF", "AURA: HF raw labels, LF stacked at preprocessing"),
)

DATASET_JSON = {
    "channel_names": {"0": "ciso"},
    "labels": {
        "background": 0,
        **{f"class{i}": i for i in range(1, 12)},
    },
    "file_ending": ".nii.gz",
}


def discover_cases(data_dir: Path) -> list[str]:
    cases = sorted(
        match.group(1)
        for path in data_dir.iterdir()
        if (match := CASE_PATTERN.match(path.name))
    )
    if not cases:
        raise SystemExit(f"No LISA_*_ciso.nii.gz volumes found in {data_dir}")
    return cases


def place(source: Path, destination: Path, *, copy: bool) -> None:
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    if copy:
        shutil.copy2(source, destination)
    else:
        destination.symlink_to(source)


def build(data_dir: Path, raw_root: Path, name: str, cases: list[str], copy: bool) -> None:
    dataset_dir = raw_root / name
    images = dataset_dir / "imagesTr"
    labels = dataset_dir / "labelsTr"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)

    for case in cases:
        place(data_dir / f"{case}_ciso.nii.gz", images / f"{case}_0000.nii.gz", copy=copy)
        place(data_dir / f"{case}_seg.nii.gz", labels / f"{case}.nii.gz", copy=copy)

    metadata = {**DATASET_JSON, "numTraining": len(cases)}
    (dataset_dir / "dataset.json").write_text(json.dumps(metadata, indent=4) + "\n")
    print(f"  {name}: {len(cases)} cases -> {dataset_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Directory holding the released LISA NIfTI volumes")
    parser.add_argument("--raw-root", type=Path, default=None,
                        help="nnU-Net raw root (default: $nnUNet_raw)")
    parser.add_argument("--copy", action="store_true",
                        help="Copy files instead of symlinking them")
    args = parser.parse_args()

    raw_root = args.raw_root
    if raw_root is None:
        env = os.environ.get("nnUNet_raw")
        if not env:
            raise SystemExit("Pass --raw-root or set the nnUNet_raw environment variable")
        raw_root = Path(env)

    data_dir = args.data_dir.resolve()
    cases = discover_cases(data_dir)
    print(f"Found {len(cases)} cases in {data_dir}")

    missing_lf = [c for c in cases if not (data_dir / f"{c}_LF_seg.nii.gz").is_file()]
    if missing_lf:
        print(
            f"[warn] {len(missing_lf)} case(s) have no _LF_seg.nii.gz; "
            f"scripts/append_lf_labels.py will fail for them: {missing_lf[:5]}"
        )

    for name, purpose in DATASETS:
        print(f"{purpose}")
        build(data_dir, raw_root.resolve(), name, cases, copy=args.copy)

    print("\nNext: nnUNetv2_plan_and_preprocess -d 1 2 --verify_dataset_integrity")


if __name__ == "__main__":
    main()
