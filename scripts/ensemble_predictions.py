#!/usr/bin/env python3
"""Fuse nnU-Net probability maps into LISA Task 2 prediction files.

Weighted soft-probability averaging, argmax, optional per-class largest
connected component filtering, and a flat zip named for challenge upload. The
paper configuration is two probability directories at weights 0.6 / 0.4 with
LCC on labels 1-11 and no probability rescaling.

The optional `--fg-scale`, `--class-scales` and `--target-volume-ratios` knobs
were explored during the challenge cycle but are NOT part of the reported
system; leave them at their defaults to reproduce the paper.

Every run writes a `qc_report.json` recording the inputs, weights, per-case
label histograms and NIfTI geometry, so a submission can be audited after the
fact.

Example (paper configuration):
    python scripts/ensemble_predictions.py \
        --prob-dirs predictions/os50_fold0_probs predictions/aura_fold0_probs \
        --weights 0.6,0.4 \
        --lcc-labels 1,2,3,4,5,6,7,8,9,10,11 \
        --images-dir /path/to/LISA/Dataset \
        --out-dir predictions/os50_aura_w06
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage


LABELS = tuple(range(12))


def parse_float_list(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def parse_int_list(text: str | None) -> list[int]:
    if not text:
        return []
    return [int(x) for x in text.split(",") if x.strip()]


def parse_class_scales(text: str | None) -> dict[int, float]:
    if not text:
        return {}
    scales: dict[int, float] = {}
    for item in text.split(","):
        if not item.strip():
            continue
        label_text, scale_text = item.split(":", 1)
        label_id = int(label_text)
        if label_id not in LABELS:
            raise ValueError(f"Invalid label in --class-scales: {label_id}")
        scales[label_id] = float(scale_text)
    return scales


def parse_label_float_map(text: str | None) -> dict[int, float]:
    return parse_class_scales(text)


def find_reference(images_dir: Path, case_id: str) -> Path:
    candidates = [
        images_dir / f"{case_id}_ciso.nii.gz",
        images_dir / f"{case_id}_0000.nii.gz",
        images_dir / f"{case_id}.nii.gz",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"No reference image found for {case_id} in {images_dir}")


def load_prob(path: Path) -> np.ndarray:
    with np.load(path) as data:
        if "probabilities" not in data:
            raise KeyError(f"{path} does not contain 'probabilities'")
        probs = data["probabilities"].astype(np.float32, copy=False)
    if probs.ndim != 4:
        raise ValueError(f"{path}: expected CZYX probabilities, got shape {probs.shape}")
    if probs.shape[0] != len(LABELS):
        raise ValueError(f"{path}: expected 12 channels, got {probs.shape[0]}")
    return probs


def largest_connected_components(seg: np.ndarray, labels: list[int]) -> np.ndarray:
    if not labels:
        return seg
    out = seg.copy()
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    for label_id in labels:
        mask = out == label_id
        cc, num = ndimage.label(mask, structure=structure)
        if num <= 1:
            continue
        counts = np.bincount(cc.ravel())
        counts[0] = 0
        keep = counts.argmax()
        out[(cc != 0) & (cc != keep)] = 0
    return out


def trim_low_margin_voxels(
    seg: np.ndarray,
    probs: np.ndarray,
    target_volume_ratios: dict[int, float],
) -> np.ndarray:
    if not target_volume_ratios:
        return seg
    out = seg.copy()
    flat_out = out.ravel()
    for label_id, ratio in sorted(target_volume_ratios.items()):
        if label_id == 0:
            continue
        if not (0.0 < ratio <= 1.0):
            raise ValueError(f"Target volume ratio must be in (0, 1], got {label_id}:{ratio}")
        mask = out == label_id
        current = int(np.count_nonzero(mask))
        target = int(round(current * ratio))
        remove = current - target
        if remove <= 0:
            continue
        coords = np.flatnonzero(mask.ravel())
        voxel_probs = probs[:, mask].copy()
        label_probs = voxel_probs[label_id].copy()
        voxel_probs[label_id] = -np.inf
        margin = label_probs - voxel_probs.max(axis=0)
        remove_local = np.argpartition(margin, remove - 1)[:remove]
        flat_out[coords[remove_local]] = 0
    return out


def combine_case(
    case_id: str,
    prob_dirs: list[Path],
    weights: np.ndarray,
    fg_scale: float,
    class_scales: dict[int, float],
    target_volume_ratios: dict[int, float],
    lcc_labels: list[int],
) -> np.ndarray:
    combined: np.ndarray | None = None
    for prob_dir, weight in zip(prob_dirs, weights):
        probs = load_prob(prob_dir / f"{case_id}.npz")
        if combined is None:
            combined = np.zeros_like(probs, dtype=np.float32)
        elif combined.shape != probs.shape:
            raise ValueError(f"{case_id}: probability shape mismatch {combined.shape} vs {probs.shape}")
        combined += weight * probs
    assert combined is not None
    if fg_scale != 1.0:
        combined[1:] *= float(fg_scale)
    for label_id, scale in class_scales.items():
        combined[label_id] *= float(scale)
    seg = np.argmax(combined, axis=0).astype(np.uint8)
    seg = trim_low_margin_voxels(seg, combined, target_volume_ratios)
    return largest_connected_components(seg, lcc_labels)


def label_counts(seg: np.ndarray) -> dict[str, int]:
    values, counts = np.unique(seg, return_counts=True)
    return {str(int(v)): int(c) for v, c in zip(values, counts)}


def read_seg_array(path: Path) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))


def write_case(seg: np.ndarray, ref_path: Path, out_path: Path) -> dict:
    ref_img = sitk.ReadImage(str(ref_path))
    ref_arr = sitk.GetArrayFromImage(ref_img)
    if tuple(seg.shape) != tuple(ref_arr.shape):
        raise ValueError(f"{out_path.name}: seg shape {seg.shape} != ref shape {ref_arr.shape}")
    out_img = sitk.GetImageFromArray(seg.astype(np.uint8, copy=False))
    out_img.CopyInformation(ref_img)
    sitk.WriteImage(out_img, str(out_path), useCompression=True)
    check_img = sitk.ReadImage(str(out_path))
    return {
        "ref_path": str(ref_path),
        "shape": list(seg.shape),
        "spacing": list(ref_img.GetSpacing()),
        "origin": list(ref_img.GetOrigin()),
        "direction": list(ref_img.GetDirection()),
        "out_spacing": list(check_img.GetSpacing()),
        "out_size": list(check_img.GetSize()),
    }


def make_zip(pred_files: list[Path], zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(pred_files):
            zf.write(path, arcname=path.name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prob-dirs", nargs="+", required=True, type=Path)
    parser.add_argument("--weights", required=True, help="Comma-separated weights matching --prob-dirs")
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--fg-scale", type=float, default=1.0)
    parser.add_argument(
        "--class-scales",
        default="",
        help="Comma-separated label:scale probability multipliers applied after --fg-scale",
    )
    parser.add_argument(
        "--target-volume-ratios",
        default="",
        help="Comma-separated label:ratio map. Removes low-margin voxels after argmax to match per-class ratios.",
    )
    parser.add_argument("--lcc-labels", default="", help="Comma-separated labels for per-class LCC filtering")
    parser.add_argument("--compare-seg-dir", type=Path, default=None)
    args = parser.parse_args()

    prob_dirs = [p.resolve() for p in args.prob_dirs]
    weights = np.array(parse_float_list(args.weights), dtype=np.float32)
    if len(weights) != len(prob_dirs):
        raise ValueError("--weights length must match --prob-dirs")
    if np.any(weights < 0):
        raise ValueError("Weights must be non-negative")
    weights = weights / weights.sum()

    case_ids = sorted(p.stem for p in prob_dirs[0].glob("*.npz"))
    if not case_ids:
        raise RuntimeError(f"No .npz files found in {prob_dirs[0]}")
    for prob_dir in prob_dirs[1:]:
        other = sorted(p.stem for p in prob_dir.glob("*.npz"))
        if other != case_ids:
            raise ValueError(f"Case list mismatch in {prob_dir}")

    out_dir = args.out_dir.resolve()
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    lcc_labels = parse_int_list(args.lcc_labels)
    class_scales = parse_class_scales(args.class_scales)
    target_volume_ratios = parse_label_float_map(args.target_volume_ratios)

    report = {
        "prob_dirs": [str(p) for p in prob_dirs],
        "weights": [float(x) for x in weights],
        "fg_scale": float(args.fg_scale),
        "class_scales": {str(k): float(v) for k, v in sorted(class_scales.items())},
        "target_volume_ratios": {str(k): float(v) for k, v in sorted(target_volume_ratios.items())},
        "lcc_labels": lcc_labels,
        "cases": {},
    }
    pred_files: list[Path] = []
    for case_id in case_ids:
        seg = combine_case(
            case_id,
            prob_dirs,
            weights,
            args.fg_scale,
            class_scales,
            target_volume_ratios,
            lcc_labels,
        )
        ref_path = find_reference(args.images_dir, case_id)
        out_path = pred_dir / f"{case_id}_seg_prediction.nii.gz"
        meta = write_case(seg, ref_path, out_path)
        counts = label_counts(seg)
        case_report = {
            **meta,
            "labels": sorted(int(x) for x in np.unique(seg)),
            "voxel_counts": counts,
            "foreground_voxels": int(np.count_nonzero(seg)),
        }
        if args.compare_seg_dir is not None:
            base_path = args.compare_seg_dir / f"{case_id}.nii.gz"
            if not base_path.is_file():
                base_path = args.compare_seg_dir / f"{case_id}_seg_prediction.nii.gz"
            if base_path.is_file():
                base = read_seg_array(base_path)
                base_fg = int(np.count_nonzero(base))
                case_report["compare_path"] = str(base_path)
                case_report["compare_foreground_voxels"] = base_fg
                case_report["foreground_ratio_vs_compare"] = (
                    float(np.count_nonzero(seg) / base_fg) if base_fg else None
                )
                case_report["changed_voxels_vs_compare"] = int(np.count_nonzero(seg != base))
        report["cases"][case_id] = case_report
        pred_files.append(out_path)

    zip_path = out_dir / "LISA_SEG_predictions.zip"
    make_zip(pred_files, zip_path)
    report["zip_path"] = str(zip_path)
    report["num_files"] = len(pred_files)

    report_path = out_dir / "qc_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"zip_path": str(zip_path), "report_path": str(report_path), "num_files": len(pred_files)}, indent=2))


if __name__ == "__main__":
    main()
