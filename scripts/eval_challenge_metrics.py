"""Evaluate LISA segmentation predictions with challenge-style metrics.

Computes DSC, HD, HD95, ASSD, and RVE for selected labels, defaulting to the
hippocampi labels used by the LISA segmentation ranking text. Metrics are
computed per case and per label, then macro-averaged over all (case, label)
pairs; non-finite distance values are dropped from the mean.

The reference labels default to ``$nnUNet_raw/Dataset001_LISA/labelsTr``.

Example:
    python scripts/eval_challenge_metrics.py \
        --pred_dir predictions/os50_fold0 \
        --labels 1,2,3,4,5,6,7,8,9,10,11 \
        --out_json results/metrics/os50_all11.json \
        --out_csv  results/metrics/os50_all11.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure


DEFAULT_LABELS = (1, 2)


def default_gt_dir() -> Path | None:
    """Reference labels for the scored HF target, taken from $nnUNet_raw."""
    raw = os.environ.get("nnUNet_raw")
    return Path(raw) / "Dataset001_LISA" / "labelsTr" if raw else None


def load_seg(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    nii = nib.load(str(path))
    seg = np.asanyarray(nii.dataobj).astype(np.int16)
    spacing = tuple(float(x) for x in nii.header.get_zooms()[:3])
    return seg, spacing


def binary_surface(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return mask.astype(bool)
    structure = generate_binary_structure(mask.ndim, 1)
    eroded = binary_erosion(mask, structure=structure, border_value=0)
    return mask & ~eroded


def surface_distances(a: np.ndarray, b: np.ndarray, spacing: tuple[float, float, float]) -> np.ndarray:
    """Distances from surface voxels of a to the nearest surface voxel of b."""
    a_surf = binary_surface(a)
    b_surf = binary_surface(b)
    if not a_surf.any() or not b_surf.any():
        return np.array([], dtype=np.float64)
    # distance_transform_edt computes distance to zero elements, so invert b_surf.
    dt = distance_transform_edt(~b_surf, sampling=spacing)
    return dt[a_surf]


def label_metrics(pred: np.ndarray, ref: np.ndarray, label: int, spacing: tuple[float, float, float]) -> dict:
    p = pred == label
    r = ref == label
    p_count = int(p.sum())
    r_count = int(r.sum())
    inter = int((p & r).sum())

    if p_count + r_count == 0:
        dice = 1.0
    elif p_count == 0 or r_count == 0:
        dice = 0.0
    else:
        dice = 2.0 * inter / (p_count + r_count)

    if r_count == 0:
        rve = math.nan
        abs_rve = math.nan
    else:
        rve = (p_count - r_count) / r_count
        abs_rve = abs(rve)

    if p_count and r_count:
        coords = np.nonzero(p | r)
        crop = tuple(
            slice(max(0, int(axis.min()) - 1), min(p.shape[i], int(axis.max()) + 2))
            for i, axis in enumerate(coords)
        )
        p_surface_input = p[crop]
        r_surface_input = r[crop]
    else:
        p_surface_input = p
        r_surface_input = r

    d_pr = surface_distances(p_surface_input, r_surface_input, spacing)
    d_rp = surface_distances(r_surface_input, p_surface_input, spacing)
    if d_pr.size == 0 or d_rp.size == 0:
        hd = math.inf
        hd95 = math.inf
        assd = math.inf
    else:
        all_d = np.concatenate([d_pr, d_rp])
        hd = float(np.max(all_d))
        hd95 = float(np.percentile(all_d, 95))
        assd = float(np.mean(all_d))

    return {
        "DSC": float(dice),
        "HD": hd,
        "HD95": hd95,
        "ASSD": assd,
        "RVE": float(rve) if not math.isnan(rve) else math.nan,
        "absRVE": float(abs_rve) if not math.isnan(abs_rve) else math.nan,
        "n_pred": p_count,
        "n_ref": r_count,
    }


def finite_mean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else math.inf


def parse_labels(raw: str) -> tuple[int, ...]:
    labels = tuple(int(x) for x in raw.split(",") if x.strip())
    if not labels:
        raise ValueError("At least one label is required")
    return labels


def find_gt(case_id: str, gt_dir: Path) -> Path:
    gt = gt_dir / f"{case_id}.nii.gz"
    if not gt.exists():
        raise FileNotFoundError(f"Missing GT for {case_id}: {gt}")
    return gt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True, type=Path,
                    help="Folder containing prediction .nii.gz files")
    ap.add_argument("--gt_dir", type=Path, default=default_gt_dir(),
                    help="Reference labels folder "
                         "(default: $nnUNet_raw/Dataset001_LISA/labelsTr)")
    ap.add_argument("--labels", default=",".join(str(x) for x in DEFAULT_LABELS),
                    help="Comma-separated labels to evaluate, default hippocampi 1,2")
    ap.add_argument("--pred_suffix", default="_seg_prediction",
                    help="Suffix stripped from prediction filenames to recover the "
                         "case ID; challenge-named files use _seg_prediction")
    ap.add_argument("--out_json", type=Path, default=None)
    ap.add_argument("--out_csv", type=Path, default=None)
    args = ap.parse_args()

    if args.gt_dir is None:
        raise SystemExit(
            "--gt_dir is required when the nnUNet_raw environment variable is not set"
        )

    labels = parse_labels(args.labels)
    pred_files = sorted(args.pred_dir.glob("*.nii.gz"))
    if not pred_files:
        raise FileNotFoundError(f"No .nii.gz predictions found in {args.pred_dir}")

    rows = []
    case_summaries = []
    for pred_path in pred_files:
        case_id = pred_path.name[:-7] if pred_path.name.endswith(".nii.gz") else pred_path.stem
        if args.pred_suffix and case_id.endswith(args.pred_suffix):
            case_id = case_id[: -len(args.pred_suffix)]
        gt_path = find_gt(case_id, args.gt_dir)
        pred, spacing = load_seg(pred_path)
        gt, gt_spacing = load_seg(gt_path)
        if pred.shape != gt.shape:
            raise ValueError(f"Shape mismatch for {case_id}: pred {pred.shape}, gt {gt.shape}")
        if not np.allclose(spacing, gt_spacing, atol=1e-4):
            print(f"[warn] spacing mismatch for {case_id}: pred={spacing}, gt={gt_spacing}; using pred spacing")

        per_label = {}
        for label in labels:
            m = label_metrics(pred, gt, label, spacing)
            per_label[str(label)] = m
            rows.append({"case": case_id, "label": label, **m})

        case_summary = {
            "case": case_id,
            "mean_DSC": finite_mean([per_label[str(l)]["DSC"] for l in labels]),
            "mean_HD": finite_mean([per_label[str(l)]["HD"] for l in labels]),
            "mean_HD95": finite_mean([per_label[str(l)]["HD95"] for l in labels]),
            "mean_ASSD": finite_mean([per_label[str(l)]["ASSD"] for l in labels]),
            "mean_absRVE": finite_mean([per_label[str(l)]["absRVE"] for l in labels]),
            "labels": per_label,
        }
        case_summaries.append(case_summary)

    summary = {
        "pred_dir": str(args.pred_dir),
        "gt_dir": str(args.gt_dir),
        "labels": labels,
        "n_cases": len(case_summaries),
        "mean": {
            "DSC": finite_mean([r["DSC"] for r in rows]),
            "HD": finite_mean([r["HD"] for r in rows]),
            "HD95": finite_mean([r["HD95"] for r in rows]),
            "ASSD": finite_mean([r["ASSD"] for r in rows]),
            "RVE": finite_mean([r["RVE"] for r in rows]),
            "absRVE": finite_mean([r["absRVE"] for r in rows]),
        },
        "per_case": case_summaries,
    }

    print(json.dumps(summary["mean"], indent=2))
    print(f"Cases: {summary['n_cases']}  labels: {labels}")

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(summary, indent=2))
        print(f"Saved JSON: {args.out_json}")

    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["case", "label", "DSC", "HD", "HD95", "ASSD", "RVE", "absRVE", "n_pred", "n_ref"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved CSV: {args.out_csv}")


if __name__ == "__main__":
    main()
