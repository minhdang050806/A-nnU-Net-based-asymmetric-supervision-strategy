# Reproducing the reported results

End-to-end, from the released LISA 2026 Task 2 volumes to Table 1 of the paper.

Throughout, `$LISA_DATA` is the directory of released NIfTI files
(`LISA_XXXX_ciso.nii.gz`, `LISA_XXXX_seg.nii.gz`, `LISA_XXXX_LF_seg.nii.gz`),
and `$WORK` is a scratch directory for predictions.

> **Compute.** Each branch was trained on a single 24 GB GPU. OS50 ran ~1000
> epochs and AURA ~500, at roughly 240 s per epoch — on the order of several
> GPU-days for the pair. Skip to [step 6](#6-inference) if you only want
> inference from released checkpoints.

---

## 0. Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/install_extensions.py
python scripts/install_extensions.py --check     # all four files should read "ok"

export nnUNet_raw=/path/to/nnUNet_raw
export nnUNet_preprocessed=/path/to/nnUNet_preprocessed
export nnUNet_results=/path/to/nnUNet_results
export LISA_DATA=/path/to/LISA/Dataset
export WORK=/path/to/scratch
```

Verify the extensions are importable through nnU-Net's own package:

```bash
python -c "from nnunetv2.training.loss.aura_loss import AURAAsymmetricLoss; print('ok')"
```

---

## 1. Build the raw datasets

Both branches consume the same images and the same HF labels in raw form. They
diverge only at preprocessing, where LF is stacked onto Dataset002.

```bash
python scripts/prepare_lisa_datasets.py --data-dir "$LISA_DATA"
```

This writes `Dataset001_LISA` (OS50) and `Dataset002_LISA_VCF` (AURA) under
`$nnUNet_raw`, each with 79 cases, symlinked by default. Pass `--copy` if
symlinks are impractical.

## 2. Plan, preprocess, and pin the split

```bash
nnUNetv2_plan_and_preprocess -d 1 2 --verify_dataset_integrity
```

The reported numbers use one specific 5-fold split — 63 training / 16
development cases in fold 0. Install it in both datasets **before** training,
otherwise nnU-Net will generate its own and the development set will differ:

```bash
cp results/splits/splits_final.json "$nnUNet_preprocessed/Dataset001_LISA/"
cp results/splits/splits_final.json "$nnUNet_preprocessed/Dataset002_LISA_VCF/"
```

## 3. Stack the LF annotation onto Dataset002

Re-runs the nnU-Net preprocessor with the LF label in place of HF — same image,
so cropping and resampling are identical — and concatenates the result as
channel 1 of each `*_seg.b2nd`:

```bash
python scripts/append_lf_labels.py \
    --dataset_id 2 \
    --configuration 3d_fullres \
    --lf_label_dir "$LISA_DATA" \
    --lf_suffix _LF_seg.nii.gz
```

Each case should report `stacked HF+LF seg -> (2, Z, Y, X)`. The script is
idempotent: already-stacked cases are skipped.

Then make Dataset002's *online* validation metric HF-only, so EMA pseudo-Dice
and `checkpoint_best.pth` selection track the scored target rather than a
mixture of both annotations (see [METHOD.md §4](METHOD.md#4-validation-signal-on-dataset002)):

```bash
python scripts/swap_d2_val_to_hf.py           # reversible with --restore
```

This touches the 16 fold-0 validation cases only. Training cases keep their
true (HF, LF) pair, which the AURA loss still needs.

## 4. Train the OS50 anchor

Initialized from a TotalSegmentator MRI checkpoint — the `part1_organs`
`3d_fullres` fold-0 checkpoint, which is compatible with the standard
single-channel architecture:

```bash
export TS_CKPT=/path/to/Dataset850_TotalSegMRI_part1_organs_1088subj/\
nnUNetTrainer_2000epochs_NoMirroring__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth

nnUNetv2_train 1 3d_fullres 0 \
    -tr nnUNetTrainerDiceTopK10_OS50_LS005 \
    -pretrained_weights "$TS_CKPT"
```

SGD, initial LR 0.01 with polynomial decay, weight decay 3e-5, batch size 2,
patch size 112×160×128, 1000 epochs. The reported checkpoint is epoch 896
(`checkpoint_best.pth`, best EMA 0.8027).

Training without `-pretrained_weights` still works and lands close, but is not
what the paper reports.

## 5. Train the AURA branch

Initialized from the **best OS50 checkpoint**, not from TS850:

```bash
nnUNetv2_train 2 3d_fullres 0 \
    -tr nnUNetTrainerAURA_v0 \
    -pretrained_weights "$nnUNet_results/Dataset001_LISA/\
nnUNetTrainerDiceTopK10_OS50_LS005__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth"
```

Expected in the training log:

```
[AURA-v0] Disabling L-R mirror augmentation: mirror_axes (0, 1, 2) -> (0, 1)
[AURA-v0] Warmup ended at epoch 25; activating asymmetric LF reliability gate.
```

`num_epochs` is left at nnU-Net's default of 1000, but training was stopped
after ~500 epochs; the reported checkpoint is **epoch 458** (best EMA 0.8044).
The disabled left–right mirror axis is written into the checkpoint's
`inference_allowed_mirroring_axes`, so `nnUNetv2_predict` honours it without
extra flags.

## 6. Inference

Stage the 16 fold-0 development cases as an nnU-Net input folder:

```bash
mkdir -p "$WORK/dev_images"
python - <<'PY'
import json, os, shutil
from pathlib import Path
val = json.loads(Path("results/splits/splits_final.json").read_text())[0]["val"]
data, out = Path(os.environ["LISA_DATA"]), Path(os.environ["WORK"]) / "dev_images"
for case in val:
    shutil.copy2(data / f"{case}_ciso.nii.gz", out / f"{case}_0000.nii.gz")
print(f"staged {len(val)} cases")
PY
```

Predict with both branches, keeping probability maps for fusion:

```bash
nnUNetv2_predict -d 1 -c 3d_fullres -f 0 -chk checkpoint_best.pth \
    -tr nnUNetTrainerDiceTopK10_OS50_LS005 --save_probabilities \
    -i "$WORK/dev_images" -o "$WORK/probs_os50"

nnUNetv2_predict -d 2 -c 3d_fullres -f 0 -chk checkpoint_best.pth \
    -tr nnUNetTrainerAURA_v0 --save_probabilities \
    -i "$WORK/dev_images" -o "$WORK/probs_aura"
```

## 7. Fusion weight selection

Screens `w ∈ {0.0, 0.1, …, 1.0}` under the full pipeline (fuse → argmax → LCC):

```bash
python scripts/sweep_ensemble_weight.py \
    --os50-probs "$WORK/probs_os50" \
    --aura-probs "$WORK/probs_aura" \
    --out "$WORK/ensemble_weight_sweep.json"
```

DSC peaks flat at 0.7988 for both `w = 0.6` and `w = 0.7`; the paper uses 0.6.
Compare against the committed
[`results/metrics/ensemble_weight_sweep.json`](../results/metrics/ensemble_weight_sweep.json).

**This selection uses the same 16 cases the results are reported on.** The
ensemble row is validation-tuned by construction and is not an unbiased
estimate of generalization.

## 8. Build the final predictions

```bash
python scripts/ensemble_predictions.py \
    --prob-dirs "$WORK/probs_os50" "$WORK/probs_aura" \
    --weights 0.6,0.4 \
    --lcc-labels 1,2,3,4,5,6,7,8,9,10,11 \
    --images-dir "$LISA_DATA" \
    --out-dir "$WORK/os50_aura_w06"
```

Writes `<case>_seg_prediction.nii.gz` per case, a flat
`LISA_SEG_predictions.zip` for challenge upload, and a `qc_report.json`
recording weights, per-case label histograms, and NIfTI geometry.

Leave `--fg-scale`, `--class-scales` and `--target-volume-ratios` at their
defaults: those knobs were explored during the challenge cycle but are not part
of the reported system.

## 9. Evaluate

```bash
python scripts/eval_challenge_metrics.py \
    --pred_dir "$WORK/os50_aura_w06/predictions" \
    --labels 1,2,3,4,5,6,7,8,9,10,11 \
    --out_json "$WORK/os50_aura_w06_all11.json" \
    --out_csv  "$WORK/os50_aura_w06_all11.csv"
```

Reference labels default to `$nnUNet_raw/Dataset001_LISA/labelsTr`. Repeat with
`--labels 1,2` for the hippocampi-only report, and point `--pred_dir` at each
branch's own predictions for the OS50 and AURA rows.

Expected, matching [`results/metrics/`](../results/metrics/):

| Method | DSC | HD | HD95 | ASSD | RVE |
| --- | --: | --: | --: | --: | --: |
| OS50 | 0.7984 | 3.5245 | 1.8822 | 0.7873 | −0.0109 |
| AURA | 0.7950 | 3.5902 | 1.9041 | 0.7988 | 0.0036 |
| OS50 + AURA | 0.7988 | 3.5243 | 1.8892 | 0.7855 | −0.0093 |

## 10. Figures

```bash
python scripts/figures/make_dataset_figure.py --data-dir "$LISA_DATA"
python scripts/figures/make_qualitative_figure.py \
    --image-dir "$LISA_DATA" \
    --os50-dir "$WORK/probs_os50" \
    --ensemble-dir "$WORK/os50_aura_w06/predictions"
```

Both write PDF + PNG into `paper/figures/`. The displayed cases are fixed by a
pre-specified selection rule, not chosen visually — see the module docstrings.

Figure 2, the framework diagram, is hand-authored in diagrams.net rather than
generated; see [`paper/README.md`](../paper/README.md#figures) for how to edit
it and re-export.

---

## Expected deviations

- **AURA standalone DSC.** The exact snapshot behind the paper's 0.7950 was an
  intermediate `checkpoint_best.pth` that was overwritten rather than archived.
  The surviving epoch-458 checkpoint scores 0.7961 under the same
  cross-evaluation. A fresh training run will land somewhere in this
  neighbourhood rather than on either number exactly.
- **Run-to-run variance.** Differences at the third decimal of mean DSC are
  within nnU-Net's own seed noise on 16 cases. The reported OS50 → ensemble gain
  is +0.0004, which is *below* that threshold; this is stated plainly in the
  paper and should not be read as a superiority claim.
- **Library versions.** `requirements.txt` pins the versions used in the
  submission container. Newer PyTorch or nnU-Net releases may shift results
  slightly.
