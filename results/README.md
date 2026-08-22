# Results artifacts

Everything needed to check the reported numbers without rerunning training.
The narrative reading of these files is in [`docs/RESULTS.md`](../docs/RESULTS.md).

## `metrics/`

Produced by [`scripts/eval_challenge_metrics.py`](../scripts/eval_challenge_metrics.py)
on the 16-case fold-0 development split, scored against
`Dataset001_LISA/labelsTr`. Each `.json` carries the macro `mean` block plus a
`per_case` breakdown; the matching `.csv` is one row per (case, label).

| File | Paper row |
| :-- | :-- |
| `os50_all11.{json,csv}` | OS50, all 11 labels |
| `os50_hippocampi.{json,csv}` | OS50, labels 1–2 |
| `aura_all11.{json,csv}` | AURA, all 11 labels |
| `aura_hippocampi.{json,csv}` | AURA, labels 1–2 |
| `os50_aura_w06_all11.{json,csv}` | **OS50 + AURA (0.6/0.4)**, all 11 labels |
| `os50_aura_w06_hippocampi.{json,csv}` | OS50 + AURA, labels 1–2 |
| `ensemble_weight_sweep.json` | The `w ∈ {0.0 … 1.0}` screen that selected 0.6/0.4 |

Context values referenced in the paper text but not in Table 1:

| File | What it is |
| :-- | :-- |
| `reference_os50_nnunet_validation_all11.json` | OS50 scored from nnU-Net's own in-training `validation/` folder (0.7975) rather than a fresh `nnUNetv2_predict` run (0.7984) |
| `reference_lstr_v0_final_all11.json` | LSTR-v0, a different paired-supervision variant from the same challenge cycle (0.7974) |
| `reference_labelwise_ensemble_all11.json` | Label-wise ensemble (0.7994), **excluded** from the paper to avoid compounding validation-set selection bias |
| `aura_packaged_checkpoint_all11.json` | The epoch-458 AURA checkpoint packaged in the Docker submission (0.7961) — see checkpoint provenance below |

The `pred_dir` and `gt_dir` fields inside each JSON record the absolute paths
of the original run, and are kept as provenance rather than as usable paths.

## `splits/splits_final.json`

The nnU-Net 5-fold split used throughout. Fold 0 is 63 training / 16
development cases. Copy it into both preprocessed dataset folders before
training — otherwise nnU-Net generates its own and the development set will not
match.

## `training_logs/`

Per branch (`os50/`, `aura/`):

| File | Contents |
| :-- | :-- |
| `plans.json` | Full nnU-Net plan: 1 mm isotropic spacing, patch 112×160×128, batch 2, PlainConvUNet with 32/64/128/256/320/320 channels |
| `dataset.json` | Channel names and the 12-label map |
| `debug.json` | Resolved trainer state: epochs, LR, weight decay, oversampling, `inference_allowed_mirroring_axes` |
| `training_log.txt` | Complete nnU-Net training log |
| `progress.png` | Loss and pseudo-Dice curves |

`debug.json` is the authoritative record of what actually ran, including the
mirroring axes — `(0,1,2)` for OS50 and `(0,1)` for AURA, whose left–right axis
is disabled.

## Not included

**Model weights.** The two `checkpoint_best.pth` files are ~236 MB each. Their
SHA-256 digests are in [`docker/SHA256SUMS`](../docker/SHA256SUMS) so a
downloaded copy can be verified against the packaged submission.

**Predictions.** The `.nii.gz` and `.npz` outputs are regenerable from the
checkpoints; [`docs/REPRODUCE.md`](../docs/REPRODUCE.md) §6–8 covers it.

**Challenge data.** LISA 2026 Task 2 data is distributed by the organizers
under their data use agreement and is not redistributed here.

## Checkpoint provenance

`aura_all11.json` (DSC 0.7950) comes from an intermediate AURA
`checkpoint_best.pth` that was later overwritten rather than archived. The
surviving epoch-458 checkpoint — the one in the container — scores 0.7961
(`aura_packaged_checkpoint_all11.json`). Trainer, fold, architecture, fusion
weights, mirroring metadata and post-processing are identical. Both numbers are
kept here rather than reconciled.
