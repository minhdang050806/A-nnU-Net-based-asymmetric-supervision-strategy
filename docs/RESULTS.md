# Results

All numbers are macro-averaged over the **16-case fold-0 development split**
and scored against the **HF reference labels** (`Dataset001_LISA/labelsTr`).
Metrics are computed per case and per foreground label, then averaged over all
(case, label) pairs; non-finite distance values are dropped from the mean.

The exact split is committed at
[`results/splits/splits_final.json`](../results/splits/splits_final.json):

```
LISA_0003  LISA_0005  LISA_0007  LISA_0014  LISA_0021  LISA_0029
LISA_0031  LISA_0038  LISA_0041  LISA_0043  LISA_0045  LISA_0052
LISA_0054  LISA_0063  LISA_1009  LISA_1010
```

---

## Main table — all 11 foreground labels

| Method | DSC ↑ | HD ↓ | HD95 ↓ | ASSD ↓ | RVE (→0) | Source |
| :-- | --: | --: | --: | --: | --: | :-- |
| OS50 | 0.7984 | 3.5245 | **1.8822** | 0.7873 | −0.0109 | [`os50_all11.json`](../results/metrics/os50_all11.json) |
| AURA | 0.7950 | 3.5902 | 1.9041 | 0.7988 | **0.0036** | [`aura_all11.json`](../results/metrics/aura_all11.json) |
| **OS50 + AURA (0.6/0.4)** | **0.7988** | **3.5243** | 1.8892 | **0.7855** | −0.0093 | [`os50_aura_w06_all11.json`](../results/metrics/os50_aura_w06_all11.json) |

Distance metrics are in millimetres at 1 mm isotropic resolution. For signed
RVE the value closest to zero is bolded.

## Hippocampi only (labels 1, 2)

The challenge-focused subset, and the hardest structures in the task.

| Method | DSC ↑ | HD ↓ | HD95 ↓ | ASSD ↓ | RVE | absRVE |
| :-- | --: | --: | --: | --: | --: | --: |
| OS50 | 0.6655 | 3.9288 | **2.5405** | **1.0323** | −0.0450 | **0.2095** |
| AURA | 0.6582 | 3.9408 | 2.5700 | 1.0600 | **0.0356** | 0.2223 |
| OS50 + AURA | **0.6655** | **3.9180** | 2.5729 | 1.0325 | −0.0307 | 0.2119 |

## Per-class DSC

| ID | Structure | OS50 | AURA | OS50 + AURA | Δ vs OS50 |
| --: | :-- | --: | --: | --: | --: |
| 1 | Hippocampus L | 0.6440 | 0.6347 | 0.6436 | −0.0004 |
| 2 | Hippocampus R | 0.6870 | 0.6818 | 0.6874 | +0.0004 |
| 3 | Ventricle L | 0.8037 | 0.7978 | 0.8036 | −0.0001 |
| 4 | Ventricle R | 0.7805 | 0.7825 | 0.7833 | +0.0028 |
| 5 | Caudate L | 0.8250 | 0.8143 | 0.8227 | −0.0023 |
| 6 | Caudate R | 0.8227 | 0.8258 | 0.8253 | +0.0026 |
| 7 | Lentiform L | 0.8321 | 0.8285 | 0.8323 | +0.0002 |
| 8 | Lentiform R | 0.8417 | 0.8447 | 0.8437 | +0.0020 |
| 9 | Thalamus L | 0.8965 | 0.8940 | 0.8960 | −0.0005 |
| 10 | Thalamus R | 0.9047 | 0.9041 | 0.9056 | +0.0009 |
| 11 | Corpus callosum | 0.7445 | 0.7372 | 0.7435 | −0.0010 |

The mean hides a wide anatomical range. Thalami are effectively saturated above
0.89, while the hippocampi — the smallest structures, around 1.2k voxels, where
label noise dominates — sit near 0.64–0.69, and the thin corpus callosum near
0.74. The ensemble wins on 6 of 11 labels and loses small amounts on the rest,
which is exactly why the global gain is +0.0004 rather than something larger.

## Fusion weight sweep

`w · p(OS50) + (1 − w) · p(AURA)`, each point evaluated through the full
pipeline (fuse → argmax → per-class LCC).
Full data: [`ensemble_weight_sweep.json`](../results/metrics/ensemble_weight_sweep.json).

| w(OS50) | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | **0.6** | 0.7 | 0.8 | 0.9 | 1.0 |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| DSC | .7950 | .7957 | .7964 | .7968 | .7975 | .7983 | **.7988** | .7988 | .7986 | .7985 | .7984 |
| Hippocampi | .6582 | .6595 | .6605 | .6609 | .6620 | .6644 | .6655 | .6658 | .6656 | .6658 | .6655 |

Per-class largest-connected-component filtering is a **no-op on this split**:
every prediction of every method already has exactly one component per label
(0 of 176 case-label pairs are fragmented). The single-model rows and the
LCC-filtered ensemble row are therefore directly comparable, and the sweep
endpoints reproduce the standalone rows exactly.

The curve is flat across its top — 0.7988 at both `w = 0.6` and `w = 0.7` — and
every point between 0.5 and 1.0 lies within 0.0005 DSC of the maximum. The
paper uses 0.6.

## Reference points

Other configurations evaluated on the same split, for context.

| Configuration | DSC | Note |
| :-- | --: | :-- |
| Dice+TopK10 baseline (matched) | 0.7971 | The recipe OS50 builds on |
| OS50, nnU-Net internal validation | 0.7975 | Predicted from preprocessed data during training, not from raw NIfTI |
| LSTR-v0, final checkpoint | 0.7974 | A different paired-supervision variant from the same challenge cycle |
| Label-wise ensemble | 0.7994 | **Excluded** from the paper — picking the better model per label compounds validation-set selection bias |
| AURA, packaged epoch-458 checkpoint | 0.7961 | See checkpoint provenance below |

## Challenge validation leaderboard

The submitted OS50+AURA system scored **DSC 0.82 / HD 3.41** on the official
LISA 2026 Task 2 validation leaderboard, at the upper edge of the DSC
distribution and the lower edge of the HD distribution. Of 191 submissions
accepted by Synapse, 182 remained valid after excluding entries with `inf` or
`NaN` HD. The distribution is plotted in
[`paper/LISA2026_Task2_validation_distribution.pdf`](../paper/LISA2026_Task2_validation_distribution.pdf).

The better-performing submissions occupy a narrow band of roughly 0.82–0.83
DSC, so leaderboard position separates methods weakly in this operating region.
The distribution is also over *submissions*, not distinct teams or distinct
methods, so it characterizes leaderboard density rather than methodological
diversity.

---

## How to read all of this

Three caveats are load-bearing, and none of them is boilerplate.

**The ensemble gain is +0.0004 DSC.** That is below nnU-Net's own seed-to-seed
variance on 16 cases. It does not support a claim of practical or clinical
superiority over the HF-supervised baseline, and the paper does not make one.
The defensible reading is narrower: LF-edited annotations carry complementary
image-aligned information when admitted as a bounded auxiliary signal rather
than as a second ground truth. Note also that AURA *alone* is weaker than OS50
on the global average — the fusion gain comes from error complementarity, not
from AURA being the better model.

**The ensemble row is validation-tuned.** The 0.6/0.4 weight, the checkpoint
choice, and the decision to exclude the label-wise ensemble were all made on
the same 16 cases these numbers report. This is a challenge-cycle evaluation,
not an unbiased estimate of generalization.

**The gate is evaluated as a whole.** No component-wise ablation of the
disagreement, boundary, entropy, class-weight, or ramp factors was completed
within the challenge compute budget. Nothing here shows that any individual
factor is necessary or sufficient.

Since the development split is drawn from the same source distribution as the
training data, cross-cohort robustness remains uncharacterized. Evaluation on
the hidden test set and on independent external ULF pediatric cohorts is what
would settle whether any of this transfers.

## Checkpoint provenance

The AURA snapshot that produced the standalone DSC of **0.7950** reported above
was evaluated from an intermediate `checkpoint_best.pth` that was subsequently
overwritten rather than archived. The surviving epoch-458 checkpoint — the one
packaged in the submission container — scores **0.7961** under the same
Dataset002→Dataset001 cross-evaluation
([`aura_packaged_checkpoint_all11.json`](../results/metrics/aura_packaged_checkpoint_all11.json)).

Trainer, fold, architecture, fusion weights, mirroring metadata and LCC
procedure are identical between the two. The discrepancy is recorded rather
than reconciled, and it is the reason a fresh training run should be expected
to land near, but not exactly on, either number.

## Training provenance

[`results/training_logs/`](../results/training_logs/) carries the nnU-Net
`plans.json`, `dataset.json`, `debug.json`, full training log and progress plot
for both branches.

| | OS50 | AURA |
| :-- | :-- | :-- |
| Dataset | `Dataset001_LISA` | `Dataset002_LISA_VCF` |
| Trainer | `nnUNetTrainerDiceTopK10_OS50_LS005` | `nnUNetTrainerAURA_v0` |
| Initialization | TotalSegmentator MRI (TS850) | best OS50 checkpoint |
| Best epoch | 896 | 458 |
| Best EMA pseudo-Dice | 0.8027 | 0.8044 |
| Inference mirror axes | (0, 1, 2) | (0, 1) |
| Patch size / batch | 112×160×128 / 2 | 112×160×128 / 2 |
| Optimizer | SGD, LR 0.01 poly, WD 3e-5 | SGD, LR 0.01 poly, WD 3e-5 |

AURA's `num_epochs` was left at nnU-Net's default of 1000; training was stopped
after roughly 500 epochs, past the point where the best checkpoint was reached.
