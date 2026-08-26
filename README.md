<div align="center">

# AURA — Asymmetric Paired-Annotation Learning for ULF Pediatric Brain MRI

**Official code release for the LISA 2026 Challenge (Task 2) report**

*LISA 2026 Challenge Report: Asymmetric Paired-Annotation Learning for
Multi-Structure ULF Pediatric Brain MRI Segmentation*

[![nnU-Net v2](https://img.shields.io/badge/backbone-nnU--Net%20v2-1f6feb)](https://github.com/MIC-DKFZ/nnUNet)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776ab)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)

</div>

---

## Overview

Portable ultra-low-field (ULF) MRI at 0.064 T makes pediatric neuroimaging
deployable outside the radiology suite, but it segments poorly: low SNR, weak
tissue contrast, strong partial-volume effects, and motion sensitivity leave
small deep structures only partially expressed in the image.

LISA 2026 Task 2 releases **two non-equivalent annotations** per training case:

| Annotation | Origin | Property |
| --- | --- | --- |
| **HF** (`_seg.nii.gz`) | High-field anatomy registered to the ULF volume | Defines the **scored** target, but can be locally displaced by registration error |
| **LF** (`_LF_seg.nii.gz`) | Edited directly on the ULF acquisition | Aligned with **visible** ULF anatomy, but may omit or simplify unresolvable structure |

These are not two samples of one ground truth. They differ in what they are
*for* (scoring), in what they are *aligned to* (image vs. atlas anatomy), and
in what they can *see*. Training on LF alone moves away from the scored target;
averaging the two assumes a uniform reliability that does not hold across
voxels, structures, or training stages.

**AURA** treats them asymmetrically. The HF mask stays the anchor. The LF mask
enters only through a bounded, detached reliability gate driven by HF/LF
disagreement, label boundaries, predictive entropy, a per-class factor, and a
training-stage ramp — capped so that LF can refine the target but never
overwrite it, and never create foreground where HF says background.

> **AURA is a training-time supervision strategy only.**
> At inference both branches see nothing but the ULF volume — no high-field
> image, no HF mask, no LF mask.

<div align="center">
<img src="paper/figures/framework_os50_aura.png" width="92%" alt="OS50 + AURA framework"/>
</div>

---

## Results

Macro-averaged over the 16-case fold-0 development split and all 11 foreground
labels, scored against the HF reference.

| Method | DSC ↑ | HD ↓ | HD95 ↓ | ASSD ↓ | RVE (→0) |
| :-- | --: | --: | --: | --: | --: |
| OS50 (HF-supervised anchor) | 0.7984 | 3.5245 | **1.8822** | 0.7873 | −0.0109 |
| AURA (asymmetric HF/LF) | 0.7950 | 3.5902 | 1.9041 | 0.7988 | **0.0036** |
| **OS50 + AURA (0.6 / 0.4)** | **0.7988** | **3.5243** | 1.8892 | **0.7855** | −0.0093 |

On the official Task 2 validation leaderboard the submitted OS50+AURA system
scored **DSC 0.82 / HD 3.41**, at the upper edge of a densely packed field.

### How to read these numbers

The ensemble gain over the HF-supervised baseline is **+0.0004 DSC**. That is
not a meaningful superiority claim, and the paper does not make one. It is
evidence that LF-edited annotations carry *complementary* image-aligned
information when admitted as a bounded auxiliary signal. Two further caveats
are load-bearing:

- The **0.6 / 0.4 fusion weight was selected on the same 16 cases** these
  numbers are reported on, so the ensemble row is validation-tuned.
- The reliability gate is evaluated **as a whole**. No component-wise ablation
  of the disagreement, boundary, entropy, class, or ramp factors was completed
  within the challenge compute budget, so no individual factor is shown to be
  necessary or sufficient.

Full per-case and per-label metrics live in [`results/`](results/); the
narrative version is in [`docs/RESULTS.md`](docs/RESULTS.md).

---

## The method in one screen

Both branches are stock 3D full-resolution nnU-Net v2 with deep supervision.
Nothing about the network changes — the entire method is in the loss.

**OS50** — the anchor. Soft Dice + TopK10 cross-entropy on the HF target,
label smoothing 0.05, foreground oversampling raised 0.33 → 0.50, initialized
from a TotalSegmentator MRI checkpoint.

**AURA** — initialized from the best OS50 checkpoint, fine-tuned on paired
HF/LF labels. Per voxel `v`, with everything below detached from the gradient:

```
d_v = 1[y_HF ≠ y_LF]                     label disagreement
b_v = 1[v ∈ ∂y_HF ∪ ∂y_LF]               3×3×3 label boundary
u_v = normalized predictive entropy      model uncertainty
r(t) = clip((t − 25)/100, 0, 1)          25-epoch HF-only warm-up, then ramp

g_v = clip[ 0.35 · r(t) · w_class
            · (0.12 + 0.88 d_v)
            · (0.35 + 0.65 b_v)
            · (0.35 + 0.65 u_v),  0,  0.42 ]

q_v = (1 − g_v) · q_HF + g_v · q_LF      the supervised soft target
```

The gate is **multiplicative**: agreement, distance from a boundary, or model
confidence each independently suppress the LF term. `w_class` is zero on
background — LF can never invent foreground — and is tuned per structure (1.20
hippocampi, 0.95 caudate/lentiform, 0.90 thalami, 0.85 ventricles, 0.80 corpus
callosum). Left–right mirroring is disabled so paired bilateral labels are
never exchanged.

**Fusion** — `0.6 · p(OS50) + 0.4 · p(AURA)`, argmax, then the largest
26-connected component per label. No volume calibration or class thresholding.

Derivation, hyperparameters, and design rationale: [`docs/METHOD.md`](docs/METHOD.md).

---

## Repository layout

```
├── nnunet_extensions/     Files overlaid onto an nnU-Net v2 install
│   └── nnunetv2/
│       ├── training/loss/aura_loss.py                       the reliability gate
│       ├── training/nnUNetTrainer/variants/loss/
│       │   ├── nnUNetTrainerAURA_v0.py                      AURA branch
│       │   └── nnUNetTrainerDiceTopK10_OS50_LS005.py        OS50 anchor
│       └── run/load_pretrained_weights.py                   tolerant weight loading
├── scripts/
│   ├── install_extensions.py       overlay the above onto nnunetv2
│   ├── prepare_lisa_datasets.py    build Dataset001 / Dataset002 raw
│   ├── append_lf_labels.py         stack LF as seg channel 1 of Dataset002
│   ├── swap_d2_val_to_hf.py        make Dataset002 online validation HF-only
│   ├── ensemble_predictions.py     weighted fusion + argmax + LCC + zip
│   ├── sweep_ensemble_weight.py    reproduce the 0.6 / 0.4 selection
│   ├── eval_challenge_metrics.py   DSC / HD / HD95 / ASSD / RVE
│   ├── postprocess_predictions.py  standalone LCC + symmetry utilities
│   └── figures/                    regenerate the three paper figures
├── docker/                Challenge submission container (Synapse)
├── docs/                  METHOD.md · REPRODUCE.md · RESULTS.md
├── paper/                 LaTeX source and figures
└── results/               metrics, fold split, training logs and plans
```

---

## Quick start

```bash
git clone https://github.com/minhdang050806/LISA-2026-Challenge-Report-A-nnU-Net-based-asymmetric-supervision-strategy.git
cd LISA-2026-Challenge-Report-A-nnU-Net-based-asymmetric-supervision-strategy

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # includes nnunetv2 (developed against 2.7.0)

# nnU-Net discovers trainers inside its own package, so the AURA loss and
# trainers are overlaid onto the installed nnunetv2 rather than imported
# alongside it.
python scripts/install_extensions.py
python scripts/install_extensions.py --check
```

Working from an nnU-Net checkout instead of a wheel is also supported:

```bash
git clone https://github.com/MIC-DKFZ/nnUNet.git && pip install -e nnUNet
python scripts/install_extensions.py --nnunet-dir nnUNet
```

Then set the three nnU-Net roots and follow
[`docs/REPRODUCE.md`](docs/REPRODUCE.md), which walks the full path from the
released NIfTI volumes to the reported table:

```bash
export nnUNet_raw=/path/to/nnUNet_raw
export nnUNet_preprocessed=/path/to/nnUNet_preprocessed
export nnUNet_results=/path/to/nnUNet_results
```

### Inference only

The [`docker/`](docker/) container is the exact challenge submission: it reads
ULF volumes from `/input`, runs both branches, fuses at 0.6 / 0.4, applies LCC,
and writes `<case>_seg_prediction.nii.gz` to `/output`. Drop the two
`checkpoint_best.pth` files into `docker/models/` (see
[`docker/README.md`](docker/README.md)) and:

```bash
cd docker && ./build.sh && ./test.sh <image> /path/to/input
```

---

## Model weights

The two `checkpoint_best.pth` files are hosted in the
[AURA model repository on Hugging Face](https://huggingface.co/hieuphamha/A-nnU-Net-based-asymmetric-supervision-strategy).
Download the complete inference bundle with:

```bash
hf download hieuphamha/A-nnU-Net-based-asymmetric-supervision-strategy \
  --local-dir aura-models
```

The repository preserves the nnU-Net result-folder layout expected by the
inference container. Copy its `models/` directory into `docker/models/`. The
SHA-256 digests are recorded in [`docker/SHA256SUMS`](docker/SHA256SUMS) so the
downloaded files can be verified against the packaged submission.

| Branch | Dataset | Trainer | Fold | Best epoch |
| --- | --- | --- | --- | --- |
| OS50 | `Dataset001_LISA` | `nnUNetTrainerDiceTopK10_OS50_LS005` | 0 | 896 |
| AURA | `Dataset002_LISA_VCF` | `nnUNetTrainerAURA_v0` | 0 | 458 |

> **Checkpoint provenance.** The AURA snapshot that produced the paper's
> standalone DSC of 0.7950 was evaluated from an intermediate
> `checkpoint_best.pth` that was overwritten rather than archived. The
> surviving epoch-458 checkpoint — the one packaged in the container — scores
> DSC **0.7961** under the same cross-evaluation
> ([`results/metrics/aura_packaged_checkpoint_all11.json`](results/metrics/aura_packaged_checkpoint_all11.json)).
> Trainer, fold, architecture, fusion weights, mirroring metadata and LCC
> procedure are identical. This distinction is deliberate and should be carried
> forward rather than quietly reconciled.

---

## Data

LISA 2026 Task 2 data is distributed by the challenge organizers and is **not**
redistributed here. Obtain it from the official challenge, then point
`scripts/prepare_lisa_datasets.py` at the directory of `LISA_XXXX_ciso.nii.gz`,
`LISA_XXXX_seg.nii.gz` and `LISA_XXXX_LF_seg.nii.gz` files.

The exact 5-fold split used throughout — 63 training / 16 development cases in
fold 0 — is committed at
[`results/splits/splits_final.json`](results/splits/splits_final.json). Copy it
into both preprocessed dataset folders before training to reproduce the
reported numbers.

---

## Citation

```bibtex
@inproceedings{pham2026aura,
  title     = {{LISA} 2026 Challenge Report: Asymmetric Paired-Annotation Learning
               for Multi-Structure {ULF} Pediatric Brain {MRI} Segmentation},
  author    = {Pham, Ha-Hieu and Cao, Dang P. M. and Pham, Minh Hoang and
               Vo Ngoc, Khanh Nguyen and Nguyen, Thanh-Huy and
               Bagci, Ulas and Pham, Huy-Hieu},
  booktitle = {MICCAI Challenge on Low-field pediatric brain magnetic resonance
               Image Segmentation and quality Assurance (LISA)},
  year      = {2026}
}
```

Please also cite nnU-Net, on which both branches are built:

```bibtex
@article{isensee2021nnunet,
  title   = {nnU-Net: a self-configuring method for deep learning-based
             biomedical image segmentation},
  author  = {Isensee, Fabian and Jaeger, Paul F. and Kohl, Simon A. A. and
             Petersen, Jens and Maier-Hein, Klaus H.},
  journal = {Nature Methods}, volume = {18}, number = {2}, pages = {203--211},
  year    = {2021}, doi = {10.1038/s41592-020-01008-z}
}
```

---

## Acknowledgments

Built on [nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet) (Apache 2.0) by
MIC-DKFZ, with encoder initialization from a
[TotalSegmentator MRI](https://github.com/wasserth/TotalSegmentator) checkpoint.
We thank the LISA 2026 organizers for the challenge data and evaluation
infrastructure.

## License

Apache License 2.0 — see [LICENSE](LICENSE). The `nnunet_extensions/` tree
derives from nnU-Net and remains under nnU-Net's Apache 2.0 license. Challenge
data is governed by the LISA 2026 data use agreement and is not covered here.
