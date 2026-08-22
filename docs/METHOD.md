# Method

Reference for the two branches, the reliability gate, and every hyperparameter
that differs from stock nnU-Net. Section numbers follow the paper.

---

## 1. The supervision problem

Each LISA 2026 Task 2 training case ships two annotations of the same 11
structures over the same ULF volume:

- **HF** — high-field anatomy linearly registered onto the ULF scan. This is
  the annotation the challenge scores against, and it inherits whatever local
  displacement the registration introduced.
- **LF** — edited directly on the 0.064 T acquisition. Aligned with what is
  actually visible, but structures that cannot be resolved at low field may be
  omitted or simplified.

Disagreement between them is **structured, not i.i.d.** Measured over the
training set, macro HF/LF Dice ranges from 0.66 on the worst case to well above
0.86 at the median, and the disagreement concentrates along anatomical
boundaries — occasionally covering most of a structure on a given slice. See
`paper/figures/dataset_hf_lf_disagreement.png`.

Two obvious treatments both discard information:

| Treatment | Failure |
| --- | --- |
| Train on LF only | Optimizes something other than the scored target |
| Average HF and LF uniformly | Assumes equal reliability across voxels, structures, and training stages |

AURA instead keeps HF as the anchor and admits LF as a *conditional* auxiliary
observation.

---

## 2. OS50 — the HF-supervised anchor

`nnUNetTrainerDiceTopK10_OS50_LS005`, trained on `Dataset001_LISA`
(3d_fullres, fold 0), initialized from a TotalSegmentator MRI checkpoint.

Three deviations from the default nnU-Net trainer, each aimed at small
structures and uncertain boundaries:

| Change | Value | Why |
| --- | --- | --- |
| Loss | soft Dice + TopK10 cross-entropy | Concentrates the CE term on the hardest 10 % of valid voxels instead of letting confident background dominate |
| Label smoothing | `0.05` | Reduces overconfident fitting to uncertain boundary annotations |
| Foreground oversampling | `0.33 → 0.50` | More training patches actually contain annotated structure |

Dice excludes background and uses smoothing constant `1e-5`. Deep supervision
uses the standard nnU-Net weights with the lowest-resolution output zeroed.

---

## 3. AURA — asymmetric paired-annotation supervision

`nnUNetTrainerAURA_v0`, trained on `Dataset002_LISA_VCF` whose preprocessed
segmentation carries **two channels**: channel 0 = HF, channel 1 = LF. It is
initialized from the best OS50 checkpoint via `-pretrained_weights`.

The network is unchanged. All of the method is inside the loss
(`nnunetv2/training/loss/aura_loss.py`), which builds its own supervision
target from the two label channels.

### 3.1 Reliability factors

Let `p_{v,c}` be the predicted probability of class `c` at voxel `v`, over
`C = 12` classes including background. Three voxel-wise factors, all detached
from the gradient:

```
d_v = 1[ y_HF(v) ≠ y_LF(v) ]                          label disagreement
b_v = 1[ v ∈ ∂y_HF ∪ ∂y_LF ]                          3×3×3 label boundary
u_v = stopgrad( −(1/log C) · Σ_c p_{v,c} log p_{v,c} ) normalized entropy ∈ [0,1]
```

`∂y` is computed as a `3×3×3` max-pool minus min-pool on the label map — a
voxel is on a boundary when its neighbourhood is not label-constant.

A training-stage ramp delays LF influence entirely until after a 25-epoch
HF-only warm-up, then phases it in over 100 epochs:

```
r(t) = clip( (t − 25) / 100, 0, 1 )
```

### 3.2 The gate

```
g_v = stopgrad( clip[ 0.35 · r(t) · w_{y_HF(v)}
                      · (0.12 + 0.88 · d_v)
                      · (0.35 + 0.65 · b_v)
                      · (0.35 + 0.65 · u_v),
                      0,  0.42 ] )
```

The gate is **multiplicative by design**. Any one of "HF and LF agree", "far
from a boundary", or "the model is already confident" independently suppresses
the LF term, even after warm-up. Only when all three signals fire does LF reach
its cap. That cap — `0.42 = 0.35 × 1.20` — is what prevents LF from dominating
the scored HF target.

Because `g_v` and `u_v` are detached, the network **cannot lower its loss by
manipulating the reliability signal**: it has no gradient path to the gate.

`w_class` is indexed by the **HF** label, and background is zero — so LF can
never create foreground where the HF anchor says background:

| Class | ID | `w` | Rationale |
| --- | --- | --- | --- |
| Background | 0 | 0.00 | LF cannot invent foreground |
| Hippocampus L/R | 1, 2 | 1.20 | Weakest structures; most to gain from image-aligned edits |
| Ventricle L/R | 3, 4 | 0.85 | LF systematically over-segments these |
| Caudate L/R | 5, 6 | 0.95 | — |
| Lentiform L/R | 7, 8 | 0.95 | — |
| Thalamus L/R | 9, 10 | 0.90 | Already near-saturated under HF supervision |
| Corpus callosum | 11 | 0.80 | Thin, partial-volume dominated; LF edits least trustworthy |

### 3.3 Supervised target and objective

The gate interpolates between the smoothed HF one-hot target (`ε = 0.03`) and
the LF one-hot target, renormalized over classes:

```
q_v = (1 − g_v) · q_HF(v) + g_v · q_LF(v)
```

The objective combines class-weighted soft Dice, soft TopK10 cross-entropy,
and a boundary-restricted soft Dice term:

```
L_AURA = L_Dice(p, q) + L_TopK10(p, q) + 0.03 · L_boundary(p, q)
```

Dice class weights are normalized to mean 1 after setting hippocampi to 1.25
and corpus callosum to 1.15. `L_boundary` is the same soft Dice restricted to
the union label boundary.

### 3.4 Augmentation

**Left–right mirroring is disabled** (mirror axes `(0,1,2) → (0,1)`) at both
training and inference, because flipping would exchange paired bilateral labels
against a target built from two annotations of the same anatomy. This is
recorded in the checkpoint's `inference_allowed_mirroring_axes`, so
`nnUNetv2_predict` honours it automatically.

Note that this costs some test-time-augmentation coverage relative to OS50,
which keeps all three mirror axes. Part of AURA's standalone deficit against
OS50 is attributable to that, not to the paired-label objective.

### 3.5 Hyperparameter summary

| Symbol | Trainer attribute | Value |
| --- | --- | --- |
| max LF weight | `AURA_LF_WEIGHT` | 0.35 |
| gate cap | `max_lf_weight × 1.20` | 0.42 |
| HF label smoothing | `AURA_LABEL_SMOOTHING` | 0.03 |
| TopK percentage | `AURA_TOPK` | 10 |
| boundary Dice weight | `AURA_BOUNDARY_WEIGHT` | 0.03 |
| warm-up epochs | `AURA_WARMUP_EPOCHS` | 25 |
| ramp epochs | `AURA_RAMP_EPOCHS` | 100 |
| disabled mirror axis | `DISABLE_LR_MIRROR_AXIS` | 2 |
| foreground oversampling | `oversample_foreground_percent` | 0.50 |

---

## 4. Validation signal on Dataset002

nnU-Net's online validation Dice scatters **all** target channels into the
one-hot tensor. On a two-channel (HF, LF) target that makes the tracked metric
a mixture of both annotations — which is not the challenge ranking metric, and
therefore selects the wrong `checkpoint_best.pth`.

`scripts/swap_d2_val_to_hf.py` replaces channel 1 with channel 0 **for the 16
fold-0 validation cases only**. Training cases keep their true (HF, LF) pair, so
the AURA loss still sees both observations, while EMA pseudo-Dice and best-
checkpoint selection now track the scored HF target. The original two-channel
files are moved aside and restored by `--restore`.

---

## 5. Fusion and post-processing

At inference each branch emits a 12-channel probability tensor at the original
image geometry:

```
p_ens = 0.6 · p_OS50 + 0.4 · p_AURA
ŷ     = argmax_c p_ens
```

followed by per-class largest 26-connected-component filtering over labels
1–11. Nothing else: no test-time volume calibration, no class-specific
thresholding, no probability rescaling.

The 0.6 / 0.4 weight was chosen by screening `w ∈ {0.0, 0.1, …, 1.0}` on the
development split (`scripts/sweep_ensemble_weight.py`). DSC is flat across the
top of that sweep — 0.7988 at both `w = 0.6` and `w = 0.7` — and the weight was
selected on the same 16 cases the paper reports, so the ensemble row is
validation-tuned by construction.

---

## 6. Deliberate non-features

Stated explicitly because they bound what the results can support:

- AURA adds **no** new network, no second decoder, no extra head. The only
  change is the training target.
- Neither annotation, nor a high-field image, is used at inference.
- A label-wise ensemble reached DSC 0.7994 but was **excluded** from the paper
  to avoid compounding validation-set selection bias.
- The gate is evaluated as a whole. No component-wise ablation of the five
  factors was completed within the challenge compute budget.
