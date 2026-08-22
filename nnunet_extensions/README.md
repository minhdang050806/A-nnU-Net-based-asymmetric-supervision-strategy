# nnU-Net extensions

Files overlaid onto an nnU-Net v2 installation by
[`scripts/install_extensions.py`](../scripts/install_extensions.py). Paths under
`nnunetv2/` mirror the paths they take inside the installed package.

nnU-Net resolves trainers with `recursive_find_python_class` over its own
package directory, so custom trainers and losses cannot simply sit beside it on
`sys.path` — they have to be copied in. Hence the overlay rather than a plugin.

## Additive files

Nothing upstream is touched by these; they are new modules.

| File | Contents |
| :-- | :-- |
| `training/loss/aura_loss.py` | `AURAAsymmetricLoss` — the reliability gate, soft-target construction, soft TopK cross-entropy, class-weighted soft Dice, and the boundary-restricted Dice term |
| `training/nnUNetTrainer/variants/loss/nnUNetTrainerAURA_v0.py` | `nnUNetTrainerAURA_v0` — wires the loss into deep supervision, advances the epoch counter that drives the ramp, disables the left–right mirror axis, sets foreground oversampling to 0.50 |
| `training/nnUNetTrainer/variants/loss/nnUNetTrainerDiceTopK10_OS50_LS005.py` | `nnUNetTrainerDiceTopK10_OS50_LS005` — the OS50 anchor: Dice + TopK10 with label smoothing 0.05 and foreground oversampling 0.50 |

## Replaced upstream file

| File | Change |
| :-- | :-- |
| `run/load_pretrained_weights.py` | Widens the set of parameter names treated as optional when transferring pretrained weights, so auxiliary segmentation heads that are absent from a single-head checkpoint are skipped instead of raising |

The installer writes a `.orig` backup before the first overwrite, so the
installation can be reverted by restoring it.

Strictly speaking AURA does not need this patch — it uses the standard
single-head architecture and loads cleanly from the OS50 checkpoint. It is
included because it is part of the tree the reported models were trained under,
and dropping it would make the released code diverge from what actually ran.

## Interfaces worth knowing

**`AURAAsymmetricLoss.forward(logits, target)`** expects `target` to carry
**two segmentation channels**: `target[:, 0]` is HF, `target[:, 1]` is LF. This
is what `scripts/append_lf_labels.py` produces in the preprocessed
`Dataset002_LISA_VCF`. Passing a single-channel target raises.

**`AURAAsymmetricLoss.current_epoch`** must be advanced externally — the
trainer does it in `on_train_epoch_start`. It drives `r(t)`, the warm-up and
ramp schedule. Left at 0, the LF branch never activates and the loss reduces to
HF-only supervision.

**Gate detachment.** The entropy term, the gate, and the constructed soft
target are all computed under `torch.no_grad()`. The network has no gradient
path to the reliability signal and therefore cannot reduce its loss by
manipulating it. Preserve this if you modify the loss.

## Version

Developed against **nnU-Net v2.7.0**. The additive files use only stable public
interfaces (`nnUNetTrainer`, `DeepSupervisionWrapper`, `label_manager`) and
should port across minor versions with little friction. The replaced
`load_pretrained_weights.py` is version-sensitive — check it against upstream
before installing onto a substantially newer nnU-Net.

## License

nnU-Net is Apache 2.0 (© MIC-DKFZ). These files derive from it and are released
under the same license. See [`../LICENSE`](../LICENSE).
