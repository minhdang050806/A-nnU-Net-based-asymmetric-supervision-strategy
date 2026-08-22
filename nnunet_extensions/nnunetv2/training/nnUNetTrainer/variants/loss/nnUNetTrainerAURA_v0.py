"""AURA v0 trainer for Dataset002_LISA_VCF.

The method keeps HF labels as the scored anchor and uses LF labels only through
an uncertainty/disagreement/boundary gate. This is intended as the first
paper-facing ablation for asymmetric HF/LF supervision, initialized from the
strong OS50_LS005 Dataset001 checkpoint.
"""
from __future__ import annotations

import numpy as np

from nnunetv2.training.loss.aura_loss import AURAAsymmetricLoss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerAURA_v0(nnUNetTrainer):
    AURA_LF_WEIGHT: float = 0.35
    AURA_LABEL_SMOOTHING: float = 0.03
    AURA_TOPK: int = 10
    AURA_BOUNDARY_WEIGHT: float = 0.03
    AURA_WARMUP_EPOCHS: int = 25
    AURA_RAMP_EPOCHS: int = 100
    DISABLE_LR_MIRROR_AXIS: int = 2

    def __init__(self, plans, configuration, fold, dataset_json, device=None):
        import torch

        if device is None:
            device = torch.device("cuda")
        super().__init__(plans, configuration, fold, dataset_json, device=device)
        self.oversample_foreground_percent = 0.5

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        rot, do_dummy, init_patch, mirror_axes = (
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        )
        if mirror_axes is not None and self.DISABLE_LR_MIRROR_AXIS in mirror_axes:
            new_axes = tuple(a for a in mirror_axes if a != self.DISABLE_LR_MIRROR_AXIS)
            self.print_to_log_file(
                f"[AURA-v0] Disabling L-R mirror augmentation: "
                f"mirror_axes {mirror_axes} -> {new_axes}"
            )
            mirror_axes = new_axes
            self.inference_allowed_mirroring_axes = mirror_axes
        return rot, do_dummy, init_patch, mirror_axes

    def _build_loss(self):
        assert not self.label_manager.has_regions, "regions not supported"
        ignore_label = self.label_manager.ignore_label if self.label_manager.has_ignore_label else -1
        loss = AURAAsymmetricLoss(
            num_classes=self.label_manager.num_segmentation_heads,
            max_lf_weight=self.AURA_LF_WEIGHT,
            label_smoothing=self.AURA_LABEL_SMOOTHING,
            topk=self.AURA_TOPK,
            boundary_weight=self.AURA_BOUNDARY_WEIGHT,
            ignore_label=int(ignore_label),
            warmup_epochs=self.AURA_WARMUP_EPOCHS,
            ramp_epochs=self.AURA_RAMP_EPOCHS,
        )
        if self.enable_deep_supervision:
            ds_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(ds_scales))])
            weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)
        return loss

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        inner = self.loss.loss if isinstance(self.loss, DeepSupervisionWrapper) else self.loss
        if isinstance(inner, AURAAsymmetricLoss):
            inner.current_epoch = int(self.current_epoch)
            if self.current_epoch == self.AURA_WARMUP_EPOCHS:
                self.print_to_log_file(
                    f"[AURA-v0] Warmup ended at epoch {self.current_epoch}; "
                    "activating asymmetric LF reliability gate."
                )
