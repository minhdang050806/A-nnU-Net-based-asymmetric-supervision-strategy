"""nnUNet trainer = Dice + TopK10 (label_smoothing=0.05) + oversample_foreground=0.5.

Designed to stack with TS850 MR pretrain via -pretrained_weights.
"""
import numpy as np

from nnunetv2.training.loss.compound_losses import DC_and_topk_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerDiceTopK10_OS50_LS005(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json, device=None):
        import torch
        if device is None:
            device = torch.device('cuda')
        super().__init__(plans, configuration, fold, dataset_json, device=device)
        # boost foreground sampling: small structures dominate the loss
        self.oversample_foreground_percent = 0.5

    def _build_loss(self):
        assert not self.label_manager.has_regions, "regions not supported by this trainer"
        loss = DC_and_topk_loss(
            {"batch_dice": self.configuration_manager.batch_dice, "smooth": 1e-5,
             "do_bg": False, "ddp": self.is_ddp},
            {"k": 10, "label_smoothing": 0.05},
            weight_ce=1, weight_dice=1,
            ignore_label=self.label_manager.ignore_label,
        )
        if self.enable_deep_supervision:
            ds_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(ds_scales))])
            weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)
        return loss
