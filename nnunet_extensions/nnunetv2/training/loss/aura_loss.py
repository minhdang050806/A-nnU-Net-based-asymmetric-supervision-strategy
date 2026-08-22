"""AURA loss for asymmetric HF/LF supervision in LISA Dataset002.

AURA keeps the scored HF annotation as the anchor target and lets the LF
annotation contribute only through a bounded reliability gate. The gate is high
where HF/LF disagree near anatomical boundaries and the model is uncertain, and
low in confident interiors. This gives a small, falsifiable variant of
asymmetric dual-observation learning without changing the nnU-Net network.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class AURAAsymmetricLoss(nn.Module):
    def __init__(
        self,
        num_classes: int = 12,
        max_lf_weight: float = 0.35,
        label_smoothing: float = 0.03,
        weight_ce: float = 1.0,
        weight_dice: float = 1.0,
        boundary_weight: float = 0.03,
        topk: int = 10,
        ignore_label: int = -1,
        warmup_epochs: int = 25,
        ramp_epochs: int = 100,
        class_lf_weights: Sequence[float] | None = None,
        class_dice_weights: Sequence[float] | None = None,
        smooth: float = 1e-5,
    ):
        super().__init__()
        self.C = int(num_classes)
        self.max_lf_weight = float(max_lf_weight)
        self.label_smoothing = float(label_smoothing)
        self.weight_ce = float(weight_ce)
        self.weight_dice = float(weight_dice)
        self.boundary_weight = float(boundary_weight)
        self.topk = int(topk)
        self.ignore_label = int(ignore_label)
        self.warmup_epochs = int(warmup_epochs)
        self.ramp_epochs = max(1, int(ramp_epochs))
        self.smooth = float(smooth)
        self.current_epoch = 0

        if class_lf_weights is None:
            class_lf_weights = (
                0.00,  # background
                1.20,  # 1 left hippocampus
                1.20,  # 2 right hippocampus
                0.85,  # 3 left ventricle
                0.85,  # 4 right ventricle
                0.95,  # 5 left caudate
                0.95,  # 6 right caudate
                0.95,  # 7 left lentiform
                0.95,  # 8 right lentiform
                0.90,  # 9 left thalamus
                0.90,  # 10 right thalamus
                0.80,  # 11 corpus callosum
            )
        if len(class_lf_weights) != self.C:
            raise ValueError(f"class_lf_weights length must be {self.C}, got {len(class_lf_weights)}")
        self.register_buffer("class_lf_weights", torch.tensor(class_lf_weights, dtype=torch.float32), persistent=True)

        if class_dice_weights is None:
            class_dice_weights = (
                1.25,  # 1 left hippocampus
                1.25,  # 2 right hippocampus
                1.00,  # 3 left ventricle
                1.00,  # 4 right ventricle
                1.00,  # 5 left caudate
                1.00,  # 6 right caudate
                1.00,  # 7 left lentiform
                1.00,  # 8 right lentiform
                1.00,  # 9 left thalamus
                1.00,  # 10 right thalamus
                1.15,  # 11 corpus callosum
            )
        if len(class_dice_weights) != self.C - 1:
            raise ValueError(f"class_dice_weights length must be {self.C - 1}, got {len(class_dice_weights)}")
        dice_w = torch.tensor(class_dice_weights, dtype=torch.float32)
        dice_w = dice_w / dice_w.mean().clamp_min(1e-6)
        self.register_buffer("class_dice_weights", dice_w, persistent=True)

    def _ramp(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        value = (float(self.current_epoch) - float(self.warmup_epochs)) / float(self.ramp_epochs)
        value = min(1.0, max(0.0, value))
        return torch.tensor(value, device=device, dtype=dtype)

    @staticmethod
    def _boundary_mask(label: torch.Tensor) -> torch.Tensor:
        x = label.float().unsqueeze(1)
        p_max = F.max_pool3d(x, kernel_size=3, padding=1, stride=1)
        p_min = -F.max_pool3d(-x, kernel_size=3, padding=1, stride=1)
        return (p_max - p_min > 0).float()

    def _one_hot(self, label: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return F.one_hot(label.clamp(min=0).long(), num_classes=self.C).permute(0, 4, 1, 2, 3).to(dtype)

    def _make_target_soft(
        self,
        logits: torch.Tensor,
        hf: torch.Tensor,
        lf: torch.Tensor,
        valid_f: torch.Tensor,
    ) -> torch.Tensor:
        hf_oh = self._one_hot(hf, logits.dtype)
        if self.label_smoothing > 0:
            hf_soft = hf_oh * (1.0 - self.label_smoothing) + self.label_smoothing / self.C
        else:
            hf_soft = hf_oh
        hf_soft = hf_soft * valid_f

        with torch.no_grad():
            ramp = self._ramp(logits.device, torch.float32)
            if ramp.item() == 0.0 or self.max_lf_weight <= 0:
                return hf_soft

            prob = torch.softmax(logits.float(), dim=1)
            entropy = -(prob * torch.log(prob.clamp_min(1e-9))).sum(dim=1, keepdim=True)
            entropy = (entropy / torch.log(torch.tensor(float(self.C), device=logits.device))).clamp(0, 1)
            boundary = ((self._boundary_mask(hf) + self._boundary_mask(lf)) > 0).float().to(logits.device)
            disagree = (hf != lf).float().unsqueeze(1).to(logits.device)

            class_w = self.class_lf_weights.to(logits.device)[hf.clamp(min=0).long()].unsqueeze(1)
            lf_gate = (
                self.max_lf_weight
                * ramp
                * class_w
                * (0.12 + 0.88 * disagree)
                * (0.35 + 0.65 * boundary)
                * (0.35 + 0.65 * entropy)
            ).clamp(0.0, self.max_lf_weight * 1.20)
            lf_gate = lf_gate.to(logits.dtype) * valid_f

            lf_oh = self._one_hot(lf, logits.dtype) * valid_f
            target_soft = (1.0 - lf_gate) * hf_soft + lf_gate * lf_oh
            target_soft = target_soft * valid_f
            target_soft = target_soft / target_soft.sum(dim=1, keepdim=True).clamp_min(1e-8)
            return target_soft

    def _soft_topk_ce(self, logits: torch.Tensor, target_soft: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        log_p = F.log_softmax(logits, dim=1)
        ce_per_voxel = -(target_soft * log_p).sum(dim=1)
        ce_valid = ce_per_voxel[valid]
        if ce_valid.numel() == 0:
            return logits.new_zeros(())
        if self.topk:
            k = max(1, ce_valid.numel() * self.topk // 100)
            return ce_valid.topk(k)[0].mean()
        return ce_valid.mean()

    def _soft_dice(self, prob: torch.Tensor, target_soft: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        p_fg = prob[:, 1:] * mask
        t_fg = target_soft[:, 1:].float() * mask
        inter = (p_fg * t_fg).sum(dim=(0, 2, 3, 4))
        denom = p_fg.sum(dim=(0, 2, 3, 4)) + t_fg.sum(dim=(0, 2, 3, 4))
        dice_per_class = 2.0 * inter / (denom + self.smooth)
        weights = self.class_dice_weights.to(prob.device)
        return ((1.0 - dice_per_class) * weights).sum() / weights.sum()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.ndim == logits.ndim - 1:
            target = target.unsqueeze(1)
        assert target.shape[1] == 2, f"AURA expects target channels [HF, LF], got {target.shape}"

        hf = target[:, 0]
        lf = target[:, 1]
        valid = (hf != self.ignore_label) & (lf != self.ignore_label)
        valid_f = valid.unsqueeze(1).to(logits.dtype)

        target_soft = self._make_target_soft(logits, hf, lf, valid_f)
        ce = self._soft_topk_ce(logits, target_soft, valid)

        prob = torch.softmax(logits.float(), dim=1)
        dice = self._soft_dice(prob, target_soft, valid_f.float())

        boundary_loss = logits.new_zeros(())
        if self.boundary_weight > 0:
            with torch.no_grad():
                boundary = ((self._boundary_mask(hf) + self._boundary_mask(lf)) > 0).float().to(logits.device)
                boundary = boundary * valid_f.float()
            if boundary.sum() > 0:
                boundary_loss = self._soft_dice(prob, target_soft, boundary)

        return self.weight_ce * ce + self.weight_dice * dice + self.boundary_weight * boundary_loss
