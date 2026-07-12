import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.registry import register_module


@register_module(parent="criterions")
class MetNetLoss(nn.Module):
    def __init__(self, loss_type: str = "cross_entropy", ignore_index: int = -1):
        super().__init__()
        self.loss_type = loss_type
        self.ignore_index = ignore_index

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred:   (B, C, H, W) logits for one lead time
        target: (B, H, W) bin indices for one lead time
        """
        if self.loss_type == "cross_entropy":
            return F.cross_entropy(pred, target, ignore_index=self.ignore_index)
        if self.loss_type == "mse":
            target_float = target.float().unsqueeze(1)
            return F.mse_loss(pred, target_float)
        raise ValueError(f"Unsupported loss_type: {self.loss_type}")
