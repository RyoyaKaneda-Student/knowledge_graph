from typing import Final

import torch
from torch import nn
import torch.nn.functional as F

GAMMA: Final = 'gamma'


class FocalLoss(nn.Module):
    """
    This is focal loss.
    """
    def __init__(self, weight=None, gamma=2.5, reduction='mean'):
        nn.Module.__init__(self)
        self.gamma = gamma
        self.reduction = reduction
        self.weight = weight

    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor,
            weight=self.weight,
            reduction=self.reduction
        )
