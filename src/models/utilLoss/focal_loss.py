#!/usr/bin/python
# -*- coding: utf-8 -*-
""" Focal Loss and some more.
* This file is used for Focal Loss.
todo:
    * Check for correct behavior.
"""
from typing import Final, Optional

import torch
from torch import nn
import torch.nn.functional as F

GAMMA: Final = 'gamma'


class FocalLoss(nn.Module):
    """This is focal loss class.

    * This is focal loss.
    """

    def __init__(self, weight: Optional[torch.Tensor] = None, gamma: float = 2.5, reduction='mean'):
        """__init__
        if gamma is larger, FocalLoss is more effectiveness.
        Args:
            weight(:obj:`torch.Tensor`, optional): weight
            gamma(float): gamma. 0.0~5.0
            reduction: 'mean' or 'sum'? I forgot.
        """
        nn.Module.__init__(self)
        self.gamma = gamma
        self.reduction = reduction
        self.weight = weight

    def forward(self, input_tensor, target_tensor):
        """forward

        * Check "Cross Entropy Loss". Almost same usage.
        Args:
            input_tensor(torch.Tensor): input tensor
            target_tensor(torch.Tensor): target tensor

        Returns:
            torch.Tensor: the loss by using input_tensor and target_tensor.
        """
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor,
            weight=self.weight,
            reduction=self.reduction
        )
