import os
import warnings
import random
import numpy as np

import torch
import torch.nn as nn
from logging import Logger
# noinspection PyUnresolvedReferences
from typing import List, Dict, Tuple, Optional, Callable, Union, Final, NamedTuple
import math

from utils.progress_manager import ProgressHelper
from utils.utils import none_count, is_same_len_in_list

INF = float('inf')


class FeedForwardNetworks(nn.Module):
    def __init__(self, d_model: int, dim_feedforward: int, activation=nn.ReLU) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.activate = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.activate(x)
        return self.linear2(x)


class PositionalEncoding(nn.Module):
    """ PositionalEncoding
    This is batch first.
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        x = self.dropout(x)
        return x


if __name__ == '__main__':
    a = torch.tensor([
        [[0, 0, 0, 0, ], [0, 0, 0, 0, ], [0, 0, 0, 0, ]],
        [[0, 0, 0, 0, ], [0, 0, 0, 0, ], [0, 0, 0, 0, ]],
    ])
    print(a.shape)  # (2, 3, 4)
    model = PositionalEncoding(4, dropout=0, )
    print(model(a))
