import os
import copy
import warnings
import random
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
from logging import Logger
# noinspection PyUnresolvedReferences
from typing import List, Dict, Tuple, Optional, Callable, Union, Final, NamedTuple
import math


class MlpMixerLayer(nn.Module):
    def __init__(
            self, d_model: int, dim_feedforward: int = 2048,
            activation=F.gelu, norm_first=False, batch_first=True,
    ):
        super().__init__()
        self.mlp1 = nn.Linear(d_model, dim_feedforward)
        self.layer_norm = nn.LayerNorm(d_model)
        self.mlp2 = nn.Linear(dim_feedforward, d_model)
        self.activation = activation
        self.norm_first = norm_first

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm_first:
            _x = self.layer_norm(x)
            _x = self.mlp2(self.activation(self.mlp1(_x)))
            x = x + _x
        else:
            _x = x
            _x = self.mlp2(self.activation(self.mlp1(_x)))
            x = x + self.layer_norm(_x)
        return x


class MlpMixer(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for i in range(num_layers)])
        self.norm = norm

    def forward(self, x: torch.Tensor):
        output = x

        for _layer in self.layers:
            output = _layer(output)

        if self.norm is not None:
            output = self.norm(output)

        return output


def main():
    from tranformer import PositionalEncoding
    _batch_size = 2
    _sequence_len = 3
    _d_model = 4  # embedding len
    a = torch.zeros((_batch_size, _sequence_len, _d_model))
    num_layer = 4
    layer_ = MlpMixerLayer(_d_model)
    model = MlpMixer(layer_, num_layer)
    pe = PositionalEncoding(_d_model)
    b = model(pe(a))
    print(b)


if __name__ == '__main__':
    main()


