#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Datasets for story triple data

* Story triple data is like "Knowledge graph challenge" triple data.
* [[title.scene01, subject, Holmes  ],
*  [title.scene01, predict, standUp ],
*  [title.scene01, object, char     ],
* ...                               ]]

"""

# ========== python ==========
# noinspection PyUnresolvedReferences
from typing import List, Dict, Tuple, Optional, Union, Callable, Final
from dataclasses import dataclass
import itertools
from operator import itemgetter
# ========== Machine learning ==========
import numpy as np
import pandas as pd
# ========== torch ==========
import torch
from torch.utils.data import Dataset
# ========== My Utils ==========
from utils.utils import version_check


@dataclass(init=False)
class WN18RRDataset(Dataset):
    """Dataset using for WN18RR

    """
    sequence_tensor: torch.Tensor

    def __init__(self, sequence_array, ):
        """Dataset using for Story Triple.

        * One output shape is [series_len, 3], not [3]

        Args:
            triple(np.ndarray): triple.shape==[len_of_triple, 3]
            bos_indexes(np.ndarray): the list of indexes
            max_len(int): the max size of time series
            padding_h(int): head padding token
            padding_r(int): relation padding token
            padding_t(int): tail padding token
            sep_h(int): head sep token
            sep_r(int): relation sep token
            sep_t(int): tail sep token
        """
        self.sequence_tensor = torch.tensor(sequence_array).clone()

    def shuffle_per_1scene(self):
        """shuffle per one scene.

        * example
        * before_triple: 0, 1, 2, 3, 4, 5, 6, ...
        * bos_indexes  : 0, 4,
        * after_triple: 0, 3, 1, 2, 4, 6, 5, ...

        """
        sequence_tensor = self.sequence_tensor

        for j in range(sequence_tensor.shape[0]):
            bos_indexes = [0, *itertools.accumulate([len(g) for _, g in itertools.groupby(sequence_tensor[j, :, 0])])]
            for i, i_next in itertools.pairwise(bos_indexes):
                sequence_tensor[j, i: i_next] = sequence_tensor[j, i: i_next][torch.randperm(i_next - i)]
        self.sequence_tensor = sequence_tensor

    def __getitem__(self, index: int):
        return self.sequence_tensor[index, :3]

    def __len__(self) -> int:
        return len(self.sequence_tensor)


@dataclass(init=False)
class WN18RRDatasetForValid(WN18RRDataset):
    """Dataset using for Valid Story Triple.

    """
    valid_mode: int

    def __init__(self, sequence_array, valid_mode):
        super().__init__(sequence_array)
        self.valid_mode = valid_mode

    def shuffle_per_1scene(self):
        """Raise NotImplementedError

        """
        raise NotImplementedError("If Valid, This function never use.")

    def __getitem__(self, index: int):
        return self.sequence_tensor[index, :3], self.sequence_tensor[index, 3] == self.valid_mode


def main():
    """main

    """
    version_check(pd, np)
    pass


if __name__ == '__main__':
    main()
    pass
