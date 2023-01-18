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


def add_bos(triple: np.ndarray, head_bos_token: int, relation_bos_token: int, tail_bos_token: int):
    """Add bos token each head change.

    Args:
        triple(np.ndarray): triple.shape = [triple_len, 3]
        head_bos_token(int): head_bos_token
        relation_bos_token(int): relation_bos_token
        tail_bos_token(int): tail_bos_token

    Returns:
        np.ndarray: .shape = [(triple_len+story_num), 3]. story_num is the number of type of head.

    """
    bos_array = np.array([head_bos_token, relation_bos_token, tail_bos_token])
    new_triple_list = [np.stack([bos_array] + list(g)) for _, g in itertools.groupby(triple, itemgetter(0))]
    new_triple = np.concatenate(new_triple_list)
    return new_triple


@dataclass(init=False)
class SimpleTriple(Dataset):
    """Dataset of Simple Triple

    """
    triple: torch.Tensor

    def __init__(self, triple: np.ndarray):
        self.triple = torch.from_numpy(triple)

    def __getitem__(self, index: int):
        return self.triple[index]

    def __len__(self) -> int:
        return len(self.triple)


@dataclass(init=False)
class StoryTriple(Dataset):
    """Dataset using for Story Triple.

    """
    padding_tensor: torch.Tensor
    sep_tensor: torch.Tensor
    triple: torch.Tensor
    bos_indexes: torch.Tensor
    max_len: int
    _bos_end: torch.Tensor

    def __init__(self, triple, bos_indexes, max_len, padding_h, padding_r, padding_t, sep_h, sep_r, sep_t, ):
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
        # set variable
        self.padding_tensor = torch.tensor([padding_h, padding_r, padding_t])
        self.sep_tensor = torch.tensor([sep_h, sep_r, sep_t])
        self.triple = torch.from_numpy(np.concatenate((triple, triple[:max_len]))).clone()
        self.bos_indexes = torch.from_numpy(bos_indexes).clone()
        self.max_len = max_len
        # make bos_end
        self._bos_end = torch.stack((self.bos_indexes, self.bos_indexes + self.max_len)).T

    def shuffle_per_1scene(self):
        """shuffle per one scene.

        * example
        * before_triple: 0, 1, 2, 3, 4, 5, 6, ...
        * bos_indexes  : 0, 4,
        * after_triple: 0, 3, 1, 2, 4, 6, 5, ...

        """
        triple = self.triple
        for i, i_next in itertools.pairwise(list(self.bos_indexes) + [len(triple)]):
            triple[i + 1: i_next] = triple[i + 1: i_next][torch.randperm(i_next - (i + 1))]
        self.triple = triple

    def __getitem__(self, index: int):
        bos_index, end_index = self._bos_end[index]
        return self.triple[bos_index: end_index]

    def __len__(self) -> int:
        return len(self.bos_indexes)


@dataclass(init=False)
class StoryTripleForValid(StoryTriple):
    """Dataset using for Valid Story Triple.

    """
    valid_filter: torch.Tensor

    def __init__(
            self, triple, bos_indexes, valid_filter, max_len, padding_h, padding_r, padding_t, sep_h, sep_r, sep_t):
        """Dataset using for Valid Story Triple.

        * One output shape is [series_len, 3], not [3]

        Args:
            triple(np.ndarray): triple.shape==[len_of_triple, 3]
            bos_indexes(np.ndarray): the list of indexes
            valid_filter(np.ndarray): valid_filter.shape==[len_of_triple, 3]
            max_len(int): the max size of time series
            padding_h(int): head padding token
            padding_r(int): relation padding token
            padding_t(int): tail padding token
            sep_h(int): head sep token
            sep_r(int): relation sep token
            sep_t(int): tail sep token
        """
        super().__init__(triple, bos_indexes, max_len, padding_h, padding_r, padding_t, sep_h, sep_r, sep_t)
        assert len(triple) == len(valid_filter)
        self.valid_filter = torch.from_numpy(np.concatenate((valid_filter, valid_filter[:max_len])))

    def __getitem__(self, index: int):
        bos, end = self._bos_end[index]
        return self.triple[bos: end], self.valid_filter[bos: end]

    def shuffle_per_1scene(self):
        """Raise NotImplementedError

        """
        raise NotImplementedError("If Valid, This function never use.")


def main():
    """main

    """
    version_check(pd, np)
    pass


if __name__ == '__main__':
    main()
    pass
