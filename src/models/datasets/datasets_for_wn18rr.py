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
# ========== Machine learning ==========
import numpy as np
import pandas as pd
# ========== torch ==========
import torch
from torch.utils.data import Dataset
# ========== My Utils ==========
from utils.utils import version_check
# ========== My Dataset ==========
from models.datasets.datasets_for_sequence import SequenceTriple


@dataclass(init=False)
class WN18RRDataset(SequenceTriple):
    """Dataset using for WN18RR

    """
    #
    head_index2count: torch.Tensor
    relation_index2count: torch.Tensor
    tail_index2count: torch.Tensor
    #
    sequence_tensor: torch.Tensor

    def __init__(self, triple, sequence_array, entity_num, relation_num):
        """Dataset using for Story Triple.

        * One output shape is [series_len, 3], not [3]

        Args:
            triple(np.ndarray|None): triple.shape==[len_of_triple, 3]
            sequence_array(np.ndarray): sequence_array.shape = [sequence_array, max_ken, 3]
            entity_num(int): entity_num
            relation_num(int): relation_num
        """
        super(WN18RRDataset, self).__init__()
        if triple is not None:
            triple = torch.tensor(triple).clone()
            head_index2count = torch.bincount(triple[:, 0], minlength=entity_num).to(torch.float)
            relation_index2count = torch.bincount(triple[:, 1], minlength=relation_num).to(torch.float)
            tail_index2count = torch.bincount(triple[:, 2], minlength=entity_num).to(torch.float)
        else:
            head_index2count, relation_index2count, tail_index2count = None, None, None

        # set index2count
        self.head_index2count = head_index2count
        self.relation_index2count = relation_index2count
        self.tail_index2count = tail_index2count
        # set sequence
        self.sequence_tensor = torch.tensor(sequence_array).clone()

    def per1epoch(self):
        """per1epoch

        """
        self.shuffle_per_head()

    def shuffle_per_head(self):
        """shuffle per one scene.

        * example
        * before_triple: 0, 1, 2, 3, 4, 5, 6, ...
        * bos_indexes  : 0, 4,
        * after_triple: 0, 3, 1, 2, 4, 6, 5, ...

        """
        sequence_tensor = self.sequence_tensor

        for j in range(len(sequence_tensor)):
            triples = sequence_tensor[j]
            bos_indexes = [
                0, *itertools.accumulate([len(list(g)) for _, g in itertools.groupby(triples[:, 0])]), len(triples)]
            for i, i_next in itertools.pairwise(bos_indexes):
                sequence_tensor[j, i: i_next] = sequence_tensor[j, i: i_next][torch.randperm(i_next - i)]
        self.sequence_tensor = sequence_tensor

    def get_index2count(self, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """get index2count

        """
        return self.head_index2count.to(device),  self.relation_index2count.to(device), self.tail_index2count.to(device)

    def __getitem__(self, index: int):
        return self.sequence_tensor[index, :, :3]

    def __len__(self) -> int:
        return len(self.sequence_tensor)


@dataclass(init=False)
class WN18RRDatasetForValid(WN18RRDataset):
    """Dataset using for Valid Story Triple.

    """
    valid_mode: int

    def __init__(self, triple, sequence_array, entity_num, relation_num, valid_mode):
        super().__init__(triple, sequence_array, entity_num, relation_num)
        self.valid_mode = valid_mode
        self.valid_mode_filter = self.sequence_tensor[:, :, 3] == self.valid_mode

    def shuffle_per_1scene(self):
        """Raise NotImplementedError

        """
        raise NotImplementedError("If Valid, This function never use.")

    def __getitem__(self, index: int):
        return self.sequence_tensor[index, :, :3], self.valid_mode_filter[index]


def main():
    """main

    """
    version_check(pd, np)
    pass


if __name__ == '__main__':
    main()
    pass
