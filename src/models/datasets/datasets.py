# coding: UTF-8
import os
import sys
from pathlib import Path

# ========== My Utils ==========
from utils.utils import version_check, is_same_len_in_list

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(os.path.join(PROJECT_DIR, 'src'))

# ========== python ==========
from logging import Logger
# noinspection PyUnresolvedReferences
from typing import List, Dict, Tuple, Optional, Union, Callable
# noinspection PyUnresolvedReferences
from tqdm import tqdm
import dataclasses
# ========== Machine learning ==========
import h5py
import numpy as np
import pandas as pd
# ========== torch ==========
import torch
from torch.utils.data import Dataset


def _del_tail_if_no_target(ers, label_sparse, target_num):
    target_num = torch.tensor(target_num)
    save_indices = [i for i in range(len(label_sparse))
                    if torch.any(label_sparse[i].coalesce().values() == target_num)]

    ers = ers[save_indices]
    label_sparse = torch.stack([label_sparse[i] for i in save_indices])
    return ers, label_sparse


def _del_non_target(label_sparse: torch.sparse.Tensor, target_num):
    label_sparse = label_sparse.coalesce()
    indices = label_sparse.indices()
    values = label_sparse.values()

    rev = torch.sparse_coo_tensor(
        indices[:, values == target_num], values[values == target_num], size=label_sparse.size()
    )

    return rev


@dataclasses.dataclass(init=False)
class MyDataset(Dataset):
    er_or_ee: torch.Tensor
    label_sparse: torch.sparse.Tensor

    def __init__(
            self, er_or_ee: np.ndarray, label_sparse: torch.sparse.Tensor, target_num: int, *, del_if_no_tail=False):
        er_or_ee = torch.from_numpy(er_or_ee)
        if del_if_no_tail:
            er_or_ee, label_sparse = _del_tail_if_no_target(er_or_ee, label_sparse, target_num)
        label_sparse = _del_non_target(label_sparse, target_num)
        label_sparse = label_sparse.to(torch.float32)
        self.er_or_ee = er_or_ee
        self.label_sparse = label_sparse
        assert len(label_sparse) == len(er_or_ee)
        assert er_or_ee.shape[1] == 2

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        er_or_ee = self.er_or_ee[index]
        all_target = self.label_sparse[index].to_dense()
        return er_or_ee, all_target

    def __len__(self) -> int:
        return len(self.er_or_ee)


@dataclasses.dataclass(init=False)
class MyDatasetWithFilter(Dataset):
    er_or_ee: torch.Tensor
    label_sparse: torch.sparse.Tensor
    target_num: int

    def __init__(self, er_or_ee: np.ndarray, label_sparse: torch.sparse.Tensor, target_num: int, del_if_no_tail=False):
        er_or_ee = torch.from_numpy(er_or_ee)
        if del_if_no_tail: er_or_ee, label_sparse = _del_tail_if_no_target(er_or_ee, label_sparse, target_num)
        self.er_or_ee = er_or_ee
        self.label_sparse = label_sparse
        self.target_num = target_num

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        er = self.er_or_ee[index]
        e2s_all = self.label_sparse[index].to_dense()
        e2s_target = (e2s_all == self.target_num).to(torch.float32)
        return er, e2s_target, e2s_all

    def __len__(self) -> int:
        return len(self.er_or_ee)


@dataclasses.dataclass(init=False)
class MyTripleDataset(Dataset):
    triple: torch.Tensor
    is_reverse: torch.Tensor
    index2er_or_ee_id: torch.Tensor
    is_exist: torch.Tensor

    @classmethod
    def default_init(cls, triple: np.ndarray,
                     r_is_reverse: np.ndarray, er_or_ee2id_dict: Optional[Dict[Tuple[int, int], int]], ):
        # noinspection PyTypeChecker
        triple_list: list = triple.tolist()

        is_reverse = torch.tensor(
            [r_is_reverse[r] for _, r, _ in triple_list], requires_grad=False, dtype=torch.bool)

        index2er_or_ee_id = torch.tensor(
            [er_or_ee2id_dict[(e, r)] for e, r, _ in triple_list],
            requires_grad=False, dtype=torch.int64
        )

        assert is_same_len_in_list(triple_list, is_reverse, index2er_or_ee_id)

        triple = torch.tensor(triple_list, requires_grad=False)
        is_reverse = is_reverse
        index2er_or_ee_id = index2er_or_ee_id
        return cls(triple, is_reverse, index2er_or_ee_id)

    def __init__(self, triple: torch.Tensor,
                 is_reverse: torch.Tensor, index2er_or_ee_id: torch.Tensor):
        self.triple = triple
        self.is_reverse = is_reverse
        self.index2er_or_ee_id = index2er_or_ee_id

    def __getitem__(self, index: int
                    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        head, relation, tail = self.triple[index]
        is_reverse = self.is_reverse[index]
        er_or_ee_id = self.index2er_or_ee_id[index]
        return head, relation, tail, is_reverse, er_or_ee_id

    def __len__(self) -> int:
        return len(self.triple)


@dataclasses.dataclass(init=False)
class SimpleTriple(Dataset):
    triple: torch.Tensor

    def __init__(self, triple: np.ndarray, ):
        self.triple = torch.from_numpy(triple)

    def __getitem__(self, index: int):
        return self.triple[index]

    def __len__(self) -> int:
        return len(self.triple)


@dataclasses.dataclass(init=False)
class StoryTriple(Dataset):
    padding_tensor: torch.Tensor
    sep_tensor: torch.Tensor
    triple: torch.Tensor
    bos_indices: torch.Tensor
    max_len: int

    def __init__(self, triple: np.ndarray,
                 bos_indices: np.ndarray,
                 max_len: int,
                 padding_h, padding_r, padding_t,
                 sep_h, sep_r, sep_t,
                 ):
        self.padding_tensor = torch.tensor([padding_h, padding_r, padding_t])
        self.sep_tensor = torch.tensor([sep_h, sep_r, sep_t])
        self.triple = torch.from_numpy(triple)
        self.bos_indices = torch.from_numpy(bos_indices)
        self.max_len = max_len

    def shuffle_in_1story(self):
        triple = self.triple
        len_triple = len(self.triple)
        bos_index = self.bos_indices[0]
        for bos_index_new in self.bos_indices[1:]:
            triple[bos_index+1: bos_index_new-1] = \
                triple[bos_index+1: bos_index_new-1][torch.randperm(bos_index_new-bos_index-2)]
            bos_index = bos_index_new
        triple[bos_index+1: len(triple)-1] = \
            triple[bos_index+1: len(triple)-1][torch.randperm(len(triple) - bos_index-2)]
        assert len_triple == len(triple)
        self.triple = triple


    def __getitem__(self, index: int):
        index_ = self.bos_indices[index]
        stories = self.triple[index_:]
        if len(stories) > self.max_len:
            return stories[:self.max_len]
        else:
            return torch.cat((stories, self.triple[0:self.max_len - len(stories)]))

    def __len__(self) -> int:
        return len(self.bos_indices)


@dataclasses.dataclass(init=False)
class StoryTripleForValid(StoryTriple):
    valid_filter: torch.Tensor

    def __init__(self, triple: np.ndarray,
                 bos_indices: np.ndarray,
                 valid_filter: np.ndarray,
                 max_len: int,
                 padding_h, padding_r, padding_t,
                 sep_h, sep_r, sep_t,
                 ):
        super().__init__(triple, bos_indices, max_len, padding_h, padding_r, padding_t, sep_h, sep_r, sep_t)
        assert len(triple) == len(valid_filter)
        self.valid_filter = torch.from_numpy(valid_filter)

    def __getitem__(self, index: int):
        triple = super(StoryTripleForValid, self).__getitem__(index)
        index_ = self.bos_indices[index]
        valid_filter = self.valid_filter[index_:]
        max_len = self.max_len
        if len(valid_filter) > max_len:
            rev = valid_filter[:max_len]
        else:
            rev = torch.cat((valid_filter, self.valid_filter[0:max_len - len(valid_filter)]))
        return triple, rev


def main():
    pass


if __name__ == '__main__':
    version_check(pd, np, h5py, Logger)
