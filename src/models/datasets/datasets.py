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


def main():
    pass


if __name__ == '__main__':
    version_check(pd, np, h5py, Logger)
