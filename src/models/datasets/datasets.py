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


def _del_tail_if_no_target(er_list, label_sparse, target_num):
    target_num = torch.tensor(target_num)
    save_indices = [i for i in range(len(label_sparse))
                    if torch.any(label_sparse[i]._values() == target_num)]

    er_list = er_list[save_indices]
    label_sparse = torch.stack([label_sparse[i] for i in save_indices])
    return er_list, label_sparse


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
    er_list: torch.Tensor
    label_sparse: torch.sparse.Tensor

    def __init__(
            self, er_list: np.ndarray, label_sparse: torch.sparse.Tensor, target_num: int, *, del_if_no_tail=False):
        er_list = torch.from_numpy(er_list)
        if del_if_no_tail:
            er_list, label_sparse = _del_tail_if_no_target(er_list, label_sparse, target_num)
        label_sparse = _del_non_target(label_sparse, target_num)
        label_sparse = label_sparse.to(torch.float32)
        self.er_list = er_list
        self.label_sparse = label_sparse

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        er = self.er_list[index]
        e2s_target = self.label_sparse[index].to_dense()
        return er, e2s_target

    def __len__(self) -> int:
        return len(self.er_list)


@dataclasses.dataclass(init=False)
class MyDatasetWithFilter(Dataset):
    er_list: torch.Tensor
    label_sparse: torch.sparse.Tensor
    target_num: int

    def __init__(self, er_list: np.ndarray, label_sparse: torch.sparse.Tensor, target_num: int, del_if_no_tail=False):
        er_list = torch.from_numpy(er_list)
        if del_if_no_tail: er_list, label_sparse = _del_tail_if_no_target(er_list, label_sparse, target_num)
        # label_sparse = _del_non_target(label_sparse, target_num)
        # label_sparse = label_sparse.to(torch.float32)
        self.er_list = er_list
        self.label_sparse = label_sparse
        self.target_num = target_num

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        er = self.er_list[index]
        e2s_all = self.label_sparse[index].to_dense()
        e2s_target = (e2s_all == self.target_num).to(torch.float32)
        return er, e2s_target, e2s_all

    def __len__(self) -> int:
        return len(self.er_list)


@dataclasses.dataclass(init=False)
class MyTripleDataset(Dataset):
    head: torch.Tensor
    relation: torch.Tensor
    tail: torch.Tensor
    is_reverse_list: torch.Tensor
    er_list_index_list: torch.Tensor
    is_exist_list: torch.Tensor

    def __init__(self, triples: np.ndarray, r_is_reverse_list: np.ndarray, er2index: Dict[Tuple[int, int], int]):
        triples = torch.from_numpy(triples)
        head, relation, tail = torch.split(triples, [1, 1, 1], dim=1)

        is_reverse_list = torch.tensor([r_is_reverse_list[r] for r in relation], requires_grad=False, dtype=torch.int8)
        er_list_index_list = torch.tensor(
            [er2index[(e.item(), r.item())] for e, r, _ in triples], requires_grad=False, dtype=torch.int64)
        is_exist_list = torch.ones(len(triples), dtype=torch.bool)

        assert is_same_len_in_list(head, relation, tail, is_reverse_list, er_list_index_list, is_exist_list)
        self.head, self.relation, self.tail = head, relation, tail
        self.is_reverse_list = is_reverse_list
        self.er_list_index_list = er_list_index_list
        self.is_exist_list = is_exist_list

    def add_negative(self, negative_count: int, train_data: MyDataset, special_num: int):
        #  train_data = train_data  # data_helper.get_train_dataset(more_eco_memory=True)
        # todo
        negative_triples_list: list = []
        for er, tails in tqdm(train_data, leave=False):
            # er, e2s_target_raw = train_data.get_raw_items(i)
            tails[0:special_num] = -1
            negative_tails_indices: torch.Tensor = np.random.choice(len(tails), negative_count, p=(tails == 0))
            er = er.reshape(1, 2).repeat(negative_count, 1)
            negative_tails_indices = negative_tails_indices.reshape(negative_count, 1)
            negative_ere = torch.cat((er, negative_tails_indices), dim=1)
            negative_triples_list.append(negative_ere)

        negative_triples_tensor: torch.Tensor = torch.cat(negative_triples_list, dim=0)

        negative_er, negative_tail = negative_triples_tensor.split(2, 1)
        negative_index = -1 * torch.ones(len(negative_tail))

        self.er = torch.cat((self.er, negative_er), dim=0)
        self.tail = torch.cat((self.tail, negative_tail), dim=0)
        self.er_list_index = torch.cat((self.er_list_index, negative_index), dim=0)
        self.is_exist = torch.cat((self.is_exist, torch.zeros(len(negative_er), dtype=torch.bool)), dim=0)
        assert is_same_len_in_list(self.er, self.tail, self.er_list_index, self.is_exist), \
            "check the len of self.er, self.tail, self.er_list_index, self.is_exist"

    def __getitem__(self, index: int
                    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        head, relation, tail = self.head[index], self.relation[index], self.tail[index]
        is_reverse = self.is_reverse_list[index]
        er_list_index = self.er_list_index_list[index]
        is_exist = self.is_exist_list[index]
        return head, relation, tail, is_reverse, er_list_index, is_exist

    def __len__(self) -> int:
        return len(self.er)


if __name__ == '__main__':
    version_check(pd, np, h5py, Logger)
