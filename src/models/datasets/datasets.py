# coding: UTF-8
import os
import sys
from pathlib import Path
# ========== My Utils ==========
from utils.utils import version_check

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


@dataclasses.dataclass(init=False)
class MyDataset(Dataset):
    data: torch.Tensor
    label: torch.Tensor

    def __init__(self, data: np.ndarray, label: np.ndarray, target_num: int, del_if_no_tail=False):
        if del_if_no_tail:
            tmp = np.count_nonzero(label == target_num, axis=1) > 0
            data = data[tmp]
            label = label[tmp]

        self.data = torch.from_numpy(data)
        self.label = torch.from_numpy(label == target_num).to(torch.float32)
        self.del_if_no_tail = del_if_no_tail

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        er = self.data[index]
        e2s = self.label[index]
        return er, e2s

    def __len__(self) -> int:
        return len(self.data)


@dataclasses.dataclass(init=False)
class MyDatasetEcoMemory(Dataset):
    data: torch.Tensor
    label_all: torch.Tensor
    target_num: int

    def __init__(self, data: np.ndarray, label: np.ndarray, target_num: int, del_if_no_tail=False):
        if del_if_no_tail:
            tmp = np.count_nonzero(label == target_num, axis=1) > 0
            data = data[tmp]
            label = label[tmp]
        self.data = torch.from_numpy(data)
        self.label_all = torch.from_numpy(label)
        self.target_num = target_num

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        er = self.data[index]
        e2s = self.label_all[index]
        e2s = (e2s == self.target_num).to(torch.float32)
        return er, e2s

    def __len__(self) -> int:
        return len(self.data)


@dataclasses.dataclass(init=False)
class MyDatasetWithFilter(MyDatasetEcoMemory):
    def __getitem__(self, index: int
                    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        er = self.data[index]
        e2s_all = self.label_all[index]
        e2s_target = (e2s_all == self.target_num).to(torch.float32)
        return er, e2s_target, e2s_all


@dataclasses.dataclass(init=False)
class MyDatasetMoreEcoMemory(Dataset):
    data: torch.Tensor
    label_sparce_all: List[Tuple[np.ndarray, torch.Tensor]]
    target_num: int
    item1_tmp: torch.Tensor
    entity_special_num: int
    relation_special_num: int

    def __init__(self, data: np.ndarray, label_sparce_all: List[Tuple[np.ndarray, np.ndarray]], target_num: int,
                 len_e: int, del_if_no_tail=False, entity_special_num=0, relation_special_num=0):
        label_sparce_all = [(_data + entity_special_num, _data_type) for (_data, _data_type) in
                            label_sparce_all]
        if del_if_no_tail:
            index_, label_sparce_all = zip(*[
                (i, (_data, _data_type)) for i, (_data, _data_type) in enumerate(label_sparce_all)
                if np.any(_data_type == target_num)
            ])
            data = data[index_, :]

        self.data = torch.from_numpy(data) + torch.tensor([entity_special_num, relation_special_num])
        assert torch.count_nonzero(self.data[:, 0] < entity_special_num) == 0
        assert torch.count_nonzero(self.data[:, 1] < relation_special_num) == 0
        for (_data, _data_type) in label_sparce_all:
            assert _data.shape == _data_type.shape
        self.label_sparce_all = [
            (_data, torch.tensor(_data_type)) for _data, _data_type in label_sparce_all
        ]
        self.target_num = target_num
        self.item1_tmp = torch.tensor([0] * (len_e + entity_special_num), dtype=torch.int8)
        self.entity_special_num = entity_special_num
        self.relation_special_num = relation_special_num

    def getitem(self, index) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        er = self.data[index]
        data, data_type = self.label_sparce_all[index]
        e2s_all = self.item1_tmp.clone()
        e2s_all[data] = data_type
        e2s_target = (e2s_all == self.target_num).to(torch.float32)
        return er, e2s_target, e2s_all

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        er, e2s_target, _ = self.getitem(index)
        return er, e2s_target

    def __len__(self) -> int:
        return len(self.data)


class MyDatasetMoreEcoWithFilter(MyDatasetMoreEcoMemory):

    def __getitem__(self, index: int
                    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        er, e2s_target, e2s_all = self.getitem(index)
        # e2s_all[e2s_all == 3] = 0  # 3 is Test
        return er, e2s_target, e2s_all


@dataclasses.dataclass(init=False)
class MyTripleDataset(Dataset):
    triples: torch.Tensor
    er: torch.Tensor
    tail: torch.Tensor
    is_reverse: torch.Tensor
    er_list_index: torch.Tensor
    entity_special_num: int
    relation_special_num: int

    def __init__(self, triples: np.ndarray, r_is_reverse_list: np.ndarray, er2index: Dict[Tuple[int, int], int],
                 entity_special_num=0, relation_special_num=0):
        self.triples = torch.from_numpy(triples)
        self.er, self.tail = self.triples.split(2, 1)
        self.r_is_reverse_list = torch.from_numpy(r_is_reverse_list)
        self.er_list_index = torch.tensor(
            [er2index[(e.item(), r.item())] for e, r in self.er], requires_grad=False, dtype=torch.int64
        )
        torch.add(self.er, torch.tensor([entity_special_num, relation_special_num]), out=self.er)
        torch.add(self.tail, torch.tensor([entity_special_num]), out=self.tail)
        self.entity_special_num = entity_special_num
        self.relation_special_num = relation_special_num

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        er = self.er[index]
        tail = self.tail[index]
        er_list_index = self.er_list_index[index]
        return er, tail, er_list_index

    def __len__(self) -> int:
        return len(self.er)


if __name__ == '__main__':
    version_check(pd, np, h5py, Logger)
