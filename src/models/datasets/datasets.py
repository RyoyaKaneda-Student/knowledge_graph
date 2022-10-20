# coding: UTF-8
import os
import sys
from pathlib import Path
# ========== My Utils ==========
from utils.torch import SparceData, random_indices_choice, onehot_target
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
    label_sparce_all: SparceData
    target_num: int
    item1_tmp: torch.Tensor
    entity_special_num: int
    relation_special_num: int

    def __init__(self, data: np.ndarray, label_sparce_all: SparceData, target_num: int,
                 len_e: int, del_if_no_tail=False, entity_special_num=0, relation_special_num=0):
        label_sparce_all.add_to_data_index(entity_special_num)

        if del_if_no_tail:
            del_indices = label_sparce_all.del_indices(
                key=lambda _data, _data_type: not torch.any(_data_type == target_num).item()
            )
            data = np.delete(data, [i for i, _if_deleted in enumerate(del_indices) if _if_deleted], 0)
            assert len(label_sparce_all) == len(data), f"{len(label_sparce_all)=}, {len(data)=}"

        self.data = torch.from_numpy(data) + torch.tensor([entity_special_num, relation_special_num])
        assert torch.count_nonzero(self.data[:, 0] < entity_special_num) == 0
        assert torch.count_nonzero(self.data[:, 1] < relation_special_num) == 0

        self.label_sparce_all = label_sparce_all
        self.target_num = target_num
        self.item1_tmp = torch.tensor([0] * (len_e + entity_special_num), dtype=torch.int8)
        self.entity_special_num = entity_special_num
        self.relation_special_num = relation_special_num

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        target_num = self.target_num
        er = self.data[index]
        e2s_target = self.label_sparce_all.get_data(
            index,
            filter_func=lambda _, data: (data == target_num).numpy(), fill_value=1
        ).to(torch.float32)
        return er, e2s_target

    def get_raw_items(self, index) -> Tuple[torch.Tensor, tuple[np.ndarray, Union[np.ndarray, torch.Tensor]]]:
        er = self.data[index]
        e2s_target_raw = self.label_sparce_all.get_raw_data(index)
        return er, e2s_target_raw

    def __len__(self) -> int:
        return len(self.data)


class MyDatasetMoreEcoWithFilter(MyDatasetMoreEcoMemory):
    def __getitem__(self, index: int
                    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        er = self.data[index]
        target_num = self.target_num

        e2s_all = self.label_sparce_all.get_data(index, )
        e2s_target = self.label_sparce_all.get_data(
            index, filter_func=lambda _, data: (data == target_num).numpy(), fill_value=1
        )
        e2s_target = e2s_target
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


@dataclasses.dataclass(init=False)
class MyTripleTrainDataset(Dataset):
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
        self.is_exist = onehot_target(0, 2, len(self.triples), dtype=torch.long)
        torch.add(self.er, torch.tensor([entity_special_num, relation_special_num]), out=self.er)
        torch.add(self.tail, torch.tensor([entity_special_num]), out=self.tail)
        self.entity_special_num = entity_special_num
        self.relation_special_num = relation_special_num

    def add_negative(self, negative_count: int, train_data: MyDatasetMoreEcoMemory, special_num: int):
        #  train_data = train_data  # data_helper.get_train_dataset(more_eco_memory=True)
        negative_triples_list: list = []
        for er, tails in tqdm(train_data, leave=False):
            # er, e2s_target_raw = train_data.get_raw_items(i)
            tails[0:special_num] = -1
            negative_tails_indices: torch.Tensor = np.random.choice(len(tails), negative_count, p=(tails == 0))
            er = er.reshape(1, 2).repeat(negative_count, 1)
            negative_tails_indices = negative_tails_indices.reshape(negative_count, 1)
            negative_ere = torch.cat((er, negative_tails_indices), dim=1)
            negative_triples_list.append(negative_ere)
            # print(negative_ere.shape)
        negative_triples_tensor: torch.Tensor = torch.cat(negative_triples_list, dim=0)
        print(negative_triples_tensor.shape)
        negative_er, negative_tail = negative_triples_tensor.split(2, 1)
        negative_index = -1 * torch.ones(len(negative_tail))

        self.er = torch.cat((self.er, negative_er), dim=0)
        self.tail = torch.cat((self.tail, negative_tail), dim=0)
        self.er_list_index = torch.cat((self.er_list_index, negative_index), dim=0)
        self.is_exist = torch.cat((self.is_exist, onehot_target(1, 2, len(negative_er))), dim=0)
        assert is_same_len_in_list(self.er, self.tail, self.er_list_index, self.is_exist), \
            "check the len of self.er, self.tail, self.er_list_index, self.is_exist"

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        er = self.er[index]
        tail = self.tail[index]
        is_exist = self.is_exist[index]
        return er, tail, is_exist

    def __len__(self) -> int:
        return len(self.er)


if __name__ == '__main__':
    version_check(pd, np, h5py, Logger)
