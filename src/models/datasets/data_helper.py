# coding: UTF-8
# region !import area!
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(os.path.join(PROJECT_DIR, 'src'))

# python
from logging import Logger
from typing import Dict, Tuple, Optional, Union  # Callable
import dataclasses
# from tqdm import tqdm
# from argparse import Namespace
# Machine learning
import h5py
import numpy as np
import pandas as pd
import optuna
# torch
import torch
from torch.utils.data.dataloader import DataLoader

from models.datasets.datasets import (
    MyDataset, MyDatasetWithFilter, MyTripleDataset
)
from utils.utils import version_check

# endregion


PROCESSED_DATA_PATH = './data/processed/'
EXTERNAL_DATA_PATH = './data/external/'


@dataclasses.dataclass(init=False)
class MyTrainTestData:
    train_path: str
    valid_path: str
    test_path: str
    is_use_zero: bool
    e_length: int
    r_length: int
    e_dict: dict
    r_dict: dict
    r_is_reverse_list: np.ndarray
    er_list: np.ndarray
    er_is_reverse: np.ndarray
    er2index: Dict[Tuple[int, int], int]
    er_tails_data: np.ndarray
    er_tails_row: np.ndarray
    er_tails_data_type: np.ndarray
    _train_triple: Optional[np.ndarray]
    _valid_triple: Optional[np.ndarray]
    _test_triple: Optional[np.ndarray]

    def __init__(self, info_path, train_path, valid_path, test_path, del_zero2zero, *, logger):
        self.train_path, self.valid_path, self.test_path = (
            train_path, valid_path, test_path
        )
        self.is_use_zero = not del_zero2zero
        with h5py.File(info_path, 'r') as f:
            self.e_length = f['e_length'][()] - (1 if del_zero2zero else 0)
            self.r_length = f['r_length'][()] - (1 if del_zero2zero else 0)
            self.r_is_reverse_list = f['item_r_is_reverse'][(1 if del_zero2zero else 0):]
            self.e_dict = {i: bite_.decode() for i, bite_ in enumerate(f['item_e'][(1 if del_zero2zero else 0):])}
            self.r_dict = {i: bite_.decode() for i, bite_ in enumerate(f['item_r'][(1 if del_zero2zero else 0):])}
            logger.debug(self.r_dict)

        self._get_er_tails()
        self._train_triple = None
        # self._get_train()
        # self._get_valid()
        # self._get_test()

    def _get_er_tails(self):
        tmp = 0 if self.is_use_zero else 1
        with h5py.File(self.train_path) as f:
            self.er_list = f['er_list'][tmp:] - tmp
            self.er_is_reverse = f['er_is_reverse'][tmp:]
            self.er_tails_data = f['er_tails_data'][tmp:] - tmp
            self.er_tails_row = f['er_tails_row'][tmp:] - tmp
            self.er_tails_data_type = f['er_tails_data_type'][tmp:]

        self.er2index = {tuple(er.tolist()): i for i, er in enumerate(self.er_list)}

    def _get_train(self):
        tmp = 0 if self.is_use_zero else 1
        with h5py.File(self.train_path) as f:
            self._train_triple = f['triple'][tmp:] - tmp
            pass

    def _get_valid(self):
        tmp = 0 if self.is_use_zero else 1
        with h5py.File(self.valid_path) as f:
            self._valid_triple = f['triple'][tmp:] - tmp
            pass

    def _get_test(self):
        tmp = 0 if self.is_use_zero else 1
        with h5py.File(self.test_path) as f:
            self._test_triple = f['triple'][tmp:] - tmp
            pass

    @property
    def train_triple(self):
        self._get_train()
        return self._train_triple

    @property
    def valid_triple(self):
        self._get_valid()
        return self._valid_triple

    @property
    def test_triple(self):
        self._get_test()
        return self._test_triple

    def __del__(self):
        pass

    def show(self, logger: Logger):
        logger.info("==========Show TrainTestData==========")
        logger.info(f"is_use_zero: {self.is_use_zero}")
        logger.info(f"e_length: {self.e_length}")
        logger.info(f"r_length: {self.r_length}")
        logger.info(f"r_dict: {self.r_dict}")
        logger.info(f"er_list size: {self.er_list.shape}")
        logger.info("==========Show TrainTestData==========")


class MyDataHelper:
    def __init__(self, info_path, train_path, valid_path, test_path, del_zero2zero=True, *,
                 logger=None, eco_memory, entity_special_num, relation_special_num):
        super().__init__()
        my_data = MyTrainTestData(info_path, train_path, valid_path, test_path, del_zero2zero, logger=logger)

        self._data: MyTrainTestData = my_data
        self._eco_memory: bool = eco_memory
        self._entity_special_num: int = entity_special_num
        self._relation_special_num: int = relation_special_num

        self._train_dataloader = None
        self._valid_dataloader = None
        self._test_dataloader = None

    # region get dataset functions
    def _get_dataset(self, target_num, is_need_filter) -> Union[MyDataset, MyDatasetWithFilter]:
        target_num = target_num
        er_list = self.processed_er_list
        label_sparse = self.label_sparse
        if not is_need_filter:  # if train
            return MyDataset(er_list, label_sparse, target_num, del_if_no_tail=False)
        else:
            return MyDatasetWithFilter(er_list, label_sparse, target_num, del_if_no_tail=True)

    def get_train_dataset(self) -> MyDataset:
        return self._get_dataset(1, is_need_filter=False)

    def get_train_valid_dataset(self) -> MyDataset:
        """
        use for debug.
        """
        return self._get_dataset(1, is_need_filter=True)

    def get_valid_dataset(self) -> MyDatasetWithFilter:
        return self._get_dataset(2, is_need_filter=True)

    def get_test_dataset(self) -> MyDatasetWithFilter:
        return self._get_dataset(3, is_need_filter=True)

    # endregion

    # region get triple functions
    def _get_triple_dataset(self, triple) -> MyTripleDataset:
        e_special_num, r_special_num = self.get_er_special_num()
        triple = triple + np.array([e_special_num, r_special_num, e_special_num])
        r_is_reverse_list, er2index = self.processed_r_is_reverse_list, self.processed_er2index

        return MyTripleDataset(triple, r_is_reverse_list, er2index)

    def get_train_triple_dataset(self) -> MyTripleDataset:
        return self._get_triple_dataset(self.data.train_triple)

    def get_valid_triple_dataset(self) -> MyTripleDataset:
        return self._get_triple_dataset(self.data.valid_triple)

    def get_test_triple_dataset(self) -> MyTripleDataset:
        return self._get_triple_dataset(self.data.test_triple)

    # endregion

    # region loader functions
    def del_loaders(self):
        del self._train_dataloader, self._valid_dataloader, self._test_dataloader

    def del_test_dataloader(self):
        del self._test_dataloader

    def set_loaders(self, train_dataloader: DataLoader, valid_dataloader: DataLoader, test_dataloader: DataLoader):
        self._train_dataloader = train_dataloader
        self._valid_dataloader = valid_dataloader
        self._test_dataloader = test_dataloader

    @property
    def train_dataloader(self) -> DataLoader:
        return self._train_dataloader

    @property
    def trainvalid_dataloader(self) -> DataLoader:
        return self._train_dataloader

    @property
    def valid_dataloader(self) -> DataLoader:
        _valid_dataloader = self._valid_dataloader
        assert _valid_dataloader is not None, "valid_dataloader is not Defined"
        return _valid_dataloader

    @property
    def test_dataloader(self) -> DataLoader:
        _test_dataloader = self._test_dataloader
        assert _test_dataloader is not None, "test_dataloader is not Defined"
        return _test_dataloader

    # endregion

    def get_er_special_num(self) -> Tuple[int, int]:
        return self._entity_special_num, self._relation_special_num

    @property
    def data(self) -> MyTrainTestData:
        return self._data

    @property
    def processed_r_is_reverse_list(self):
        _, r_special_num = self.get_er_special_num()
        r_is_reverse_list = self.data.r_is_reverse_list

        r_is_reverse_list = np.concatenate([np.zeros(r_special_num, dtype=np.bool), r_is_reverse_list])
        return r_is_reverse_list

    @property
    def processed_er2index(self):
        e_special_num, r_special_num = self.get_er_special_num()
        er2index = self.data.er2index
        er2index_new = dict()
        for key, value in er2index.items():
            er2index_new[key[0] + e_special_num, key[1] + r_special_num] = value
        return er2index_new

    @property
    def processed_e_length(self) -> int:
        return self.data.e_length + self._entity_special_num

    @property
    def processed_r_length(self) -> int:
        return self.data.r_length + self._relation_special_num

    @property
    def r_is_reverse_length(self) -> int:
        assert self.data.r_length % 2 == 0
        return self.data.r_length // 2

    @property
    def processed_r_length_without_reverse(self) -> int:
        assert self.data.r_length % 2 == 0
        return self.data.r_length // 2 + self._relation_special_num

    @property
    def processed_er_list(self) -> np.ndarray:
        entity_special_num, relation_special_num = self.get_er_special_num()
        return self._data.er_list + np.array([entity_special_num, relation_special_num])

    @property
    def label_sparse_for_debug(self) -> torch.sparse.Tensor:
        entity_special_num, relation_special_num = self.get_er_special_num()
        e_length, r_length = self.processed_e_length, self.processed_r_length

        my_data = self.data
        er_list: np.ndarray = my_data.er_list
        er_tails_row: np.ndarray = np.arange(0, len(er_list))
        er_tails_data: np.ndarray = np.copy(my_data.er_list[:, 0])
        er_tails_data_type: np.ndarray = np.full(len(er_list), 1)  # 1 is train number

        tails_row_column = torch.from_numpy(np.stack([er_tails_row, er_tails_data + entity_special_num]))
        tails_value = torch.from_numpy(er_tails_data_type)

        rev = torch.sparse_coo_tensor(tails_row_column, tails_value,
                                      size=(len(er_list), e_length),
                                      dtype=torch.int8, requires_grad=False)

        return rev

    @property
    def label_sparse(self) -> torch.sparse.Tensor:
        entity_special_num, relation_special_num = self.get_er_special_num()
        e_length, r_length = self.processed_e_length, self.processed_r_length

        my_data = self.data
        er_list: np.ndarray = my_data.er_list
        er_tails_row: np.ndarray = np.copy(my_data.er_tails_row)
        er_tails_data: np.ndarray = np.copy(my_data.er_tails_data)
        er_tails_data_type: np.ndarray = np.copy(my_data.er_tails_data_type)

        tails_row_column = torch.from_numpy(np.stack([er_tails_row, er_tails_data + entity_special_num]))
        tails_value = torch.from_numpy(er_tails_data_type)

        rev = torch.sparse_coo_tensor(tails_row_column, tails_value,
                                      size=(len(er_list), e_length),
                                      dtype=torch.int8, requires_grad=False)

        return rev

    @property
    def processed_e_dict(self) -> Dict:
        _dict_new = {i: f'special_e{i}' for i in range(self._entity_special_num)}
        _dict_new.update({key + len(_dict_new): value for key, value in self.data.e_dict.items()})
        return _dict_new

    @property
    def processed_r_dict(self) -> Dict:
        _dict_new = {i: f'special_r{i}' for i in range(self._relation_special_num)}
        _dict_new.update({key + len(_dict_new): value for key, value in self.data.r_dict.items()})
        return _dict_new

    def show(self, logger: Logger):
        logger.info("==========Show DataHelper==========")
        logger.info("==========")
        self.data.show(logger)
        logger.info("==========")
        logger.info(f"entity_special_num: {self._entity_special_num}")
        logger.info(f"relation_special_num: {self._relation_special_num}")
        logger.info(f"r_dict: {self.processed_r_dict}")
        logger.info("==========Show DataHelper==========")


def load_preprocess_data(kg_data, eco_memory, entity_special_num=0, relation_special_num=0, *, logger=None):
    info_path, train_path, valid_path, test_path = [
        os.path.join(PROCESSED_DATA_PATH, 'KGdata', kg_data, _name)
        for _name in ('info.hdf5', 'train.hdf5', 'valid.hdf5', 'test.hdf5')
    ]
    data_helper = MyDataHelper(
        info_path, train_path, valid_path, test_path, eco_memory=eco_memory, logger=logger,
        entity_special_num=entity_special_num, relation_special_num=relation_special_num)  # 0 is special token
    return data_helper


def main():
    version_check(torch, pd, optuna)


if __name__ == '__main__':
    main()
