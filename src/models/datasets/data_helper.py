# coding: UTF-8
import os
import sys
from pathlib import Path

import optuna

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(os.path.join(PROJECT_DIR, 'src'))

# python
from logging import Logger
from typing import List, Dict, Tuple, Optional, Union  # Callable
import dataclasses
# from tqdm import tqdm
from argparse import Namespace
# Machine learning
import h5py
import numpy as np
import pandas as pd
# torch
import torch
from torch.utils.data import Dataset
from torch import nn
from torch.utils.data.dataloader import DataLoader

"""
# torch ignite
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.contrib.handlers.tensorboard_logger import *
"""

from models.datasets.datasets import (
    MyDatasetMoreEcoMemory, MyDatasetMoreEcoWithFilter, MyTripleDataset
)

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

        er_list: np.ndarray = my_data.er_list
        er_tails_data: np.ndarray = my_data.er_tails_data
        er_tails_row: np.ndarray = my_data.er_tails_row
        er_tails_data_type: np.ndarray = my_data.er_tails_data_type

        er_tails: List[List[Tuple[int, int]]] = [[] for _ in er_list]
        for row, data, data_type in zip(er_tails_row, er_tails_data, er_tails_data_type):
            row, data, data_type = row.item(), data.item(), data_type.item()
            er_tails[row].append((data, data_type))

        label, label_sparce = None, None
        if not eco_memory:
            raise "this mode not supported"
            """
            label = np.zeros((len(er_list), e_length), dtype=np.int8)
            for index, tails in enumerate(er_tails):
                tails_data, tails_data_type = zip(*tails)
                np.put(label[index], tails_data, tails_data_type)
                del tails
            """
        else:
            label_sparce = [zip(*tails) for tails in er_tails]
            label_sparce = [[np.array(data), np.array(data_type, dtype=np.int8)] for data, data_type in label_sparce]

        self._data: MyTrainTestData = my_data
        self._er_list: np.ndarray = er_list
        self._label: Optional[np.ndarray] = label
        self._label_sparce: Optional[List[Tuple[np.ndarray, np.ndarray]]] = label_sparce
        self._eco_memory: bool = eco_memory
        self._entity_special_num: int = entity_special_num
        self._relation_special_num: int = relation_special_num
        # dataset
        """
        self._train_dataset: Dataset = None
        self._valid_dataset: Dataset = None
        self._test_dataset: DataLoader = None
        """
        # dataloader
        self._train_dataloader: Optional[DataLoader] = None
        self._valid_dataloader: Optional[DataLoader] = None
        self._test_dataloader: Optional[DataLoader] = None

    def get_train_dataset(self, eco_memory: bool = False, more_eco_memory: bool = False) -> MyDatasetMoreEcoMemory:
        er_list, target_num = self.er_list, 1
        entity_special_num, relation_special_num = self.get_er_special_num()
        if eco_memory and more_eco_memory:
            raise "you can't select eco and more_eco"
        elif more_eco_memory:
            label_sparce, len_e = self.label_sparce, self.data.e_length
            return MyDatasetMoreEcoMemory(
                er_list, label_sparce, len_e=len_e, target_num=target_num,
                entity_special_num=entity_special_num, relation_special_num=relation_special_num)
        else:
            raise "this mode not support."
        """
        elif eco_memory:
            return MyDatasetEcoMemory(self.er_list, self.label, target_num=target_num,
                                      entity_special_num=self._entity_special_num,
                                      relation_special_num=self._relation_special_num)
        else:
            return MyDataset(self.er_list, self.label, target_num=target_num,
                             entity_special_num=self._entity_special_num,
                             relation_special_num=self._relation_special_num)
        """

    def _get_valid_test_dataset(self, more_eco_memory: bool, target_num) -> MyDatasetMoreEcoWithFilter:
        er_list = self.er_list
        entity_special_num, relation_special_num = self.get_er_special_num()
        if more_eco_memory:
            return MyDatasetMoreEcoWithFilter(
                er_list, self._label_sparce, len_e=self.data.e_length, target_num=target_num, del_if_no_tail=True,
                entity_special_num=entity_special_num, relation_special_num=relation_special_num)
        else:
            raise "this mode not support."

    def get_valid_dataset(self, more_eco_memory: bool = False) -> MyDatasetMoreEcoWithFilter:
        return self._get_valid_test_dataset(more_eco_memory, 2)

    def get_test_dataset(self, more_eco_memory: bool = False) -> MyDatasetMoreEcoWithFilter:
        return self._get_valid_test_dataset(more_eco_memory, 3)

    def get_tvt_triple_dataset(self, triple):
        r_is_reverse_list, er2index = self.data.r_is_reverse_list, self.data.er2index
        entity_special_num, relation_special_num = self.get_er_special_num()
        return MyTripleDataset(triple, r_is_reverse_list, er2index,
                               entity_special_num=entity_special_num, relation_special_num=relation_special_num)

    def get_train_triple_dataset(self) -> MyTripleDataset:
        return self.get_tvt_triple_dataset(self.data.train_triple)

    def get_valid_triple_dataset(self) -> MyTripleDataset:
        return self.get_tvt_triple_dataset(self.data.valid_triple)

    def get_test_triple_dataset(self) -> MyTripleDataset:
        return self.get_tvt_triple_dataset(self.data.test_triple)

    def del_loaders(self):
        del self._train_dataloader, self._valid_dataloader, self._test_dataloader

    def del_test_dataloader(self):
        del self._test_dataloader

    def set_loaders(self, train_dataloader: DataLoader, valid_dataloader: DataLoader, test_dataloader: DataLoader):
        self._train_dataloader = train_dataloader
        self._valid_dataloader = valid_dataloader
        self._test_dataloader = test_dataloader

    def get_er_special_num(self) -> Tuple[int, int]:
        return self._entity_special_num, self._relation_special_num

    def get_final_e_length(self):
        return self.data.e_length + self._entity_special_num

    def get_final_r_length(self):
        return self.data.r_length + self._relation_special_num

    @property
    def e_dict(self):
        _dict_new = {i: f'special_e{i}' for i in range(self._entity_special_num)}
        _dict_new.update({key + len(_dict_new): value for key, value in self.data.e_dict.items()})
        return _dict_new

    @property
    def r_dict(self):
        _dict_new = {i: f'special_r{i}' for i in range(self._relation_special_num)}
        _dict_new.update({key+len(_dict_new): value for key, value in self.data.r_dict.items()})
        return _dict_new

    @property
    def data(self) -> MyTrainTestData:
        return self._data

    @property
    def er_list(self) -> np.ndarray:
        return self._er_list

    @property
    def label(self) -> np.ndarray:
        if self._eco_memory:
            raise "eco memory mode but select label"
        return self._label

    @property
    def label_sparce(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        if not self._eco_memory:
            raise "not eco memory mode but select label sparce"
        return self._label_sparce

    @property
    def train_dataloader(self) -> DataLoader:
        return self._train_dataloader

    @property
    def valid_dataloader(self) -> DataLoader:
        _valid_dataloader = self._valid_dataloader
        if _valid_dataloader is not None:
            return _valid_dataloader
        else:
            raise "valid_dataloader is not Defined"

    @property
    def test_dataloader(self) -> DataLoader:
        _test_dataloader = self._test_dataloader
        if _test_dataloader is not None:
            return _test_dataloader
        else:
            raise "_test_dataloader is not defined"

    def show(self, logger: Logger):
        logger.info("==========Show DataHelper==========")
        logger.info("==========")
        self.data.show(logger)
        logger.info("==========")
        logger.info(f"entity_special_num: {self._entity_special_num}")
        logger.info(f"relation_special_num: {self._relation_special_num}")
        logger.info(f"r_dict: {self.r_dict}")
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
