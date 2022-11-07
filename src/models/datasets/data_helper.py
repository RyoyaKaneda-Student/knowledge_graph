# coding: UTF-8
# region !import area!
import os
import sys
from pathlib import Path

# python
from logging import Logger
# noinspection PyUnresolvedReferences
from typing import Tuple, Optional, Union, Callable
from dataclasses import dataclass
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

# noinspection PyUnresolvedReferences
from models.datasets.datasets import (
    MyDataset, MyDatasetWithFilter, MyTripleDataset
)
from utils.setup import easy_logger
from utils.utils import (
    version_check, EternalGenerator, true_count, get_true_position_items_using_getter
)
from utils.numpy import negative_sampling

# endregion

PROJECT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(os.path.join(PROJECT_DIR, 'src'))

PROCESSED_DATA_PATH = './data/processed/'
EXTERNAL_DATA_PATH = './data/external/'


@dataclass(init=False)
class MyRawData:
    # about path
    info_path: str
    all_tail_path: str
    train_path: str
    valid_path: str
    test_path: str
    # about info
    entity_length: int
    relation_length: int
    entities: list[str]
    relations: list[str]
    is_reverse_relation: np.ndarray
    id2count_entity: np.ndarray
    id2count_relation: np.ndarray
    # about er_tails
    er_length: int
    ers: np.ndarray
    id2all_tail_entity: np.ndarray
    id2all_tail_row: np.ndarray
    id2all_tail_mode: np.ndarray
    ee_length: int
    ees: np.ndarray
    id2all_relation_relation: np.ndarray
    id2all_relation_row: np.ndarray
    id2all_relation_mode: np.ndarray
    # about triple
    train_triple: np.ndarray
    valid_triple: np.ndarray
    test_triple: np.ndarray
    # about init
    loaded_triple: bool
    loaded_all_tail: bool
    loaded_all_relation: bool

    def __init__(self,
                 info_path: str,
                 all_tail_path: str,
                 all_relation_path: str,
                 train_path: str, valid_path: str, test_path: str,
                 *, logger=None):
        with h5py.File(info_path, 'r') as f:
            self.entity_length = f['entity_length'][()]
            self.relation_length = f['relation_length'][()]
            self.entities = [e.decode() for e in f['entities'][:]]
            self.id2count_entity = f['id2count_entity'][:]

            self.relations = [r.decode() for r in f['relations'][:]]
            self.is_reverse_relation = f['id2is_reverse_relation'][:]
            self.id2count_relation = f['id2count_relation'][:]

        self.all_tail_path = all_tail_path
        self.all_relation_path = all_relation_path
        self.train_path = train_path
        self.valid_path = valid_path
        self.test_path = test_path

        self.loaded_triple = False
        self.loaded_all_tail = False
        self.loaded_all_relation = False

        self.show_log(logger)

    def init_triple(self, force_init=False) -> None:
        if self.loaded_triple and not force_init: return
        train_path, valid_path, test_path = self.train_path, self.valid_path, self.test_path
        rf = lambda _path: h5py.File(_path, 'r')
        with rf(train_path) as f_train, rf(valid_path) as f_valid, rf(test_path) as f_test:
            self.train_triple = f_train['triple'][:]
            self.valid_triple = f_valid['triple'][:]
            self.test_triple = f_test['triple'][:]

    def init_all_tail(self, force_init=False) -> None:
        if self.loaded_all_tail and not force_init: return
        all_tail_path = self.all_tail_path
        with h5py.File(all_tail_path, 'r') as f:
            self.er_length = f['er_length'][()]
            self.ers = f['ers'][:]
            self.id2all_tail_row = f['id2all_tail_row'][:]
            self.id2all_tail_entity = f['id2all_tail_entity'][:]
            self.id2all_tail_mode = f['id2all_tail_mode'][:]

    def init_all_relation(self, force_init=False) -> None:
        if self.loaded_all_relation and not force_init: return
        all_relation_path = self.all_relation_path
        with h5py.File(all_relation_path, 'r') as f:
            self.ee_length = f['ee_length'][()]
            self.ees = f['ees'][:]
            self.id2all_tail_row = f['id2all_relation_row'][:]
            self.id2all_relation_relation = f['id2all_relation_relation'][:]
            self.id2all_relation_mode = f['id2all_relation_mode_mode'][:]

    def show_log(self, logger: Logger = None):
        if logger is not None:
            logger.info("==========Show MyRawData==========")
            logger.info(f"entity_length: {self.entity_length}")
            logger.info(f"relation_length: {self.relation_length}")
            logger.info(f"er_list length: {self.er_length}")
            logger.info("==========Show MyRawData==========")

    def __getattr__(self, item):
        if item in ['train_triple', 'valid_triple', 'test_triple']:
            self.init_triple()
        elif item in ['er_length', 'ers', 'id2all_tail_row', 'id2all_tail_entity', 'id2all_tail_mode']:
            self.init_all_tail()
        elif item in [
            'ee_length', 'ees', 'id2all_relation_row', 'id2all_relation_relation', 'id2all_relation_mode_mode']:
            self.init_all_relation()
        else:
            raise "not item in Data."
        return getattr(self, item)

    def __str__(self):
        entity_length = self.entity_length
        relation_length = self.relation_length
        all_triple_length = len(self.train_triple) + len(self.valid_triple) + len(self.test_triple)
        return f"MyRawData: {entity_length=}, {relation_length=}, {all_triple_length=}"

    def del_all_optional(self):
        if self.loaded_triple:
            self.loaded_triple = False
            del self.train_triple, self.valid_triple, self.test_triple
        if self.loaded_all_tail:
            self.loaded_all_tail = False
            del self.er_length, self.ers, self.id2all_tail_row, self.id2all_tail_entity, self.id2all_tail_mode
        if self.loaded_all_relation:
            self.loaded_all_relation = False
            del self.ee_length, self.ees, self.id2all_tail_row, self.id2all_relation_relation, self.id2all_relation_mode


@dataclass(init=False)
class MyDataHelper:
    _data: MyRawData
    _entity_special_num: int
    _relation_special_num: int

    _processed_train_triple: Optional[np.ndarray]
    _processed_valid_triple: Optional[np.ndarray]
    _processed_test_triple: Optional[np.ndarray]

    _train_dataloader: Optional[DataLoader]
    _valid_dataloader: Optional[DataLoader]
    _test_dataloader: Optional[DataLoader]

    def __init__(self, info_path, all_tail_path, train_path, valid_path, test_path, *,
                 logger=None, entity_special_num, relation_special_num):
        super().__init__()
        self._data: MyRawData = MyRawData(
            info_path, all_tail_path, '', train_path, valid_path, test_path, logger=None)
        self._entity_special_num: int = entity_special_num
        self._relation_special_num: int = relation_special_num

        self._processed_train_triple = None
        self._processed_valid_triple = None
        self._processed_test_triple = None

        self._train_dataloader = None
        self._valid_dataloader = None
        self._test_dataloader = None
        self._train_valid_dataloader = None

        self.show(logger)

    def _processed_triple(self, triple) -> np.ndarray:
        e_special_num, r_special_num = self.get_er_special_num()
        triple = triple + np.array([e_special_num, r_special_num, e_special_num])
        return triple

    def get_er_special_num(self) -> Tuple[int, int]:
        return self._entity_special_num, self._relation_special_num

    def set_loaders(self,
                    train_dataloader: DataLoader, train_valid_dataloader: DataLoader,
                    valid_dataloader: DataLoader, test_dataloader: DataLoader):
        self._train_dataloader = train_dataloader
        self._train_valid_dataloader = train_valid_dataloader
        self._valid_dataloader = valid_dataloader
        self._test_dataloader = test_dataloader

    @property
    def train_dataloader(self) -> DataLoader:
        _loader = self._train_dataloader
        assert _loader is not None
        return _loader

    @property
    def train_valid_dataloader(self) -> DataLoader:
        _loader = self._train_valid_dataloader
        assert _loader is not None
        return _loader

    @property
    def valid_dataloader(self) -> DataLoader:
        _loader = self._valid_dataloader
        assert _loader is not None
        return _loader

    @property
    def test_dataloader(self) -> DataLoader:
        _loader = self._test_dataloader
        assert _loader is not None
        return _loader

    def get_dataloader(self, is_train=False, is_train_valid=False, is_valid=False, is_test=False):
        assert true_count(is_train, is_train_valid, is_valid, is_test) == 1
        rev = get_true_position_items_using_getter(
            [(lambda: getattr(self, loader_name)) for loader_name
             in ['train_dataloader', 'train_valid_dataloader', 'valid_dataloader', 'test_dataloader']],
            [is_train, is_train_valid, is_valid, is_test]
        )[0]
        return rev

    @property
    def data(self) -> MyRawData:
        return self._data

    @property
    def processed_entity_length(self) -> int:
        return self.data.entity_length + self._entity_special_num

    @property
    def processed_relation_length(self) -> int:
        return self.data.relation_length + self._relation_special_num

    @property
    def processed_r_length_without_reverse(self) -> int:
        assert self.data.relation_length % 2 == 0
        return self.data.relation_length // 2 + self._relation_special_num

    @property
    def processed_ers(self) -> np.ndarray:
        entity_special_num, relation_special_num = self.get_er_special_num()
        return self._data.ers + np.array([entity_special_num, relation_special_num])

    @property
    def processed_er2id(self) -> dict[tuple[int, int], int]:
        dict_ = {}
        # noinspection PyTypeChecker
        for i, (e, r) in enumerate(self.processed_ers.tolist()):
            dict_[(e, r)] = i
        return dict_

    @property
    def sparse_all_tail_data(self) -> torch.sparse.Tensor:
        e_length, r_length = self.processed_entity_length, self.processed_relation_length
        e_special_num, r_special_num = self.get_er_special_num()

        my_data = self.data
        er_length = self.data.er_length
        er_list: np.ndarray = np.copy(my_data.ers)
        er_list += np.array([e_special_num, r_special_num], dtype=np.int8)

        id2all_tail_row: np.ndarray = np.copy(my_data.id2all_tail_row)
        id2all_tail_entity: np.ndarray = np.copy(my_data.id2all_tail_entity) + e_special_num
        id2all_tail_mode: np.ndarray = np.copy(my_data.id2all_tail_mode)

        tails_row_column = torch.from_numpy(np.stack([id2all_tail_row, id2all_tail_entity]))
        tails_value = torch.from_numpy(id2all_tail_mode)

        rev = torch.sparse_coo_tensor(
            tails_row_column, tails_value, size=(er_length, e_length), dtype=torch.int8, requires_grad=False)
        return rev

    @property
    def processed_train_triple(self) -> np.ndarray:
        triple = self._processed_train_triple
        triple = triple if triple is not None else self._processed_triple(self.data.train_triple)
        return triple

    @property
    def processed_valid_triple(self) -> np.ndarray:
        triple = self._processed_valid_triple
        triple = triple if triple is not None else self._processed_triple(self.data.valid_triple)
        return triple

    @property
    def processed_test_triple(self) -> np.ndarray:
        triple = self._processed_test_triple
        triple = triple if triple is not None else self._processed_triple(self.data.test_triple)
        return triple

    @property
    def processed_entities(self) -> list:
        entity_special_num, _ = self.get_er_special_num()
        return [f'special_e_{i}' for i in range(entity_special_num)] + self.data.entities

    @property
    def processed_id2count_entity(self):
        entity_special_num, _ = self.get_er_special_num()
        id2count_entity = self.data.id2count_entity
        return np.concatenate([np.zeros(entity_special_num), id2count_entity])

    @property
    def processed_relations(self) -> list:
        _, relation_special_num = self.get_er_special_num()
        return [f'special_r_{i}' for i in range(relation_special_num)] + self.data.relations

    @property
    def processed_id2count_relation(self):
        _, relation_special_num = self.get_er_special_num()
        id2count_relation = self.data.id2count_relation
        return np.concatenate([np.zeros(relation_special_num), id2count_relation])

    @property
    def processed_r_is_reverse_list(self):
        _, r_special_num = self.get_er_special_num()
        is_reverse_relation = self.data.is_reverse_relation
        is_reverse_relation = np.concatenate([np.zeros(r_special_num, dtype=np.bool), is_reverse_relation])
        return is_reverse_relation

    def show(self, logger: Logger = None):
        if logger is not None:
            logger.info("==========Show DataHelper==========")
            logger.info("==========")
            self.data.show_log(logger)
            logger.info("==========")
            logger.info(f"entity_special_num: {self._entity_special_num}")
            logger.info(f"relation_special_num: {self._relation_special_num}")
            logger.info("==========Show DataHelper==========")

    def __str__(self):
        entity_length = self.processed_entity_length
        relation_length = self.processed_relation_length
        all_triple_length = len(self.data.train_triple) + len(self.data.valid_triple) + len(self.data.test_triple)
        e_special_num, r_special_num = self.get_er_special_num()
        return (
            f"MyDataHelper: {entity_length=}, {relation_length=}, {all_triple_length=}, "
            f"{e_special_num=}, {r_special_num=} "
        )


def add_negative_to_triple(
        triple_dataset: MyTripleDataset,
        sparse_all_data: torch.Tensor,
        count_per_item: Union[np.ndarray],
        negative_count):
    raise "aaa"
    old_len_ = len(triple_dataset)

    eg = EternalGenerator(
        lambda: negative_sampling(np.arange(len(count_per_item)), count_per_item, size=100000)
    )
    negative_tail = [
        eg.get_next(conditional=lambda x: (sparse_all_data[er_or_ee2id][x] == 0))
        for _, _, _, _, _, er_or_ee2id in triple_dataset
        for _ in range(negative_count)
    ]

    triple = triple_dataset.triple.repeat(old_len_ * (negative_count + 1), 1)  # n*3=> ((nc+1)*n)*3
    triple[old_len_:, 2] = torch.from_numpy(np.stack(negative_tail))

    return MyTripleDataset(
        triple,
        triple_dataset.r_is_reverse.repeat(old_len_ * (negative_count + 1)),
        triple_dataset.index2er_or_ee_id.repeat(old_len_ * (negative_count + 1)),
        triple_dataset.is_exist.repeat(old_len_ * (negative_count + 1)),
    )


def load_preprocess_data(kg_data, entity_special_num, relation_special_num, *, logger=None):
    info_path, all_tail, train_path, valid_path, test_path = [
        os.path.join(PROCESSED_DATA_PATH, 'KGdata', kg_data, _name)
        for _name in ('info.hdf5', 'all_tail.hdf5', 'train.hdf5', 'valid.hdf5', 'test.hdf5')
    ]
    data_helper = MyDataHelper(
        info_path, all_tail, train_path, valid_path, test_path,
        entity_special_num=entity_special_num, relation_special_num=relation_special_num,
        logger=logger
    )
    return data_helper


def main():
    version_check(torch, pd, optuna)
    logger = easy_logger(console_level='debug')
    logger.debug(f"{PROJECT_DIR=}")
    kg_data = 'WN18RR'
    info_path, all_tail, train_path, valid_path, test_path = [
        os.path.join(PROCESSED_DATA_PATH, 'KGdata', kg_data, _name)
        for _name in ('info.hdf5', 'all_tail.hdf5', 'train.hdf5', 'valid.hdf5', 'test.hdf5')
    ]

    logger.info(f"{info_path=}, {all_tail=}, {train_path=}, {valid_path=}, {test_path=}")

    dh0 = load_preprocess_data(kg_data, 0, 0, logger=logger)
    dh1 = load_preprocess_data(kg_data, 1, 0, logger=logger)
    dh2 = load_preprocess_data(kg_data, 0, 1, logger=logger)
    dh3 = load_preprocess_data(kg_data, 1, 1, logger=logger)

    logger.debug(dh0)
    logger.debug(dh1)
    logger.debug(dh2)
    logger.debug(dh3)

    tmp1 = dh0.processed_ers
    tmp1 += np.array([1, 0])
    tmp2 = dh1.processed_ers
    assert np.array_equal(tmp1, tmp2)
    del tmp1, tmp2


if __name__ == '__main__':
    main()
