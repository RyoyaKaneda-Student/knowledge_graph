# coding: UTF-8
# region !import area!
import os
import sys
from pathlib import Path

# python
from logging import Logger
# noinspection PyUnresolvedReferences
from typing import Tuple, Optional, Union, Callable, Final, Literal, get_args
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
from utils.hdf5 import read_one_item
from utils.setup import easy_logger
from utils.typing import ConstValueClass
from utils.utils import (
    version_check, true_count, get_true_position_items_using_getter
)

# endregion

PROJECT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(os.path.join(PROJECT_DIR, 'src'))

PROCESSED_DATA_PATH = './data/processed/'
EXTERNAL_DATA_PATH = './data/external/'

KGDATA: Final = 'KGdata'
KGCDATA: Final = 'KGCdata'
KGDATA_LITERAL: Final = Literal['FB15k-237', 'WN18RR', 'YAGO3-10', 'KGC-ALL', 'KGC-ALL-SVO']
KGDATA_ALL: Final = get_args(KGDATA_LITERAL)
KGDATA2FOLDER_PATH = {
    'FB15k-237': os.path.join(PROCESSED_DATA_PATH, KGDATA, 'FB15k-237'),
    'WN18RR': os.path.join(PROCESSED_DATA_PATH, KGDATA, 'WN18RR'),
    'YAGO3-10': os.path.join(PROCESSED_DATA_PATH, KGDATA, 'YAGO3-10'),
    'KGC-ALL-SVO': os.path.join(PROCESSED_DATA_PATH, KGCDATA, 'All', 'SVO'),
}


class INFO_INDEX(ConstValueClass):
    TRIPLE: Final = 'triple'
    E_LEN: Final = 'entity_length'
    R_LEN: Final = 'relation_length'
    ENTITIES: Final = 'entities'
    ID2COUNT_ENTITY: Final = 'id2count_entity'
    RELATIONS: Final = 'relations'
    IS_REV_RELATION: Final = 'id2is_reverse_relation'
    ID2COUNT_RELATION: Final = 'id2count_relation'

    @classmethod
    def all_index(cls):
        return [
            cls.TRIPLE, cls.E_LEN, cls.R_LEN, cls.E_LEN, cls.ENTITIES,
            cls.ID2COUNT_ENTITY, cls.RELATIONS, cls.IS_REV_RELATION, cls.ID2COUNT_RELATION
        ]


class ALL_TAIL_INDEX(ConstValueClass):
    ER_LENGTH: Final = 'er_length'
    ERS: Final = 'ers'
    ID2ALL_TAIL_ROW: Final = 'id2all_tail_row'
    ID2ALL_TAIL_ENTITY: Final = 'id2all_tail_entity'
    ID2ALL_TAIL_MODE: Final = 'id2all_tail_mode'

    @classmethod
    def all_index(cls):
        return [cls.ER_LENGTH, ALL_TAIL_INDEX.ERS,
                ALL_TAIL_INDEX.ID2ALL_TAIL_ROW, ALL_TAIL_INDEX.ID2ALL_TAIL_ENTITY, ALL_TAIL_INDEX.ID2ALL_TAIL_MODE]


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

    def __init__(
            self, info_path, all_tail_path, all_relation_path, train_path, valid_path, test_path, *, logger=None):
        """

        Args:
            info_path(str): path for "info.hdf5"
            all_tail_path(str):
            all_relation_path(str):
            train_path(str):
            valid_path(str):
            test_path(str):
            logger(Logger):
        """
        # error or warning
        if info_path is None:
            raise "info_path must not None."
            pass
        else:
            if all_tail_path is None and logger is not None:
                logger.debug("no all_tail_path")
            if all_relation_path is None and logger is not None:
                logger.debug("no all_relation_path")
            if train_path is None and logger is not None:
                logger.warning("no train_path")
            if valid_path is None and logger is not None:
                logger.debug("no valid_path")
            if test_path is None and logger is not None:
                logger.debug("no test_path")
        # load info.hdf5
        with h5py.File(info_path, 'r') as f:
            self.entity_length = f[INFO_INDEX.E_LEN][()]
            self.relation_length = f[INFO_INDEX.R_LEN][()]
            self.entities = [e.decode() for e in f[INFO_INDEX.ENTITIES][:]]
            self.id2count_entity = f[INFO_INDEX.ID2COUNT_ENTITY][:]

            self.relations = [r.decode() for r in f[INFO_INDEX.RELATIONS][:]]
            self.is_reverse_relation = f[INFO_INDEX.IS_REV_RELATION][:]
            self.id2count_relation = f[INFO_INDEX.ID2COUNT_RELATION][:]
        # setting paths
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
        self.loaded_triple = True
        self.train_triple, self.valid_triple, self.test_triple = [
            read_one_item(_path, lambda f: f[INFO_INDEX.TRIPLE][:]) if _path is not None else None
            for _path in (self.train_path, self.valid_path, self.test_path)
        ]

    def init_all_tail(self, force_init=False) -> None:
        if self.loaded_all_tail and not force_init: return
        self.loaded_all_tail = True
        all_tail_path = self.all_tail_path
        with h5py.File(all_tail_path, 'r') as f:
            self.er_length = f[ALL_TAIL_INDEX.ER_LENGTH][()]
            self.ers = f[ALL_TAIL_INDEX.ERS][:]
            self.id2all_tail_row = f[ALL_TAIL_INDEX.ID2ALL_TAIL_ROW][:]
            self.id2all_tail_entity = f[ALL_TAIL_INDEX.ID2ALL_TAIL_ENTITY][:]
            self.id2all_tail_mode = f[ALL_TAIL_INDEX.ID2ALL_TAIL_MODE][:]

    def init_all_relation(self, force_init=False) -> None:
        if self.loaded_all_relation and not force_init: return
        self.loaded_all_relation = True
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
            if self.loaded_all_tail:
                logger.info(f"er_list length: {self.er_length}")
            logger.info("==========Show MyRawData==========")

    def __getattr__(self, item):
        if item in ['train_triple', 'valid_triple', 'test_triple']:
            self.init_triple()
        elif item in ['er_length', 'ers', 'id2all_tail_row', 'id2all_tail_entity', 'id2all_tail_mode']:
            self.init_all_tail()
        elif item in [
            'ee_length', 'ees', 'id2all_relation_row', 'id2all_relation_relation', 'id2all_relation_mode_mode'
        ]:
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
    _special_entity_list: list
    _special_relation_list: list

    _processed_train_triple: Optional[np.ndarray]
    _processed_valid_triple: Optional[np.ndarray]
    _processed_test_triple: Optional[np.ndarray]

    _train_dataloader: Optional[DataLoader]
    _valid_dataloader: Optional[DataLoader]
    _test_dataloader: Optional[DataLoader]

    def __init__(self, info_path, all_tail_path, train_path, valid_path, test_path, *,
                 logger: Logger = None, entity_special_num, relation_special_num):
        super().__init__()
        self._data: MyRawData = MyRawData(
            info_path, all_tail_path, '', train_path, valid_path, test_path, logger=logger)
        self._special_entity_list: list = [f'special_e_{i}' for i in range(entity_special_num)]
        self._special_relation_list: list = [f'special_d_{i}' for i in range(relation_special_num)]

        self._processed_train_triple = None
        self._processed_valid_triple = None
        self._processed_test_triple = None

        self._train_dataloader = None
        self._valid_dataloader = None
        self._test_dataloader = None
        self._train_valid_dataloader = None

        self.show(logger)

    def set_special_names(self, index2name_entity: dict[int, str], index2name_relation: dict[int, str]):
        for i, name in index2name_entity.items():
            assert type(i) is int and type(name) is str
            self._special_entity_list[i] = name
        for i, name in index2name_relation.items():
            assert type(i) is int and type(name) is str
            self._special_relation_list[i] = name

    def _processed_triple(self, triple) -> np.ndarray:
        e_special_num, r_special_num = self.get_er_special_num()
        triple = triple + np.array([e_special_num, r_special_num, e_special_num])
        return triple

    def get_er_special_num(self) -> Tuple[int, int]:
        return len(self._special_entity_list), len(self._special_relation_list)

    def set_loaders(self,
                    train_dataloader: DataLoader, train_valid_dataloader: Optional[DataLoader],
                    valid_dataloader: Optional[DataLoader], test_dataloader: Optional[DataLoader]):
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
        return self.data.entity_length + len(self._special_entity_list)

    @property
    def processed_relation_length(self) -> int:
        return self.data.relation_length + len(self._special_relation_list)

    @property
    def processed_relation_length_without_reverse(self) -> int:
        assert self.data.relation_length % 2 == 0
        return self.data.relation_length // 2 + len(self._special_relation_list)

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
        return self._special_entity_list + self.data.entities

    @property
    def processed_id2count_entity(self):
        entity_special_num, _ = self.get_er_special_num()
        id2count_entity = self.data.id2count_entity
        return np.concatenate([np.zeros(entity_special_num), id2count_entity])

    @property
    def processed_relations(self) -> list:
        return self._special_relation_list + self.data.relations

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

    @property
    def processed_relations_no_reverse(self):
        relations = self.processed_relations
        r_is_reverse = self.processed_r_is_reverse_list
        assert len(relations) == len(r_is_reverse)
        return [r for i, r in enumerate(relations) if r_is_reverse[i] == 0]

    def show(self, logger: Logger = None):
        if logger is not None:
            logger.info("==========Show DataHelper==========")
            logger.info("==========")
            self.data.show_log(logger)
            logger.info("==========")
            entity_special_num, relation_special_num = self.get_er_special_num()
            logger.info(f"entity_special_num: {entity_special_num}")
            logger.info(f"relation_special_num: {relation_special_num}")
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


def load_preprocess_data(kg_data: KGDATA_LITERAL, entity_special_num, relation_special_num, *, logger=None):
    paths = [
        os.path.join(KGDATA2FOLDER_PATH[kg_data], _name)
        for _name in ('info.hdf5', 'all_tail.hdf5', 'train.hdf5', 'valid.hdf5', 'test.hdf5')
    ]
    info_path, all_tail, train_path, valid_path, test_path = [
        _path if os.path.exists(_path) else None for _path in paths
    ]
    logger.debug(f"{info_path=}, {all_tail=}, {train_path=}, {valid_path=}, {test_path=}")
    data_helper = MyDataHelper(
        info_path, all_tail, train_path, valid_path, test_path, logger=logger,
        entity_special_num=entity_special_num, relation_special_num=relation_special_num,
    )
    return data_helper


def main():
    version_check(torch, pd, optuna)
    logger = easy_logger(console_level='debug')
    logger.debug(f"{PROJECT_DIR=}")
    kg_data: KGDATA_LITERAL = 'WN18RR'
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
    pass
