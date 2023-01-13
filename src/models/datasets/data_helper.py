# coding: UTF-8
# region !import area!
from dataclasses import dataclass
import os
# python
from logging import Logger
from pathlib import Path
# noinspection PyUnresolvedReferences
from typing import Tuple, Optional, Union, Callable, Final, Literal, get_args, Iterable

# Machine learning
import h5py
import numpy as np
import optuna
import pandas as pd
# torch
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
# my utils
from utils.error import UnderDevelopmentError
from utils.hdf5 import read_one_item
from utils.setup import easy_logger
from utils.typing import ConstMeta
from utils.utils import version_check

# endregion

PROJECT_DIR = Path(__file__).resolve().parents[3]

PROCESSED_DATA_PATH = './data/processed/'
EXTERNAL_DATA_PATH = './data/external/'

KGDATA: Final = 'KGdata'
KGCDATA: Final = 'KGCdata'
KGDATA_LITERAL: Final = Literal['FB15k-237', 'WN18RR', 'YAGO3-10', 'KGC-ALL', 'KGC-ALL-SVO']
KGDATA_ALL: Final = ('FB15k-237', 'WN18RR', 'YAGO3-10', 'KGC-ALL', 'KGC-ALL-SVO')

KGDATA2FOLDER_PATH = {
    'FB15k-237': os.path.join(PROCESSED_DATA_PATH, KGDATA, 'FB15k-237'),
    'WN18RR': os.path.join(PROCESSED_DATA_PATH, KGDATA, 'WN18RR'),
    'YAGO3-10': os.path.join(PROCESSED_DATA_PATH, KGDATA, 'YAGO3-10'),
    'KGC-ALL-SVO': os.path.join(PROCESSED_DATA_PATH, KGCDATA, 'All', 'SVO'),
}

ENTITY: Final = 'entity'
RELATION: Final = 'relation'
TAIL: Final = 'tail'


class INFO_INDEX(metaclass=ConstMeta):
    TRIPLE: Final = 'triple'
    ENTITY_NUM: Final = 'entity_length'
    RELATION_NUM: Final = 'relation_length'
    ENTITIES: Final = 'entities'
    ENTITIES_LABEL: Final = 'entities_label'
    ID2COUNT_ENTITY: Final = 'id2count_entity'
    RELATIONS: Final = 'relations'
    RELATIONS_LABEL: Final = 'relations_label'
    IS_REV_RELATION: Final = 'id2is_reverse_relation'
    ID2COUNT_RELATION: Final = 'id2count_relation'

    @classmethod
    def all_index(cls):
        return [
            cls.TRIPLE, cls.ENTITY_NUM, cls.RELATION_NUM, cls.ENTITY_NUM, cls.ENTITIES, cls.ENTITIES_LABEL,
            cls.ID2COUNT_ENTITY, cls.RELATIONS, cls.RELATIONS_LABEL, cls.IS_REV_RELATION, cls.ID2COUNT_RELATION
        ]


# about tokens
class DefaultTokens(metaclass=ConstMeta):
    PAD_E: Final[str] = '<pad_e>'
    CLS_E: Final[str] = '<cls_e>'
    MASK_E: Final[str] = '<mask_e>'
    SEP_E: Final[str] = '<sep_e>'
    BOS_E: Final[str] = '<bos_e>'
    PAD_R: Final[str] = '<pad_r>'
    CLS_R: Final[str] = '<cls_r>'
    MASK_R: Final[str] = '<mask_r>'
    SEP_R: Final[str] = '<sep_r>'
    BOS_R: Final[str] = '<bos_r>'


class DefaultIds(metaclass=ConstMeta):
    PAD_E_DEFAULT_ID: Final[int] = 0
    CLS_E_DEFAULT_ID: Final[int] = 1
    MASK_E_DEFAULT_ID: Final[int] = 2
    SEP_E_DEFAULT_ID: Final[int] = 3
    BOS_E_DEFAULT_ID: Final[int] = 4
    PAD_R_DEFAULT_ID: Final[int] = 0
    CLS_R_DEFAULT_ID: Final[int] = 1
    MASK_R_DEFAULT_ID: Final[int] = 2
    SEP_R_DEFAULT_ID: Final[int] = 3
    BOS_R_DEFAULT_ID: Final[int] = 4


class DefaultTokenIds:
    @staticmethod
    def default_token2ids_e():
        DT = DefaultTokens
        DI = DefaultIds
        return {DT.PAD_E: DI.PAD_E_DEFAULT_ID, DT.CLS_E: DI.CLS_E_DEFAULT_ID,
                DT.MASK_E: DI.MASK_E_DEFAULT_ID, DT.SEP_E: DI.SEP_E_DEFAULT_ID, DT.BOS_E: DI.BOS_E_DEFAULT_ID}

    @staticmethod
    def default_token2ids_r():
        DT = DefaultTokens
        DI = DefaultIds
        return {DT.PAD_R: DI.PAD_R_DEFAULT_ID, DT.CLS_R: DI.CLS_R_DEFAULT_ID,
                DT.MASK_R: DI.MASK_R_DEFAULT_ID, DT.SEP_R: DI.SEP_R_DEFAULT_ID, DT.BOS_R: DI.BOS_R_DEFAULT_ID}

    @staticmethod
    def default_ids2token_e():
        return {value: key for key, value in DefaultTokenIds.default_token2ids_e()}

    @staticmethod
    def default_ids2token_r():
        return {value: key for key, value in DefaultTokenIds.default_token2ids_r()}


# noinspection PyTypeChecker
def make_change_index_func(_length: int, special_ids: Iterable[int]):
    tmp_list = [True for _ in range(_length + len(special_ids))]
    for id_ in special_ids: tmp_list[id_] = None
    change_list = [i for i, tmp in enumerate(tmp_list) if tmp is not None]
    assert len(change_list) == _length
    change_tuple = tuple(change_list)

    def func(x: int) -> int:
        return change_tuple[x]

    return func


@dataclass
class SpecialTokens:
    @classmethod
    def default(cls):
        return cls()


@dataclass
class SpecialPaddingTokens(SpecialTokens):
    padding_token_e: int = DefaultIds.PAD_E_DEFAULT_ID
    padding_token_r: int = DefaultIds.PAD_R_DEFAULT_ID


@dataclass
class SpecialTokens01(SpecialPaddingTokens):
    padding_token_e: int = DefaultIds.PAD_E_DEFAULT_ID
    padding_token_r: int = DefaultIds.PAD_R_DEFAULT_ID
    cls_token_e: int = DefaultIds.CLS_E_DEFAULT_ID
    cls_token_r: int = DefaultIds.CLS_R_DEFAULT_ID
    mask_token_e: int = DefaultIds.MASK_E_DEFAULT_ID
    mask_token_r: int = DefaultIds.MASK_R_DEFAULT_ID
    sep_token_e: int = DefaultIds.SEP_E_DEFAULT_ID
    sep_token_r: int = DefaultIds.SEP_R_DEFAULT_ID
    bos_token_e: int = DefaultIds.BOS_E_DEFAULT_ID
    bos_token_r: int = DefaultIds.BOS_R_DEFAULT_ID


@dataclass(init=False)
class MyRawData:
    # about path
    info_path: str
    train_path: str
    valid_path: str
    test_path: str
    # about info
    entity_num: int
    relation_num: int
    entities: list[str]
    relations: list[str]
    entities_label: Optional[list[str]]
    relations_label: Optional[list[str]]
    is_reverse_relation: np.ndarray
    entityIdx2countFrequency: np.ndarray
    relationIdx2countFrequency: np.ndarray
    # about triple
    train_triple: np.ndarray
    valid_triple: np.ndarray
    test_triple: np.ndarray
    # about init
    loaded_triple: bool

    def __init__(
            self, info_path, train_path, valid_path, test_path, *, logger=None):
        """

        Args:
            info_path(str): path for "info.hdf5"
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
            if train_path is None and logger is not None:
                logger.warning("no train_path")
            if valid_path is None and logger is not None:
                logger.debug("no valid_path")
            if test_path is None and logger is not None:
                logger.debug("no test_path")
        # load info.hdf5
        with h5py.File(info_path, 'r') as f:
            self.entity_num = f[INFO_INDEX.ENTITY_NUM][()]
            self.relation_num = f[INFO_INDEX.RELATION_NUM][()]
            self.entities = [e.decode() for e in f[INFO_INDEX.ENTITIES][:]]
            self.entityIdx2countFrequency = f[INFO_INDEX.ID2COUNT_ENTITY][:]
            self.entities_label = [e.decode() for e in f[INFO_INDEX.ENTITIES_LABEL][:]]

            self.relations = [r.decode() for r in f[INFO_INDEX.RELATIONS][:]]
            self.is_reverse_relation = f[INFO_INDEX.IS_REV_RELATION][:]
            self.relationIdx2countFrequency = f[INFO_INDEX.ID2COUNT_RELATION][:]
            self.relations_label = [r.decode() for r in f[INFO_INDEX.RELATIONS_LABEL][:]]

        # setting paths
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
        _func = lambda f: f[INFO_INDEX.TRIPLE][:]
        self.train_triple, self.valid_triple, self.test_triple = [
            read_one_item(_path, _func) if _path is not None else None
            for _path in (self.train_path, self.valid_path, self.test_path)]

    def show_log(self, logger: Logger = None):
        if logger is not None:
            logger.info("====== Show MyRawData ======")
            logger.info(f"entity num: {self.entity_num}")
            logger.info(f"relation num: {self.relation_num}")
            logger.info("====== Show MyRawData ======")

    def __getattr__(self, item):
        if item in ('train_triple', 'valid_triple', 'test_triple'):
            self.init_triple()
        else:
            raise "not item in Data."
        return getattr(self, item)

    def __str__(self):
        entity_num = self.entity_num
        relation_num = self.relation_num
        return f"MyRawData: entity num={entity_num}, relation num={relation_num}"

    def del_all_optional(self):
        if self.loaded_triple:
            self.loaded_triple = False
            del self.train_triple, self.valid_triple, self.test_triple


@dataclass(init=False)
class MyDataHelper:
    _data: MyRawData
    _entity_special_dicts: dict[int, str]
    _relation_special_dicts: dict[int, str]

    _processed_train_triple: Optional[np.ndarray]
    _processed_valid_triple: Optional[np.ndarray]
    _processed_test_triple: Optional[np.ndarray]

    def __init__(self, info_path, train_path, valid_path, test_path, *,
                 logger: Logger = None, entity_special_dicts: dict[int, str], relation_special_dicts: dict[int, str]):
        """

        Args:
            info_path:
            train_path:
            valid_path:
            test_path:
            logger:
            entity_special_dicts:
            relation_special_dicts:
        """
        super().__init__()
        self._data: MyRawData = MyRawData(info_path, train_path, valid_path, test_path, logger=logger)
        self._entity_special_dicts: dict[int, str] = entity_special_dicts
        self._relation_special_dicts: dict[int, str] = relation_special_dicts

        self._processed_train_triple = None
        self._processed_valid_triple = None
        self._processed_test_triple = None

    def _special_token_change(self, raw_array: np.ndarray, entity_index, relation_index):
        entity_num, relation_length = self.data.entity_num, self.data.relation_num
        entity_special_ids, relation_special_ids = self.entity_special_ids, self.relation_special_ids
        _change_entity_func = np.frompyfunc(make_change_index_func(entity_num, entity_special_ids), 1, 1)
        _change_relation_func = np.frompyfunc(make_change_index_func(relation_length, relation_special_ids), 1, 1)

        processed_array = np.copy(raw_array)
        processed_array[:, entity_index] = _change_entity_func(processed_array[:, entity_index])
        processed_array[:, relation_index] = _change_relation_func(processed_array[:, relation_index])
        return processed_array

    def _processed_triple(self, triple) -> np.ndarray:
        return self._special_token_change(triple, entity_index=(0, 2), relation_index=(1,))

    @property
    def data(self) -> MyRawData:
        return self._data

    @property
    def entity_special_ids(self) -> tuple[int, ...]:
        return tuple(self._entity_special_dicts.keys())

    @property
    def relation_special_ids(self) -> tuple[int, ...]:
        return tuple(self._relation_special_dicts.keys())

    @property
    def processed_entity_num(self) -> int:
        return self.data.entity_num + len(self._entity_special_dicts)

    @property
    def processed_relation_num(self) -> int:
        return self.data.relation_num + len(self._relation_special_dicts)

    @property
    def processed_train_triple(self) -> np.ndarray:
        if self._processed_train_triple is None:
            self._processed_train_triple = self._processed_triple(self.data.train_triple)
        return self._processed_train_triple

    @property
    def processed_valid_triple(self) -> np.ndarray:
        if self._processed_valid_triple is None:
            self._processed_valid_triple = self._processed_triple(self.data.valid_triple)
        return self._processed_valid_triple

    @property
    def processed_test_triple(self) -> np.ndarray:
        if self._processed_test_triple is None:
            self._processed_test_triple = self._processed_triple(self.data.test_triple)
        return self._processed_test_triple

    @property
    def processed_entities(self) -> list:
        entities = self.data.entities[:]
        for index, value in sorted(self._entity_special_dicts.items(), key=lambda x: x[0]):
            entities.insert(index, value)
        return entities

    @property
    def processed_entities_label(self) -> list:
        entities_label = self.data.entities_label[:]
        for index, value in sorted(self._entity_special_dicts.items(), key=lambda x: x[0]):
            entities_label.insert(index, value)
        return entities_label

    @property
    def processed_entityIdx2countFrequency(self) -> np.ndarray:
        idx2count = self.data.entityIdx2countFrequency[:]
        indexes, _ = zip(*[*(sorted(self._entity_special_dicts.items(), key=lambda x: x[0]))])
        return np.insert(idx2count, indexes, [-1] * len(indexes))

    @property
    def processed_relations(self) -> list:
        relations = self.data.relations[:]
        for index, value in sorted(self._relation_special_dicts.items(), key=lambda x: x[0]):
            relations.insert(index, value)
        return relations

    @property
    def processed_relations_label(self) -> list:
        relations_label = self.data.relations_label[:]
        for index, value in sorted(self._relation_special_dicts.items(), key=lambda x: x[0]):
            relations_label.insert(index, value)
        return relations_label

    @property
    def processed_relationIdx2countFrequency(self) -> np.ndarray:
        id2count = self.data.relationIdx2countFrequency[:]
        indexes, _ = zip(*[*(sorted(self._relation_special_dicts.items(), key=lambda x: x[0]))])
        return np.insert(id2count, indexes, [-1] * len(indexes))

    def show(self, logger: Logger):
        """

        Args:
            logger(Logger):

        Raises:
            AttributeError:

        Returns:

        """

        logger.info("========== Show DataHelper ==========")
        self.data.show_log(logger)
        logger.info(f"entity_special_dicts: {self._entity_special_dicts}")
        logger.info(f"relation_special_dicts: {self._relation_special_dicts}")
        logger.info(f"processed entity num: {self.processed_entity_num}")
        logger.info(f"processed relation num: {self.processed_relation_num}")
        logger.info("========== Show DataHelper ==========")

    def __str__(self):
        entity_num = self.processed_entity_num
        relation_num = self.processed_relation_num
        return f"MyData: entity num={entity_num}, relation num={relation_num}"


@dataclass(init=False)
class MyDataLoaderHelper:
    """MyDataLoaderHelper

        This class is only used for dataclass.
        Is it really need?

        Attributes:
            _train_dataloader (DataLoader): train dataloader
            _valid_dataloader (DataLoader): valid dataloader
            _test_dataloader (DataLoader): test dataloader
            _train_valid_dataloader (DataLoader): valid dataloader using train data

    """

    datasets: tuple[Dataset, Dataset, Dataset]
    _train_dataloader: Optional[DataLoader]
    _valid_dataloader: Optional[DataLoader]
    _test_dataloader: Optional[DataLoader]
    _train_valid_dataloader: Optional[DataLoader]

    def __init__(self, datasets, train_dataloader, train_valid_dataloader, valid_dataloader, test_dataloader):
        """

        Args:
            datasets(tuple[Dataset, Dataset, Dataset]):
            train_dataloader (DataLoader):
            train_valid_dataloader (DataLoader or None):
            valid_dataloader (DataLoader or None):
            test_dataloader (DataLoader or None):

        Returns:

        """
        self.datasets = datasets
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


"""
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
"""


def main():
    version_check(torch, pd, optuna)
    logger = easy_logger(console_level='debug')
    logger.debug(f"{PROJECT_DIR=}")
    pass


if __name__ == '__main__':
    main()
    pass
