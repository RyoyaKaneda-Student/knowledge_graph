#!/usr/bin/python
# coding: UTF-8
"""Raw data and data helper.

* This file defined raw data and data helper and dataloader helper.
raw data class is the class of read file and set the data.
data helper class is the helper class to o support special tokens.
dataloader helper class is the dataclass of dataloader.

Todo:
    * Is dataloader helper really need?
    * To improve data helper class
"""

# region !import area!
import os
from dataclasses import dataclass
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
# torch geometric
from torch_geometric.data import Data as GeometricData
# my utils
from utils.hdf5 import read_one_item
from utils.setup import easy_logger
from utils.typing import ConstMeta
from utils.utils import version_check

# endregion

PROJECT_DIR = Path(__file__).resolve().parents[3]

PROCESSED_DATA_PATH: Final = './data/processed/'
EXTERNAL_DATA_PATH: Final = './data/external/'

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
    """Info file's index const parameters.
    """
    ENTITIES: Final = 'entities'
    ENTITY_NUM: Final = 'entity_num'
    ENTITIES_LABEL: Final = 'entities_label'
    ID2COUNT_ENTITY: Final = 'id2count_entity'
    RELATIONS: Final = 'relations'
    RELATION_NUM: Final = 'relation_num'
    RELATIONS_LABEL: Final = 'relations_label'
    IS_REV_RELATION: Final = 'id2is_reverse_relation'
    ID2COUNT_RELATION: Final = 'id2count_relation'

    @classmethod
    def ALL_INDEXES(cls):
        """ALL INDEXES
        Returns:
            tuple[str]: ALL INDEXES
        """
        return (
            cls.ENTITY_NUM, cls.RELATION_NUM, cls.ENTITY_NUM, cls.ENTITIES, cls.ENTITIES_LABEL,
            cls.ID2COUNT_ENTITY, cls.RELATIONS, cls.RELATIONS_LABEL, cls.IS_REV_RELATION, cls.ID2COUNT_RELATION
        )


class TRAIN_INDEX(metaclass=ConstMeta):
    """Train file's index const parameters.
    """
    TRIPLE: Final = 'triple'
    TRIPLE_RAW: Final = 'triple_raw'

    @classmethod
    def ALL_INDEXES(cls):
        """ALL INDEXES
        Returns:
            tuple[str]: ALL INDEXES
        """
        return (
            cls.TRIPLE, cls.TRIPLE_RAW
        )


# about tokens
class DefaultTokens(metaclass=ConstMeta):
    """DefaultTokens const parameters.
    """
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
    """DefaultIds const parameters.
    """
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
    """DefaultTokenIds
    """
    @staticmethod
    def default_token2ids_e():
        """default_token to ids_e
        Returns:
            dict[str, int]: the key is token about entities and the value is token_id.
        """
        DT = DefaultTokens
        DI = DefaultIds
        return {DT.PAD_E: DI.PAD_E_DEFAULT_ID, DT.CLS_E: DI.CLS_E_DEFAULT_ID,
                DT.MASK_E: DI.MASK_E_DEFAULT_ID, DT.SEP_E: DI.SEP_E_DEFAULT_ID, DT.BOS_E: DI.BOS_E_DEFAULT_ID}

    @staticmethod
    def default_token2ids_r():
        """default_token to ids_r
        Returns:
            dict[str, int]: the key is token about relations and the value is token_id.
        """
        DT = DefaultTokens
        DI = DefaultIds
        return {DT.PAD_R: DI.PAD_R_DEFAULT_ID, DT.CLS_R: DI.CLS_R_DEFAULT_ID,
                DT.MASK_R: DI.MASK_R_DEFAULT_ID, DT.SEP_R: DI.SEP_R_DEFAULT_ID, DT.BOS_R: DI.BOS_R_DEFAULT_ID}

    @staticmethod
    def default_ids2token_e():
        """default_ids to token_e
        Returns:
            dict[str, int]: the key is token_id and the value is token about entities
        """
        return {value: key for key, value in DefaultTokenIds.default_token2ids_e()}

    @staticmethod
    def default_ids2token_r():
        """default_ids to token_e
        Returns:
            dict[str, int]: the key is token_id and the value is token about relations.
        """
        return {value: key for key, value in DefaultTokenIds.default_token2ids_r()}


# noinspection PyTypeChecker
def make_change_index_func(_length: int, special_ids: Iterable[int]):
    """make the function about change index by special tokens.

    Args:
        _length(int): the length of before list.
        special_ids(Iterable[int]): special ids.
    Returns:
        Callable[[int], int]: This is the function which get old index and return new index.
    """
    tmp_list = [True for _ in range(_length + len(special_ids))]
    for id_ in special_ids: tmp_list[id_] = None
    change_list = [i for i, tmp in enumerate(tmp_list) if tmp is not None]
    assert len(change_list) == _length
    change_tuple = tuple(change_list)

    return lambda x: change_tuple[x]


@dataclass
class SpecialTokens:
    """SpecialTokens
    """
    @classmethod
    def default(cls):
        """return itself.
        todo:
            * is really need?
        """
        return cls()


@dataclass
class SpecialPaddingTokens(SpecialTokens):
    """SpecialTokens which has padding token.
    """
    padding_token_e: int = DefaultIds.PAD_E_DEFAULT_ID
    padding_token_r: int = DefaultIds.PAD_R_DEFAULT_ID


@dataclass
class SpecialTokens01(SpecialPaddingTokens):
    """SpecialTokens which has (padding, cls, mask, sep, bos) tokens par entity and relation.
    """
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
    """Raw Data Class

    * This is the class which instance has the raw data.
    * Its means this instance has the same data as file.
    * This class has the file paths, nums, the lists, the label lists, the reverse check lists and frequency lists.
    * And also (train, valid, test) triple list as optional.
    """
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
    _loaded_triple: bool

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

        self._loaded_triple = False

        self.show_log(logger)

    def init_triple(self, force_init=False) -> None:
        """init triple

        Args:
            force_init(:obj:`bool`, optional): if True, the triple will re-init with or without init. Defaults to False.

        """
        if self._loaded_triple and not force_init: return
        self._loaded_triple = True
        _func = lambda f: f[TRAIN_INDEX.TRIPLE][:]
        self.train_triple, self.valid_triple, self.test_triple = [
            read_one_item(_path, _func) if _path is not None else None
            for _path in (self.train_path, self.valid_path, self.test_path)]

    def show_log(self, logger: Logger = None):
        """show to logger if logger is not None

        Args:
            logger(Logger): logging.logger
        """
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
        """ del all optional params.

        * del triple items for memory-friendly.
        """
        if self._loaded_triple:
            self._loaded_triple = False
            del self.train_triple, self.valid_triple, self.test_triple


@dataclass(init=False)
class MyDataHelper:
    """ Data Helper class

    * This class is the data helper class.
    * This class return the items considering special tokens.
    * So, you can use items with no considering  special tokens after this processed.
    """
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

    def _processed_triple_geometric(self, processed_triple) -> GeometricData:
        x = torch.arange(self.processed_entity_num).view(-1, 1)
        edge_index = torch.from_numpy(processed_triple[:, (0, 2)].transpose(1, 0)).clone().to(torch.long)
        edge_attr = torch.from_numpy(processed_triple[:, 1]).clone().view(-1, 1)
        data = GeometricData(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return data

    @property
    def data(self) -> MyRawData:
        """raw data.
        Returns:
            MyRawData: raw data instance.
        """
        return self._data

    @property
    def entity_special_ids(self) -> tuple[int, ...]:
        """entity special ids

        Returns:
            tuple[int, ...]: The tuple of entity special ids.

        """
        return tuple(self._entity_special_dicts.keys())

    @property
    def relation_special_ids(self) -> tuple[int, ...]:
        """relation special ids

        Returns:
            tuple[int, ...]: The tuple of relation special ids.
        """
        return tuple(self._relation_special_dicts.keys())

    @property
    def processed_entity_num(self) -> int:
        """entity's num considering special_ids

        Returns:
            int: entity's num considering special_ids.
        """
        return self.data.entity_num + len(self._entity_special_dicts)

    @property
    def processed_relation_num(self) -> int:
        """relation's num considering special_ids

        Returns:
            int: relation's num considering special_ids.
        """
        return self.data.relation_num + len(self._relation_special_dicts)

    @property
    def processed_train_triple(self) -> np.ndarray:
        """train triple considering special_ids.
        For memory-friendly, this property use cache system.
        Returns:
            np.ndarray: train triple considering special_ids.
        """
        if self._processed_train_triple is None:
            self._processed_train_triple = self._processed_triple(self.data.train_triple)
        return self._processed_train_triple

    @property
    def processed_valid_triple(self) -> np.ndarray:
        """valid triple considering special_ids.
        For memory-friendly, this property use cache system.
        Returns:
            np.ndarray: valid triple considering special_ids.
        """
        if self._processed_valid_triple is None:
            self._processed_valid_triple = self._processed_triple(self.data.valid_triple)
        return self._processed_valid_triple

    @property
    def processed_test_triple(self) -> np.ndarray:
        """test triple considering special_ids.
        For memory-friendly, this property use cache system.
        Returns:
            np.ndarray: test triple considering special_ids.
        """
        if self._processed_test_triple is None:
            self._processed_test_triple = self._processed_triple(self.data.test_triple)
        return self._processed_test_triple

    @property
    def processed_train_triple_geometric(self) -> GeometricData:
        """train torch geometric triple considering special_ids.

        Returns:
            GeometricData: train torch geometric triple considering special_ids.
        """
        return self._processed_triple_geometric(self.processed_train_triple)

    @property
    def processed_valid_triple_geometric(self) -> GeometricData:
        """valid torch geometric triple considering special_ids.

        Returns:
            GeometricData: valid torch geometric triple considering special_ids.
        """
        return self._processed_triple_geometric(self.processed_valid_triple)

    @property
    def processed_test_triple_geometric(self) -> GeometricData:
        """test torch geometric triple considering special_ids.

        Returns:
            GeometricData: test torch geometric triple considering special_ids.
        """
        return self._processed_triple_geometric(self.processed_test_triple)

    @property
    def processed_entities(self) -> list[str]:
        """entities considering special_ids.

        Returns:
            list[str]: entities considering special_ids.
        """
        entities = self.data.entities[:]
        for index, value in sorted(self._entity_special_dicts.items(), key=lambda x: x[0]):
            entities.insert(index, value)
        return entities

    @property
    def processed_entities_label(self) -> list[str]:
        """entities label considering special_ids.

        Returns:
            list[str]: entities label considering special_ids.
        """
        entities_label = self.data.entities_label[:]
        for index, value in sorted(self._entity_special_dicts.items(), key=lambda x: x[0]):
            entities_label.insert(index, value)
        return entities_label

    @property
    def processed_entityIdx2countFrequency(self) -> np.ndarray:
        """entityIdx2countFrequency considering special_ids.

        Returns:
            np.ndarray: entityIdx2countFrequency label considering special_ids.
        """
        idx2count = self.data.entityIdx2countFrequency[:]
        indexes, _ = zip(*[*(sorted(self._entity_special_dicts.items(), key=lambda x: x[0]))])
        return np.insert(idx2count, indexes, [-1] * len(indexes))

    @property
    def processed_relations(self) -> list[str]:
        """relations considering special_ids.

        Returns:
            list[str]: relations considering special_ids.
        """
        relations = self.data.relations[:]
        for index, value in sorted(self._relation_special_dicts.items(), key=lambda x: x[0]):
            relations.insert(index, value)
        return relations

    @property
    def processed_relations_label(self) -> list[str]:
        """relations label considering special_ids.

        Returns:
            list[str]: relations label considering special_ids.
        """
        relations_label = self.data.relations_label[:]
        for index, value in sorted(self._relation_special_dicts.items(), key=lambda x: x[0]):
            relations_label.insert(index, value)
        return relations_label

    @property
    def processed_relationIdx2countFrequency(self) -> np.ndarray:
        """relationIdx2countFrequency considering special_ids.

        Returns:
            np.ndarray: relationIdx2countFrequency considering special_ids.
        """
        id2count = self.data.relationIdx2countFrequency[:]
        indexes, _ = zip(*[*(sorted(self._relation_special_dicts.items(), key=lambda x: x[0]))])
        return np.insert(id2count, indexes, [-1] * len(indexes))

    def show(self, logger: Logger):
        """show params ig logger is not None.

        Args:
            logger(Logger):

        Raises:
            AttributeError:

        Returns:

        """
        if logger is not None:
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
        """ the property of train_dataloader

        Returns:
            DataLoader: train_dataloader
        """
        _loader = self._train_dataloader
        assert _loader is not None
        return _loader

    @property
    def train_valid_dataloader(self) -> DataLoader:
        """ the property of train_valid_dataloader
        train_valid_dataloader is the loader_data which is formatted like valid data but is train data.
        Returns:
            DataLoader: train_valid_dataloader
        """
        _loader = self._train_valid_dataloader
        assert _loader is not None
        return _loader

    @property
    def valid_dataloader(self) -> DataLoader:
        """ the property of valid_dataloader

        Returns:
            DataLoader: valid_dataloader
        """
        _loader = self._valid_dataloader
        assert _loader is not None
        return _loader

    @property
    def test_dataloader(self) -> DataLoader:
        """ the property of test_dataloader

        Returns:
            DataLoader: test_dataloader
        """
        _loader = self._test_dataloader
        assert _loader is not None
        return _loader


def main():
    """Main class for check its movement.
    """
    print("a")
    SRO_FOLDER = f"{PROJECT_DIR}/data/processed/KGCdata/All/SRO"
    SRO_ALL_INFO_FILE = f"{SRO_FOLDER}/info.hdf5"
    SRO_ALL_TRAIN_FILE = f"{SRO_FOLDER}/train.hdf5"
    logger = easy_logger(console_level='debug')
    logger.debug(f"{PROJECT_DIR=}")
    version_check(torch, pd, optuna, logger=logger)
    data_helper = MyDataHelper(SRO_ALL_INFO_FILE, SRO_ALL_TRAIN_FILE, None, None,
                               entity_special_dicts={0: '<pad>', 1: 'cls'},
                               relation_special_dicts={0: '<pad>', 1: 'cls'},
                               logger=logger)
    logger.debug(data_helper.processed_train_triple_geometric)


if __name__ == '__main__':
    main()
    pass
