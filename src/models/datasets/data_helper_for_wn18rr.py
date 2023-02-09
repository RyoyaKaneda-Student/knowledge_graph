#!/usr/bin/python
# coding: UTF-8
"""Especially wn18rr

"""
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

from models.datasets.data_helper import make_change_index_func, MyDataHelper

from const.const_values import PROJECT_DIR

HEAD, RELATION, TAIL = 'head', 'relation', 'tail'
MODE = 'mode'
HYPERNYM = '_hypernym'


def _get_unique_list(series):
    item2count = pd.DataFrame(series.value_counts()).reset_index()
    item2count.columns = ['item', 'count']
    item2count = item2count.sort_values(['count', 'item'], ascending=[False, True])
    return item2count['item'].to_list()


def get_entity_relation(df):
    entities, relations = pd.concat([df[HEAD], df[TAIL]]).sort_values(), df[RELATION].sort_values()
    entities.columns, relations.columns = ['entity'], ['relation']
    return pd.concat([df[HEAD], df[TAIL]]).sort_values(), df[RELATION].sort_values()


def get_entity_relation_unique_list(df):
    entities, relations = get_entity_relation(df)
    return _get_unique_list(entities), _get_unique_list(relations)


def get_id2entity_id2relation(df):
    entities_list, relations_list = get_entity_relation_unique_list(df)
    return {i: e for i, e in enumerate(entities_list)}, {i: r for i, r in enumerate(relations_list)}


def get_entity2idx_relation2idx(df):
    entities_list, relations_list = get_entity_relation_unique_list(df)
    return {e: i for i, e in enumerate(entities_list)}, {r: i for i, r in enumerate(relations_list)}


def change_entity_relation(list_, entity2idx, relation2idx, entity_indexes, relation_indexes):
    new_list = []
    for i, item in enumerate(list_):
        new_list.append(entity2idx[item] if i in entity_indexes else
                        relation2idx[item] if i in relation_indexes else item)
    return tuple(new_list)


def get_hypernym_df(df):
    return df[df[RELATION] == HYPERNYM]


def get_hypernym_dict(df, entities):
    hypernym_df = get_hypernym_df(df)
    hypernym_dict = {key: [] for key in entities}
    for index, row in hypernym_df.iterrows():
        hypernym_dict[row[HEAD]].append(row[TAIL])
    return hypernym_dict


def get_entity2triples(df) -> dict:
    entities, _ = get_entity_relation(df)
    entity2triples = {key: [] for key in entities}
    for index, row in df.iterrows():
        triple = (row[HEAD], row[RELATION], row[TAIL], row[MODE])
        entity2triples[row[HEAD]].append(triple)
    return entity2triples


def get_hypernym_list(key, list_, *, hypernym_dict):
    list_ = list_ + [key]
    list_list = []
    for child_key in hypernym_dict[key]:
        if child_key in set(list_):
            list_list.append(list_)
        else:
            list_list.extend(get_hypernym_list(child_key, list_, hypernym_dict=hypernym_dict))
    if len(list_list) == 0:
        list_list = [list_]
    return list_list


def get_to_top_list_list(df):
    hypernym_df = get_hypernym_df(df)
    entities, _ = get_entity_relation(df)
    hypernym_bottom_entity_set = set(entities) - set(hypernym_df[TAIL])
    hypernym_dict = get_hypernym_dict(hypernym_df, entities)
    to_top_list_list = []
    for e in sorted(hypernym_bottom_entity_set):
        to_top_list = get_hypernym_list(e, [], hypernym_dict=hypernym_dict)
        to_top_list_list.extend(to_top_list)
    return to_top_list_list


def get_to_top_triples_list(df, to_top_list_list):
    entity2triples = get_entity2triples(df)
    to_top_triples_list = []
    for to_top_list in to_top_list_list:
        to_top_triples = []
        for e in to_top_list:
            to_top_triples.extend(entity2triples[e])
        to_top_triples_list.append(to_top_triples)
    return to_top_triples_list


def get_to_top_triples_list_limit_(to_top_triples_list, limit):
    new_to_top_triples_list_limit = []
    for list_ in to_top_triples_list:
        for i in range(0, len(list_), limit):
            new_to_top_triples_list_limit.append(list_[i: i + limit])
    return new_to_top_triples_list_limit


def get_to_top_array_list(to_top_triples_list, entity2idx, relation2idx):
    to_top_array_list = [np.array(
        [change_entity_relation(_triple, entity2idx, relation2idx, (0, 2), (1,)) for _triple in to_top_triples]
    ) for to_top_triples in to_top_triples_list]
    return to_top_array_list


def get_sequence_array(array_list, padding_token, max_len):
    sequence_array = np.array([[padding_token for _ in range(max_len)] for _ in range(len(array_list))])
    for i, _array in enumerate(array_list): sequence_array[i, :len(_array)] = _array
    return sequence_array


def make_get_sequence_array(df, entity2idx, relation2idx, to_top_list_list, limit=None):
    df = df.sort_values([HEAD, RELATION, TAIL]).reset_index()
    to_top_triples_list = get_to_top_triples_list(df, to_top_list_list)
    limit = limit or max([len(triples) for triples in to_top_triples_list])
    to_top_triples_list_limit_ = get_to_top_triples_list_limit_(to_top_triples_list, limit)

    train_to_top_array_list = get_to_top_array_list(to_top_triples_list_limit_, entity2idx, relation2idx)

    return train_to_top_array_list


@dataclass(init=False)
class RawDataForWN18RR:
    """RawDataForWN18RR

    """

    def __init__(self, train_path, valid_path, test_path, train_mode=1, valid_mode=2, test_mode=3):
        train_df = pd.read_table(train_path, header=None, names=(HEAD, RELATION, TAIL)).assign(**{MODE: train_mode})
        valid_df = pd.read_table(valid_path, header=None, names=(HEAD, RELATION, TAIL)).assign(**{MODE: valid_mode})
        test_df = pd.read_table(test_path, header=None, names=(HEAD, RELATION, TAIL)).assign(**{MODE: test_mode})
        all_df = pd.concat([train_df, valid_df, test_df])
        to_top_list_list = get_to_top_list_list(train_df)
        entities, relations = get_entity_relation(all_df)

        self.entities, self.relations = entities.to_list(), relations.to_list()
        self.entity_num, self.relation_num = len(self.entities), len(self.relations)
        self.train_df = train_df
        self.valid_df = valid_df
        self.test_df = test_df
        self.all_df = all_df
        self.to_top_list_list = to_top_list_list

    def _sequence_array(self, df, limit) -> list[np.ndarray]:
        to_top_list_list = self.to_top_list_list
        entity2idx, relation2idx = get_entity2idx_relation2idx(self.all_df)
        return make_get_sequence_array(df, entity2idx, relation2idx, to_top_list_list, limit)

    def _get_triple(self, df) -> np.ndarray:
        entity2idx, relation2idx = get_entity2idx_relation2idx(self.all_df)
        return np.ndarray([
            (entity2idx[row[HEAD]], relation2idx[row[RELATION]], entity2idx[row[TAIL]]) for index, row in df.iterrows()
        ], dtype=int)

    def get_train_triple(self):
        """get train triple

        """
        return self._get_triple(self.train_df)

    def get_valid_triple(self):
        """get valid triple

        """
        return self._get_triple(self.valid_df)

    def get_test_triple(self):
        """get test triple

        """
        return self._get_triple(self.test_df)

    def get_train_array_list(self, limit):
        """get_train_array_list

        """
        df = self.train_df
        return self._sequence_array(df, limit)

    def get_valid_array_list(self, limit):
        """get_valid_array_list

        """
        df = pd.concat([self.train_df, self.valid_df])
        return self._sequence_array(df, limit)

    def get_test_array_list(self, limit):
        """get_test_array_list

        """
        df = pd.concat([self.train_df, self.valid_df, self.test_df])
        return self._sequence_array(df, limit)


@dataclass(init=False)
class MyDataHelperForWN18RR(MyDataHelper):
    """MyDataHelperForStory For WN18RR

    """

    def __init__(self, train_path, valid_path, test_path, sequence_length, *,
                 entity_special_dicts: dict[int, str], relation_special_dicts: dict[int, str], logger: Logger = None):
        self.sequence_length = sequence_length
        self._data = RawDataForWN18RR(train_path, valid_path, test_path)
        self._entity_special_dicts: dict[int, str] = entity_special_dicts
        self._relation_special_dicts: dict[int, str] = relation_special_dicts
        self._change_entity_func = None
        self._change_relation_func = None
        # init
        self.change_special_dicts(entity_special_dicts, relation_special_dicts)

    def change_special_dicts(self, entity_special_dicts, relation_special_dicts):
        """change_special_dicts

        """
        entity_num, relation_length = self.data.entity_num, self.data.relation_num
        entity_special_ids = tuple(entity_special_dicts.keys())
        relation_special_ids = tuple(relation_special_dicts.keys())

        self._entity_special_dicts: dict[int, str] = entity_special_dicts or self._entity_special_dicts
        self._relation_special_dicts: dict[int, str] = relation_special_dicts or self._relation_special_dicts
        self._change_entity_func = np.frompyfunc(make_change_index_func(entity_num, entity_special_ids), 1, 1)
        self._change_relation_func = np.frompyfunc(make_change_index_func(relation_length, relation_special_ids), 1, 1)

    def _special_token_change(self, raw_array: np.ndarray, entity_index, relation_index):
        _change_entity_func, _change_relation_func = self._change_entity_func, self._change_relation_func

        processed_array = np.copy(raw_array)
        processed_array[:, entity_index] = _change_entity_func(processed_array[:, entity_index])
        processed_array[:, relation_index] = _change_relation_func(processed_array[:, relation_index])
        return processed_array

    def _processed_triple(self, triple) -> np.ndarray:
        return self._special_token_change(triple, entity_index=(0, 2), relation_index=(1,))

    @property
    def data(self) -> RawDataForWN18RR:
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
    def processed_entity2index(self) -> dict[str, int]:
        """entity2index considering special_ids.

        Returns:
            dict[str, int]: entity2index considering special_ids.
        """
        return {e: i for i, e in enumerate(self.processed_entities)}

    @property
    def processed_relation2index(self) -> dict[str, int]:
        """relation2index considering special_ids.

        Returns:
            dict[str, int]: relation2index considering special_ids.
        """
        return {r: i for i, r in enumerate(self.processed_relations)}

    def get_processed_triple(self, str_triple) -> np.ndarray:
        """get_processed_triple

        """
        entity2index, relation2index = self.processed_entity2index, self.processed_relation2index
        return np.array([
            (entity2index[h], relation2index[r], entity2index[t]) for h, r, t in str_triple
        ])

    def _processed_sequence(self, e_pad_id, r_pad_id, mode_pad, array_list) -> np.ndarray:
        processed_array_list = [self._processed_triple(array_) for array_ in array_list]
        padding_token = [e_pad_id, r_pad_id, e_pad_id, mode_pad]
        return get_sequence_array(processed_array_list, padding_token, self.sequence_length)

    def get_processed_train_sequence(self, e_pad_id, r_pad_id, mode_pad) -> np.ndarray:
        """get_processed_train_sequence

        """
        return self._processed_sequence(
            e_pad_id, r_pad_id, mode_pad, self.data.get_train_array_list(self.sequence_length))

    def get_processed_valid_sequence(self, e_pad_id, r_pad_id, mode_pad) -> np.ndarray:
        """get_processed_valid_sequence

        """
        return self._processed_sequence(
            e_pad_id, r_pad_id, mode_pad, self.data.get_valid_array_list(self.sequence_length))

    def get_processed_test_sequence(self, e_pad_id, r_pad_id, mode_pad) -> np.ndarray:
        """get_processed_test_sequence

        """
        return self._processed_sequence(
            e_pad_id, r_pad_id, mode_pad, self.data.get_test_array_list(self.sequence_length))


def main():
    """main

    """
    pass


if __name__ == '__main__':
    main()
