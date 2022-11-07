# coding: UTF-8
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import h5py
import numpy as np

from utils.textInOut import *
from utils.setup import setup, save_param

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(os.path.join(PROJECT_DIR, 'src'))

PROCESSED_DATA_PATH = './data/processed/'
EXTERNAL_DATA_PATH = './data/external/'


class ConstMeta(type):
    def __setattr__(self, name, value):
        if name in self.__dict__:
            raise TypeError(f'Can\'t rebind const ({name})')
        else:
            self.__setattr__(name, value)


def setup_parser(args=None):
    import argparse  # 1. argparseをインポート
    parser = argparse.ArgumentParser(description='データの初期化')
    parser.add_argument('--logfile', help='ログファイルのパス', type=str)
    parser.add_argument('--param-file', help='パラメータを保存するファイルのパス', type=str)
    parser.add_argument('--console-level', help='コンソール出力のレベル', type=str, default='info')

    parser.add_argument('--KGdata', help=' or '.join(KGDATA_ALL), type=str, choices=KGDATA_ALL)

    return parser.parse_args(args)


KGDATA_ALL = ['FB15k-237', 'WN18RR', 'YAGO3-10']


def branch_dict():
    return {
        KGDATA_ALL[0]: fb15k_237,
        KGDATA_ALL[1]: wn18rr,
        KGDATA_ALL[2]: yago3_10
    }


"""
(
    entity_length
    relation_length
    entity2id
    entity2count
    entity2used_mode
    id2entity
    relation2id
    relation2count
)   in info.hdf5
triple in train.hdf5
triple in valid.hdf5
triple in test.hdf5
"""


class INFO_KEY_NAME(ConstMeta):
    entity_length = 'entity_length'
    entities = 'entities'
    id2count_entity = 'id2count_entity'
    id2used_mode_entity = 'id2used_mode_entity'
    relation_length = 'relation_length'
    relations = 'relations'
    id2count_relation = 'id2count_relation'
    id2is_reverse_relation = 'id2is_reverse_relation'


class ALL_TAIL_KEY_NAME(ConstMeta):
    er_length = 'er_length'
    ers = 'ers'
    id2all_tail_row = 'id2all_tail_row'
    id2all_tail_entity = 'id2all_tail_entity'
    id2all_tail_mode = 'id2all_tail_mode'


class ALL_RELATION_KEY_NAME(ConstMeta):
    ee_length = 'ee_length'
    ees = 'ees'
    id2all_relation_row = 'id2all_relation_row'
    id2all_relation_relation = 'id2all_relation_relation'
    id2all_relation_mode = 'id2all_relation_mode'


ID = 'id'
COUNT = 'count'
IS_REVERSE = 'is_reverse'
TRIPLE = 'triple'
TRAIN = 'train'
VALID = 'valid'
TEST = 'test'
modeHDF5 = lambda mode: f'{mode}.hdf5'
KGDATA = 'KGdata'
REVERSE = 'REVERSE'
keyREVERSE = lambda key: f'{key}_REVERSE'


@dataclass
class WordInfo:
    id: int
    count: int
    is_reverse: Optional[bool] = None
    used_mode: dict = field(default_factory=lambda: {TRAIN: False, VALID: False, TEST: False})


WordDictType = dict[str, WordInfo]
ER_Type = tuple[int, int]
All_Tail_Type = list[tuple[int, int]]
EE_Type = tuple[int, int]
All_Relation_Type = list[tuple[int, int]]


def _make_triple(read_path, write_path, dict_e: WordDictType, dict_r: WordDictType, mode):
    list_3: list[[int, int, int]] = []
    with open(read_path) as f:
        line = f.readline()
        while line:
            e1, r, e2 = line.replace('\n', '').lower().split('\t')
            e1 = dict_e[e1]
            r = dict_r[r]
            e2 = dict_e[e2]
            list_3.append((e1.id, r.id, e2.id))
            e1.used_mode[mode] = True
            r.used_mode[mode] = True
            e2.used_mode[mode] = True
            line = f.readline()
    with h5py.File(write_path, mode='w') as f:
        f.create_dataset(TRIPLE, data=np.array(list_3))


def make_triple(dataset_name, train_file='train.txt', valid_file='valid.txt', test_file='test.txt', *, logger: Logger):
    # mode_list = [TRAIN, VALID, TEST]
    logger.info("make triple data")
    save_hdf_path = os.path.join(PROCESSED_DATA_PATH, 'KGdata', dataset_name, f"info.hdf5")
    mode2file_list = {TRAIN: train_file, VALID: valid_file, TEST: test_file}
    mode2triple_filepath = {
        mode: os.path.join(EXTERNAL_DATA_PATH, KGDATA, dataset_name, file_)
        for mode, file_ in mode2file_list.items()
    }

    counter_e = Counter()
    counter_r = Counter()

    for _, path_ in mode2triple_filepath.items():
        with open(path_) as f:
            line = f.readline()
            while line:
                e1, r, e2 = line.replace('\n', '').lower().split('\t')
                counter_e[e1] += 1
                counter_r[r] += 1
                counter_e[e2] += 1
                line = f.readline()
    assert sum(counter_e.values()) // 2 == sum(counter_r.values())

    dict_e: WordDictType = {word: WordInfo(i, c) for i, (word, c) in enumerate(counter_e.most_common())}
    dict_r: WordDictType = {word: WordInfo(i, c, False) for i, (word, c) in enumerate(counter_r.most_common())}

    len_dict_r = len(dict_r)
    dict_r |= {keyREVERSE(key): WordInfo(value.id, value.count, True) for key, value in dict_r.items()}
    assert len(dict_r) == 2 * len_dict_r

    for mode, file_ in mode2file_list.items():
        read_path = os.path.join(EXTERNAL_DATA_PATH, KGDATA, dataset_name, file_)
        write_path = os.path.join(PROCESSED_DATA_PATH, KGDATA, dataset_name, modeHDF5(mode))
        _make_triple(read_path, write_path, dict_e, dict_r, mode)

    entity_length = len(dict_e)
    id2entity = {value.id: key for key, value in dict_e.items()}
    entities = [id2entity[i] for i in sorted(id2entity.keys())]

    id2count_entity = [dict_e[e].count for e in entities]
    id2used_mode_entity = [
        [None, dict_e[e].used_mode[TRAIN], dict_e[e].used_mode[VALID], dict_e[e].used_mode[TEST]]
        for e in entities]
    id2used_mode_entity = np.array(id2used_mode_entity, dtype=bool)

    relation_length = len(dict_r)
    id2relation = {value.id: key for key, value in dict_r.items()}
    relations = [id2relation[i] for i in sorted(id2relation.keys())]

    id2count_relation = [dict_r[r].count for r in relations]
    id2is_reverse_relation = [dict_r[r].is_reverse for r in relations]

    with h5py.File(save_hdf_path, mode='w') as f:
        f.create_dataset(INFO_KEY_NAME.entity_length, data=entity_length)
        f.create_dataset(INFO_KEY_NAME.entities, data=entities)
        f.create_dataset(INFO_KEY_NAME.id2count_entity, data=id2count_entity)
        f.create_dataset(INFO_KEY_NAME.id2used_mode_entity, data=id2used_mode_entity)

        f.create_dataset(INFO_KEY_NAME.relation_length, data=relation_length)
        f.create_dataset(INFO_KEY_NAME.relations, data=relations)
        f.create_dataset(INFO_KEY_NAME.id2count_relation, data=id2count_relation)
        f.create_dataset(INFO_KEY_NAME.id2is_reverse_relation, data=id2is_reverse_relation)


def make_er_all_tail(dataset_name, *, logger: Logger):
    logger.info("make all_tail data")
    mode_list = [TRAIN, VALID, TEST]
    mode2id = {TRAIN: 1, VALID: 2, TEST: 3}
    # info_hdf_path = os.path.join(PROCESSED_DATA_PATH, 'KGdata', dataset_name, f"info.hdf5")
    save_hdf_path = os.path.join(PROCESSED_DATA_PATH, 'KGdata', dataset_name, f"all_tail.hdf5")
    mode2triple_hdf_path = {
        mode: os.path.join(PROCESSED_DATA_PATH, KGDATA, dataset_name, modeHDF5(mode))
        for mode in mode_list
    }
    er2all_tail: defaultdict[ER_Type, All_Tail_Type] = defaultdict(lambda: [])
    for mode, triple_hdf_path in mode2triple_hdf_path.items():
        mode_id = mode2id[mode]
        with h5py.File(triple_hdf_path, 'r') as f:
            [er2all_tail[(e1, r)].append((e2, mode_id)) for e1, r, e2 in f['triple'][:].tolist()]

    [all_tail.sort(key=lambda x: x[0]) for all_tail in er2all_tail.values()]

    er_length = len(er2all_tail)
    er2all_tail_items = [(key, value) for key, value in er2all_tail.items()]
    er2all_tail_items.sort(key=lambda x: len(x[1]))  # tail length

    id2er = {i: er for i, (er, _) in enumerate(er2all_tail_items)}

    ers = [id2er[i] for i in sorted(id2er.keys())]

    tmp = [(i, t[0], t[1]) for i, er in enumerate(ers) for t in er2all_tail[er]]
    id2all_tail_row, id2all_tail_entity, id2all_tail_mode = zip(*tmp)

    with h5py.File(save_hdf_path, mode='w') as f:
        f.create_dataset(ALL_TAIL_KEY_NAME.er_length, data=er_length)
        f.create_dataset(ALL_TAIL_KEY_NAME.ers, data=ers)
        f.create_dataset(ALL_TAIL_KEY_NAME.id2all_tail_row, data=id2all_tail_row)
        f.create_dataset(ALL_TAIL_KEY_NAME.id2all_tail_entity, data=id2all_tail_entity)
        f.create_dataset(ALL_TAIL_KEY_NAME.id2all_tail_mode, data=id2all_tail_mode)


def make_ee_all_relation(dataset_name, *, logger: Logger):
    logger.info("make all_relation data")
    mode_list = [TRAIN, VALID, TEST]
    mode2id = {TRAIN: 1, VALID: 2, TEST: 3}
    # info_hdf_path = os.path.join(PROCESSED_DATA_PATH, 'KGdata', dataset_name, f"info.hdf5")
    save_hdf_path = os.path.join(PROCESSED_DATA_PATH, 'KGdata', dataset_name, f"all_relation.hdf5")
    mode2triple_hdf_path = {
        mode: os.path.join(PROCESSED_DATA_PATH, KGDATA, dataset_name, modeHDF5(mode))
        for mode in mode_list
    }
    ee2all_relation: defaultdict[EE_Type, All_Relation_Type] = defaultdict(lambda: [])
    for mode, triple_hdf_path in mode2triple_hdf_path.items():
        mode_id = mode2id[mode]
        with h5py.File(triple_hdf_path, 'r') as f:
            [ee2all_relation[(e1, e2)].append((r, mode_id)) for e1, r, e2 in f['triple'][:].tolist()]

    [all_relation.sort(key=lambda x: x[0]) for all_relation in ee2all_relation.values()]

    ee_length = len(ee2all_relation)
    ee2all_relation_items = [(key, value) for key, value in ee2all_relation.items()]
    ee2all_relation_items.sort(key=lambda x: len(x[1]))  # relation length

    id2ee = {i: ee for i, (ee, _) in enumerate(ee2all_relation_items)}

    ees = [id2ee[i] for i in sorted(id2ee.keys())]

    tmp = [(i, t[0], t[1]) for i, ee in enumerate(ees) for t in ee2all_relation[ee]]
    id2all_relation_row, id2all_relation_relation, id2all_relation_mode = zip(*tmp)

    with h5py.File(save_hdf_path, mode='w') as f:
        f.create_dataset(ALL_RELATION_KEY_NAME.ee_length, data=ee_length)
        f.create_dataset(ALL_RELATION_KEY_NAME.ees, data=ees)
        f.create_dataset(ALL_RELATION_KEY_NAME.id2all_relation_row, data=id2all_relation_row)
        f.create_dataset(ALL_RELATION_KEY_NAME.id2all_relation_relation, data=id2all_relation_relation)
        f.create_dataset(ALL_RELATION_KEY_NAME.id2all_relation_mode, data=id2all_relation_mode)


def fb15k_237(*, logger: Logger):
    name = 'FB15k-237'
    make_triple(name, logger=logger)
    make_er_all_tail(name, logger=logger)
    make_ee_all_relation(name, logger=logger)


def wn18rr(*, logger: Logger):
    name = 'WN18RR'
    make_triple(
        name,
        train_file='text/train.txt',
        valid_file='text/valid.txt',
        test_file='text/test.txt',
        logger=logger
    )
    make_er_all_tail(name, logger=logger)
    make_ee_all_relation(name, logger=logger)


def yago3_10(*, logger: Logger):
    name = 'YAGO3-10'
    make_triple(name, logger=logger)
    make_er_all_tail(name, logger=logger)
    make_ee_all_relation(name, logger=logger)


def main():
    args, logger, device = setup(setup_parser, PROJECT_DIR)
    try:
        args.project_dir = PROJECT_DIR
        args.logger = logger
        logger.info(vars(args))
        branch_dict()[args.KGdata](logger=logger)

    finally:
        save_param(args)


if __name__ == '__main__':
    main()
