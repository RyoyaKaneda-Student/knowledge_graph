# coding: UTF-8
import os
import sys
from pathlib import Path
import h5py
import numpy as np
import dataclasses
from tqdm import tqdm
import os
from logging import Logger

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(os.path.join(PROJECT_DIR, 'src'))

import gzip
import json
from requests import get as requests_get

import pandas as pd

from utils.textInOut import *
from utils.setup import setup, save_param

PROCESSED_DATA_PATH = './data/processed/'
EXTERNAL_DATA_PATH = './data/external/'


def setup_parser(args=None):
    import argparse  # 1. argparseをインポート
    parser = argparse.ArgumentParser(description='データの初期化')
    parser.add_argument('--logfile', help='ログファイルのパス', type=str)
    parser.add_argument('--param-file', help='パラメータを保存するファイルのパス', type=str)
    parser.add_argument('--console-level', help='コンソール出力のレベル', type=str, default='info')
    parser.add_argument('--device-name', help='cpu or cuda or mps', type=str, default='cpu',
                        choices=['cpu', 'cuda', 'mps'])

    kd_data_names = list(branch_dict().keys())
    parser.add_argument('--KGdata', help=' or '.join(kd_data_names), type=str,
                        choices=kd_data_names)

    parser.add_argument('--force-remake', help='', action='store_true')
    parser.add_argument('--add-reverse', help='', action='store_true')
    parser.add_argument('--task-r2t', help='task(Raw to triples)', action='store_true')
    parser.add_argument('--task-t2s', help='task(triples to training case)', action='store_true')

    # コマンドライン引数をパースして対応するハンドラ関数を実行
    _args = parser.parse_args()

    return _args


def branch_dict():
    return {
        'FB15k-237': fb15k_237,
        'WN18RR': wn18rr,
        'YAGO3-10': yago3_10
    }


class Raw2Triples:
    def __init__(self, name, *,
                 train_file='train.txt',
                 valid_file='valid.txt',
                 test_file='test.txt',
                 force_remake=False,
                 add_reverse=True
                 ):
        self.dict_e: dict[str, int] = {'': 0}
        self.len_dict_e: int = 1
        self.dict_r: dict[str, int] = {'': 0}
        self.len_dict_r: int = 1
        self.dict_r_is_reverse: dict[str, bool] = {'': False}
        self.list_3: list[[int, int, int]] = [[0, 0, 0]]
        self.name = name
        self.train_file = train_file
        self.valid_file = valid_file
        self.test_file = test_file
        self.add_reverse = add_reverse

    def reset_list_3(self):
        del self.list_3
        self.list_3: list[[int, int, int]] = [[0, 0, 0]]

    def get_dicts(self):
        return self.dict_e, self.dict_r

    def _new_e(self, x):
        len_ = self.len_dict_e
        self.dict_e[x] = len_
        self.len_dict_e += 1
        return len_

    def _new_r(self, x, *, is_reverse):
        len_ = self.len_dict_r
        self.dict_r[x] = len_
        self.dict_r_is_reverse[x] = is_reverse
        self.len_dict_r += 1
        return len_

    def _str2ids(self, e1: str, r: str, e2: str, rr=None):
        dict_e, dict_r = self.get_dicts()
        e1 = dict_e[e1] if e1 in dict_e.keys() else self._new_e(e1)
        r = dict_r[r] if r in dict_r.keys() else self._new_r(r, is_reverse=False)
        e2 = dict_e[e2] if e2 in dict_e.keys() else self._new_e(e2)
        if rr is not None: rr = dict_r[rr] if rr in dict_r.keys() else self._new_r(rr, is_reverse=True)
        return e1, r, e2, rr

    def _append_list(self, e1: int, r: int, e2: int):
        self.list_3.append([e1, r, e2])  # save

    def _process1(self, line):
        e1, r, e2 = line.replace('\n', '').lower().split('\t')  # get
        rr = r + '_REVERSE' if self.add_reverse else None
        e1, r, e2, rr = self._str2ids(e1, r, e2, rr)  # to ids
        self._append_list(e1, r, e2)  # save
        if rr is not None:
            self._append_list(e2, rr, e1)  # save

    def _file_length(self, path):
        with open(path) as f:
            rep = len([True for _ in f])
            return rep

    def _process(self, path):
        total_ = self._file_length(path)
        with open(path) as f:
            for line in tqdm(f, total=total_, leave=False):
                self._process1(line)
        return total_

    def process(self, *, logger):
        total_count_dict = {}
        for mode, _file in (
                ('train', self.train_file),
                ('valid', self.valid_file),
                ('test', self.test_file)
        ):
            self.reset_list_3()
            path = os.path.join(EXTERNAL_DATA_PATH, 'KGdata', self.name, _file)
            total_count = self._process(path)
            logger.info(f"{mode} length: {total_count}")
            total_count_dict[mode] = total_count
            save_hdf_path = os.path.join(PROCESSED_DATA_PATH, 'KGdata', f"{self.name}", f"{mode}.hdf5")
            with h5py.File(save_hdf_path, mode='w') as f:
                f.create_dataset('triple_tmp', data=np.array(self.list_3))
            logger.info(f"{_file} complete")

        dict_r_items = sorted(self.dict_r.items(), key=lambda x: x[1])
        id2id = {value: (value + self.len_dict_r if self.dict_r_is_reverse[key] else value)
                 for (key, value) in dict_r_items}
        id2id = {key: i for i, (key, _) in enumerate(sorted(id2id.items(), key=lambda x: x[1]))}
        self.dict_r = {key: id2id[value] for key, value in dict_r_items}
        del dict_r_items

        for mode in ('train', 'valid', 'test'):
            save_hdf_path = os.path.join(PROCESSED_DATA_PATH, 'KGdata', f"{self.name}", f"{mode}.hdf5")
            with h5py.File(save_hdf_path, mode='a') as f:
                triple: np.ndarray = f['triple_tmp'][:]
                del f['triple_tmp']
                new_r_list = []
                for _, r, _ in triple:
                    new_r_list.append(id2id[r])
                triple[:, 1] = np.array(new_r_list)
                f.create_dataset('triple', data=triple)

        save_hdf_path = os.path.join(PROCESSED_DATA_PATH, 'KGdata', f"{self.name}", f"info.hdf5")
        logger.info(f"len_dict_e={self.len_dict_e}, len_dict_r={self.len_dict_r}")
        dict_e_ = {value: key for key, value in self.dict_e.items()}
        del self.dict_e
        dict_r_ = {value: key for key, value in self.dict_r.items()}
        dict_r_is_reverse_ = {value: self.dict_r_is_reverse[key] for key, value in self.dict_r.items()}
        del self.dict_r, self.dict_r_is_reverse
        print([dict_r_is_reverse_[i] for i in range(self.len_dict_r)])

        with h5py.File(save_hdf_path, mode='a') as f:
            f.create_dataset('e_length', data=self.len_dict_e)
            f.create_dataset('r_length', data=self.len_dict_r)
            f.create_dataset('item_e', data=[dict_e_[i] for i in range(self.len_dict_e)])
            f.create_dataset('item_r', data=[dict_r_[i] for i in range(self.len_dict_r)])
            f.create_dataset('item_r_is_reverse',
                             data=np.array([dict_r_is_reverse_[i] for i in range(self.len_dict_r)]))

        # check
        logger.info("==========check1===========")
        for mode in ('train', 'valid', 'test'):
            save_hdf_path = os.path.join(PROCESSED_DATA_PATH, 'KGdata', f"{self.name}", f"{mode}.hdf5")
            with h5py.File(save_hdf_path, mode='r') as f:
                data = f['triple'][:]
                len_data = len(data)
                logger.info(f"{mode} len = {len_data}")
                assert len_data == total_count_dict[mode] * 2 + 1
        logger.info("==========check1===========")


class Triple2StudyCase:
    def __init__(self, name, *, logger, force_remake=False):
        # train
        load_hdf_path = os.path.join(PROCESSED_DATA_PATH, 'KGdata', name, f"info.hdf5")
        with h5py.File(load_hdf_path) as f:
            e_length, r_length = f['e_length'], f['r_length']
            list_r_is_reverse = f['item_r_is_reverse'][:]

        self.train_path = os.path.join(PROCESSED_DATA_PATH, 'KGdata', name, f"train.hdf5")
        self.valid_path = os.path.join(PROCESSED_DATA_PATH, 'KGdata', name, f"valid.hdf5")
        self.test_path = os.path.join(PROCESSED_DATA_PATH, 'KGdata', name, f"test.hdf5")
        self.e_length = e_length
        self.r_length = r_length
        self.list_r_is_reverse = list_r_is_reverse

        self.er2id = {(0, 0): 0}
        self.len_er2id = 1
        self.id2list = {0: [0]}
        self.id2list_data_type = {0: [0]}  # 1: train, 2: valid, 3: test
        self.id2_is_reverse = {0: False}

    def _new_id(self, x):
        len_ = self.len_er2id
        self.er2id[x] = len_
        self.len_er2id += 1
        self.id2list[len_] = []
        self.id2list_data_type[len_] = []
        e, r = x
        self.id2_is_reverse[len_] = self.list_r_is_reverse[r]
        return len_

    def _add_er(self, e1, r, e2, data_type):
        er2id = self.er2id
        er, value = (e1, r), e2
        id_ = er2id[er] if er in er2id.keys() else self._new_id(er)
        self.id2list[id_].append(value)
        self.id2list_data_type[id_].append(data_type)

    def _process(self, path, data_type):
        with h5py.File(path, 'r') as f:
            dataset = f['triple'][:]
        # 0が足されないための配慮
        assert self.id2list_data_type[0] == [0]
        for data in dataset[1:]:
            e1, r, e2 = data.tolist()
            self._add_er(e1, r, e2, data_type)

    def process(self, *, logger):
        self._process(self.train_path, data_type=1)  # train
        self._process(self.valid_path, data_type=2)  # valid
        self._process(self.test_path, data_type=3)  # test

        len_er2id, er2id = self.len_er2id, self.er2id
        id2list, id2list_data_type = self.id2list, self.id2list_data_type
        id2_is_reverse = self.id2_is_reverse

        id2er = {value: key for key, value in er2id.items()}

        er_list = [id2er[i] for i in range(len_er2id)]
        id_list = [id2list[i] for i in range(len_er2id)]
        id_is_reverse = [id2_is_reverse[i] for i in range(len_er2id)]

        id_list_data_type = [id2list_data_type[i] for i in range(len_er2id)]
        id_list_sparce = {
            'data': np.array([n for item in id_list for n in item]),
            'row': np.array([i for i, item in enumerate(id_list) for _ in item]),
            'data_type': np.array([n for item in id_list_data_type for n in item]),
        }

        with h5py.File(self.train_path, 'a') as f:
            f.create_dataset('er_list', data=er_list)
            f.create_dataset('er_is_reverse', data=id_is_reverse)
            f.create_dataset('er_tails_data', data=id_list_sparce['data'])
            f.create_dataset('er_tails_row', data=id_list_sparce['row'])
            f.create_dataset('er_tails_data_type', data=id_list_sparce['data_type'])
        logger.info("Triple2StudyCase process complete")

        # check
        logger.info("==========check2===========")
        len_data_dict = {}
        for mode, path in {'train': self.train_path, 'valid': self.valid_path, 'test': self.test_path}.items():
            with h5py.File(path, mode='r') as f:
                data = f['triple'][:]
                len_data_dict[mode] = len(data)
        len_data_dict['all'] = sum(list(len_data_dict.values()))
        with h5py.File(self.train_path, 'r') as f:
            er_tails_data: np.ndarray = f['er_tails_data'][:]
            assert len_data_dict['all'], len(er_tails_data)
            u, counts = np.unique(er_tails_data, return_counts=True)
            assert 1, counts[0].item()
            assert len_data_dict['train'], counts[u == 1].item()
            assert len_data_dict['valid'], counts[u == 2].item()
            assert len_data_dict['test'], counts[u == 3].item()
        logger.info("==========check2===========")


def fb15k_237(*, args, logger):
    name = 'FB15k-237'
    if args.task_r2t:
        r2t = Raw2Triples(name=name, add_reverse=args.add_reverse)
        r2t.process(logger=logger)
        del r2t
    if args.task_t2s:
        t2s = Triple2StudyCase(name=name, logger=logger, force_remake=args.force_remake)
        t2s.process(logger=logger)


def wn18rr(*, args, logger):
    name = 'WN18RR'
    if args.task_r2t:
        r2t = Raw2Triples(
            name=name,
            train_file='text/train.txt',
            valid_file='text/valid.txt',
            test_file='text/test.txt',
            add_reverse=args.add_reverse
        )
        r2t.process(logger=logger)
        del r2t
    if args.task_t2s:
        t2s = Triple2StudyCase(name=name, logger=logger, force_remake=args.force_remake)
        t2s.process(logger=logger)


def yago3_10(*, args, logger):
    name = 'YAGO3-10'
    if args.task_r2t:
        r2t = Raw2Triples(
            name=name, add_reverse=args.add_reverse
        )
        r2t.process(logger=logger)
        del r2t
    if args.task_t2s:
        t2s = Triple2StudyCase(name=name, logger=logger, force_remake=args.force_remake)
        t2s.process(logger=logger)


def main():
    args, logger, device = setup(setup_parser, PROJECT_DIR)
    try:
        args.project_dir = PROJECT_DIR
        args.logger = logger
        args.device = device
        logger.info(vars(args))
        branch_dict()[args.KGdata](args=args, logger=logger)

    finally:
        save_param(args)


if __name__ == '__main__':
    main()
