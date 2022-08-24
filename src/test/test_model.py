import unittest
import model

# coding: UTF-8
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(os.path.join(PROJECT_DIR, 'src'))

# python
from logging import Logger
from typing import List, Dict, Tuple, Optional, Callable
import dataclasses
from tqdm import tqdm
# Machine learning
import h5py
import numpy as np
import pandas as pd
# torch
import torch
from torch.utils.data import Dataset
from torch import nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
# Made by me
from utils.setup import setup, save_param
from model import ConvE, DistMult, Complex, TransformerE

from models.main import *
from models.main import _info_str

PROCESSED_DATA_PATH = './data/processed/'
EXTERNAL_DATA_PATH = './data/external/'

KGDATA_ALL = ['FB15k-237', 'WN18RR', 'YAGO3-10']
name2model = {
    'conve': ConvE,
    'distmult': DistMult,
    'complex': Complex,
    'transformere': TransformerE
}


class EasyArge(dict):
    def __init__(self, **entries): self.__dict__.update(entries)


class MyTestCase(unittest.TestCase):
    def test00(self):
        tensor1 = torch.tensor([0, 1, 2])
        tensor2 = torch.tensor([3, 4, 5])
        tensor2[tensor1 != 0] = 6
        assert tensor2.tolist() == [3, 6, 6]

    def test01(self):
        from utils.setup import setup_logger
        logger = setup_logger("test", "./test.log", 'debug')
        kg_data = 'WN18RR'
        train_path = "./data/processed/KGdata/WN18RR/train.hdf5"
        valid_path = "./data/processed/KGdata/WN18RR/valid.hdf5"
        test_path = "./data/processed/KGdata/WN18RR/test.hdf5"

        logger.info(f"test01 start".center(40, '='))

        len_data_dict = {}
        for mode, path in {'train': train_path, 'valid': valid_path, 'test': test_path}.items():
            with h5py.File(path, mode='r') as f:
                data = f['triple'][1:]
                len_data_dict[mode] = len(data)
                del data
        len_data_dict['all'] = sum(list(len_data_dict.values()))

        # load data
        logger.info(_info_str(f"load data start."))
        data_helper = load_data(kg_data)
        logger.info(_info_str(f"load data complete."))

        logger.debug("==========check01==========")

        logger.debug(
            f"{len_data_dict['train']}, {len_data_dict['valid']}, {len_data_dict['test']}, {len_data_dict['all']}")
        label = data_helper.label

        u, counts = np.unique(label, return_counts=True)

        assert counts[u == 1] == len_data_dict['train']
        assert counts[u == 2] == len_data_dict['valid']
        assert counts[u == 3] == len_data_dict['test']

        dataset = data_helper.get_train_dataset(eco_memory=False)
        assert torch.count_nonzero(dataset.label).item() == len_data_dict['train']
        del dataset
        dataset = data_helper.get_valid_dataset()
        assert torch.count_nonzero(dataset.label_all == 2).item() == len_data_dict['valid']
        del dataset
        dataset = data_helper.get_test_dataset()
        assert torch.count_nonzero(dataset.label_all == 3).item() == len_data_dict['test']
        del dataset

        logger.debug("==========check01 end==========")

    def test02(self):
        from utils.setup import setup_logger
        logger = setup_logger("test", "./test.log", 'debug')
        kg_data = 'WN18RR'

        logger.info(f"test01 start".center(40, '='))

        # load data
        logger.info(_info_str(f"load data start."))
        data_helper = load_data(kg_data)
        logger.info(_info_str(f"load data complete."))

        logger.debug("==========check01==========")

        dataset1 = data_helper.get_train_dataset(eco_memory=False)
        dataset2 = data_helper.get_train_dataset(eco_memory=True)

        assert len(dataset1) == len(dataset2)
        for i in range(len(dataset1)):
            er_1, e2s_1 = dataset1[i]
            er_2, e2s_2 = dataset2[i]
            assert torch.equal(er_1, er_2)
            assert torch.equal(e2s_1.to(torch.int8), e2s_2.to(torch.int8))
        del dataset1, dataset2
        logger.debug("==========check01 end==========")

        logger.debug("==========check02==========")
        dataset1 = MyDataset(data_helper._er_list, data_helper.label, target_num=2, del_if_no_tail=True)
        dataset2 = data_helper.get_valid_dataset()

        assert len(dataset1) == len(dataset2)
        for i in range(len(dataset1)):
            er_1, e2s_1 = dataset1[i]
            er_2, e2s_target_2, e2s_2 = dataset2[i]
            assert torch.equal(er_1, er_2)
            assert torch.equal(e2s_1, e2s_target_2)

        del dataset1, dataset2
        logger.debug("==========check02 end==========")

        logger.debug("==========check03==========")
        dataset1 = MyDataset(data_helper._er_list, data_helper.label, target_num=3, del_if_no_tail=True)
        dataset2 = data_helper.get_test_dataset()

        assert len(dataset1) == len(dataset2)
        for i in range(len(dataset1)):
            er_1, e2s_1 = dataset1[i]
            er_2, e2s_target_2, e2s_2 = dataset2[i]
            assert torch.equal(er_1, er_2)
            assert torch.equal(e2s_1, e2s_target_2)

        del dataset1, dataset2
        logger.debug("==========check03 end==========")

    @torch.no_grad()
    def test03(self):
        from utils.setup import setup_logger
        logger = setup_logger("test", "./test.log", 'debug')
        kg_data = 'WN18RR'
        batch_size = 8
        label_smoothing = 0.05
        args = EasyArge(
            model='conve', use_bias=True, embedding_dim=10,
            input_drop=0., hidden_drop=0., feat_drop=0.,
            hidden_size=192, embedding_shape1=2,
        )

        logger.info(f"test01 start".center(40, '='))

        # load data
        logger.info(_info_str(f"load data start."))
        data_helper = load_data(kg_data)
        logger.info(_info_str(f"load data complete."))

        model = get_model(args, data_helper)

        logger.debug("==========check01==========")
        dataloader = DataLoader(data_helper.get_valid_dataset(), batch_size=batch_size, shuffle=False)

        mrr = torch.tensor(0., dtype=torch.float32)
        hit_ = [torch.tensor(0., dtype=torch.float32) for _ in range(10)]
        zero_tensor = torch.tensor(0., dtype=torch.float32)

        for idx, (er, e2s, e2s_all) in enumerate(dataloader):
            e2s_all = (e2s_all != 0)
            row, column = torch.where(e2s == 1)
            assert len(row) == len(column)
            logger.debug(f"{row.shape}")

            e, r = er.split(1, 1)
            logger.debug(f"{e.shape}, {r.shape}")
            pred: torch.Tensor = model(e, r)

            pred = pred[row]  # 複製
            e2s_all = e2s_all[row]  # 複製
            row = [i for i in range(len(column))]

            tmp = torch.count_nonzero(e2s_all).item()
            e2s_all[row, column] = False
            assert torch.count_nonzero(e2s_all[row, column]).item() == 0
            assert torch.count_nonzero(e2s_all).item() == (tmp - len(column))
            del tmp

            pred[e2s_all] = zero_tensor

            # logger.debug(f"{pred[row, column]}")
            assert pred[e2s_all].min() == 0.
            assert pred[e2s_all].max() == 0.
            logger.debug(f"{pred[row, column].min()}, {pred[row, column].max()}")

            ranking = torch.argsort(pred, dim=1, descending=False)  # これは0 始まり
            ranks = torch.squeeze(ranking[row, column]) + 1  # 1 始まり
            # mrr
            mrr += (1. / ranks).sum()
            # hit
            for i in range(10):
                hit_[i] += torch.count_nonzero(ranks <= (i + 1))

        logger.debug("==========check02 end==========")

    def test04(self):
        from utils.setup import setup_logger
        logger = setup_logger("test", "./test.log", 'debug')
        kg_data = 'WN18RR'
        batch_size = 8

        # load data
        logger.info(_info_str(f"load data start."))
        data_helper1 = load_data(kg_data, eco_memory=False)
        data_helper2 = load_data(kg_data, eco_memory=True)
        assert data_helper1.data.e_length == data_helper2.data.e_length
        assert len(data_helper1.er_list) == len(data_helper2.er_list)
        logger.info(_info_str(f"load data complete."))

        logger.debug("==========check01==========")

        dataset1 = MyDatasetWithFilter(data_helper1.er_list, data_helper1.label, target_num=2, del_if_no_tail=True)
        dataset2 = MyDatasetMoreEcoWithFilter(data_helper2.er_list, data_helper2._label_sparce, target_num=2,
                                              len_e=data_helper2.data.e_length, del_if_no_tail=True)
        assert len(dataset1) == len(dataset2)
        for i in range(len(dataset1)):
            er1, e2s_target1, e2s_all1 = dataset1[i]
            er2, e2s_target2, e2s_all2 = dataset2[i]

            assert torch.equal(er1, er2)
            assert torch.equal(e2s_target1, e2s_target2)
            assert torch.equal(e2s_all1, e2s_all2)

        logger.debug("==========check01 end==========")


if __name__ == '__main__':
    unittest.main()
