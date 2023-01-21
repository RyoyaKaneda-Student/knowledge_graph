# coding: UTF-8

# region !import area!
# ========== python OS level ==========
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(os.path.join(PROJECT_DIR, 'src'))
# ========== python ==========
from logging import Logger
# noinspection PyUnresolvedReferences
from typing import List, Dict, Tuple, Optional, Union, Callable
# noinspection PyUnresolvedReferences
from tqdm import tqdm
from argparse import Namespace
# ========== Machine learning ==========
import numpy as np
import pandas as pd
import optuna
# ========== torch ==========
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
# ========== torch geometric ==========
from torch_geometric.data import Data as TorchGeoData
from torch_geometric.nn.conv import GATv2Conv
# from torch_geometric.loader import DataLoader, NeighborLoader
from torch_geometric.utils import add_self_loops
# ========== My Utils ==========
# noinspection PyUnresolvedReferences
from utils.utils import force_gc, force_gc_after_function, get_from_dict, version_check
from utils.str_process import info_str as _info_str
from utils.setup import setup, save_param
# from utilModules.torch import cuda_empty_cache as _ccr, load_model, save_model, decorate_loader, onehot
# ========== Made by me ==========
from models.datasets.data_helper import MyDataHelper, KGDATA_ALL


# endregion


def setup_parser() -> Namespace:
    import argparse  # 1. argparseをインポート
    parser = argparse.ArgumentParser(description='データの初期化')
    parser.add_argument('--logfile', help='ログファイルのパス', type=str)
    parser.add_argument('--param-file', help='パラメータを保存するファイルのパス', type=str)
    parser.add_argument('--console-level', help='コンソール出力のレベル', type=str, default='info')
    parser.add_argument('--no-show-bar', help='バーを表示しない', action='store_true')
    parser.add_argument('--device-name', help='cpu or cuda or mps', type=str, default='cpu',
                        choices=['cpu', 'cuda', 'mps'])
    # select function
    # parser.add_argument('--function', help='function', type=str, choices=['do_1train', 'do_optuna'])
    """
    # optuna setting
    parser.add_argument('--optuna-file', help='optuna file', type=str)
    parser.add_argument('--study-name', help='optuna study-name', type=str)
    parser.add_argument('--n-trials', help='optuna n-trials', type=int, default=20)
    """
    parser.add_argument('--KGdata', help=' or '.join(KGDATA_ALL), type=str, choices=KGDATA_ALL)

    """
    parser.add_argument('--model', type=str,
                        help='Choose from: {conve, distmult, complex, transformere}')
    parser.add_argument('--embedding-dim', type=int, default=200,
                        help='The embedding dimension (1D). Default: 200')
    parser.add_argument('--batch-size', help='batch size', type=int)
    
    parser.add_argument('--epoch', help='max epoch', type=int)

    parser.add_argument('--model-path', type=str, help='model path')
    parser.add_argument('--do-train', help='do-train', action='store_true')
    parser.add_argument('--do-valid', help='do-valid', action='store_true')
    parser.add_argument('--do-test', help='do-test', action='store_true')
    parser.add_argument('--valid-interval', type=int, default=1, help='valid-interval', )
    
    parser.add_argument('--embedding-shape1', type=int, default=20,
                        help='The first dimension of the reshaped 2D embedding. '
                             'The second dimension is inferred. Default: 20')
    # optimizer
    parser.add_argument('--lr', type=float, default=0.003, help='learning rate (default: 0.003)')
    parser.add_argument('--l2', type=float, default=0.0,
                        help='Weight decay value to use in the optimizer. Default: 0.0')
    parser.add_argument('--label-smoothing', type=float, default=0.1, help='Label smoothing value to use. Default: 0.1')
    # convE
    parser.add_argument('--hidden-drop', type=float, default=0.3, help='Dropout for the hidden layer. Default: 0.3.')
    parser.add_argument('--input-drop', type=float, default=0.2, help='Dropout for the input embeddings. Default: 0.2.')
    parser.add_argument('--feat-drop', type=float, default=0.2,
                        help='Dropout for the convolutional features. Default: 0.2.')
    parser.add_argument('--use-bias', action='store_true', help='Use a bias in the convolutional layer. Default: True')
    parser.add_argument('--lr-decay', type=float, default=0.995,
                        help='Decay the learning rate by this factor every epoch. Default: 0.995')
    parser.add_argument('--hidden-size', type=int, default=9728,
                        help='The side of the hidden layer. The required size changes with the size of the embeddings. '
                             'Default: 9728 (embedding size 200).')
    # transformere
    parser.add_argument('--nhead', type=int, default=8, help='nhead. Default: 8.')
    parser.add_argument('--transformer-drop', type=float, default=0.1, help='transformer-drop. Default: 0.1.')
    """
    # コマンドライン引数をパースして対応するハンドラ関数を実行
    _args = parser.parse_args()

    return _args


class GAT(torch.nn.Module):
    def __init__(self, args):
        super(GAT, self).__init__()
        embedding_dim = args.embedding_dim
        nhead1 = 8
        nhead2 = 8
        dropout_conv1 = 0.2
        dropout_conv2 = 0.2
        dropout_1 = 0.2
        dropout_2 = 0.2

        self.dropout_1 = nn.Dropout(dropout_1)
        self.dropout_2 = nn.Dropout(dropout_2)
        self.elu = nn.ELU()

        self.conv1 = GATv2Conv(embedding_dim, embedding_dim, heads=nhead1, dropout=dropout_conv1)
        self.conv2 = GATv2Conv(embedding_dim * nhead1, embedding_dim, concat=False, heads=nhead2, dropout=dropout_conv2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.dropout_1(x)
        x = self.conv1(x, edge_index)
        x = self.elu(x)
        x = self.dropout_2(x)
        x = self.conv2(x, edge_index)
        return x


def make_geodata(data_helper: MyDataHelper,
                 *, is_del_reverse=True, is_add_self_loop=False, num_self_loop=-1, logger=None) -> TorchGeoData:
    e_length = data_helper.processed_entity_num
    r_length = data_helper.processed_relation_num

    reverse_count = np.count_nonzero(data_helper.data.r_is_reverse_list)
    r_length = r_length - reverse_count if is_del_reverse else r_length

    # entity_special_num, relation_special_num = data_helper.get_er_special_num()
    graph_node_num, graph_edge_num = e_length, r_length

    if logger is not None:
        logger.debug(f"e_length: {e_length},\t graph_node_num: {graph_node_num}")
        logger.debug(f"r_length: {r_length},\t graph_edge_num: {graph_edge_num}")

    train_triple = data_helper.get_train_triple_dataset()
    src_, dst_, type_ = [], [], []
    for head, relation, tail, is_reverse, _, is_exist in tqdm(train_triple, leave=False, desc="make_geodata"):
        if (not is_exist) or (is_del_reverse and is_reverse):
            continue
        src_.append(head)
        dst_.append(tail)
        type_.append(relation)

    # make geo data
    # x = model.emb_e.weight.data
    x = torch.tensor([i for i in range(graph_node_num)])
    # node
    edge_index = torch.tensor([src_, dst_], requires_grad=False)
    # edge
    edge_attr = torch.tensor(type_)
    if is_add_self_loop:
        assert num_self_loop >= 0, "can't make self loop relation"
        edge_index, edge_weight = add_self_loops(edge_index, edge_attr, fill_value=torch.tensor(num_self_loop))
    # make torch geometric data
    geo_data = TorchGeoData(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return geo_data


def separate_triples(geo_data: TorchGeoData, padding_value, self_loop_value, *, logger=None):
    if logger is not None: logger.info(f"is_directed: {geo_data.is_directed()}")
    src_, dst_ = geo_data.edge_index
    edge_attr = geo_data.edge_attr

    def _func(value) -> torch.Tensor:
        node2x_dict = {x.item(): value[src_ == x] for x in geo_data.x}
        node2x_list = [node2x_dict[i][:, None] for i in range(len(geo_data.x))]
        node2x_tensor = pad_sequence(node2x_list, batch_first=True, padding_value=padding_value)
        node2x_tensor = node2x_tensor.view(len(geo_data.x), -1)
        return node2x_tensor

    node2neighbor_node, node2neighbor_attr = map(_func, (dst_, edge_attr))
    if self_loop_value is not None:
        self_loop_node = torch.tensor([[i] for i in range(len(node2neighbor_node))])
        self_loop_attr = torch.full_like(self_loop_node, self_loop_value)
        node2neighbor_node = torch.cat((node2neighbor_node, self_loop_node), dim=1)
        node2neighbor_attr = torch.cat((node2neighbor_attr, self_loop_attr), dim=1)

    assert node2neighbor_node.shape == node2neighbor_attr.shape
    if logger is not None: logger.debug(f"node2neighbor_node shape: {node2neighbor_node.shape}")
    return node2neighbor_node, node2neighbor_attr


"""
def make_pre_emb(args, geo_data: TorchGeoData, model):
    # make GAT
    gat = GAT(args=args)
    x = gat(geo_data)
    # train
    model.model.emb_e.weight.data = x
    return geo_data
"""


@force_gc_after_function
def do_1train(args, *, logger: Logger):
    kg_data = args.KGdata
    # do_train, do_valid, do_test = args.do_train, args.do_valid, args.do_test
    # model_path = args.model_path
    # batch_size = args.batch_size
    device = args.device
    # no_show_bar = args.no_show_bar

    # if (not do_train) and (not do_valid) and (not do_test):
    #     return -1

    logger.info(f"Function start".center(40, '='))

    # padding entity is 0
    # self loop tensor is 1
    # so special token num is 2

    padding_token = 0
    self_loop_token = 1
    special_token_num = 2

    # load data
    logger.info(_info_str(f"load data start."))
    data_helper = None
    # load_preprocess_data(kg_data, eco_memory=True,
    # logger=logger,entity_special_num=special_token_num, relation_special_num=special_token_num)
    logger.info(_info_str(f"load data complete."))
    data_helper.show_log(logger)

    # make data
    logger.info(_info_str(f"make PyG data start."))
    geo_data = make_geodata(data_helper, is_del_reverse=False,
                            is_add_self_loop=True, self_loop_weight=self_loop_token, logger=logger)
    # check_graph(geo_data, logger=logger)

    # check_graph(geo_data, logger=logger)
    node2neighbor_node, node2neighbor_attr = separate_triples(geo_data, padding_value=padding_token, logger=logger)
    logger.info(_info_str(f"make PyG data complete."))


def main():
    args, logger, device = setup(setup_parser, PROJECT_DIR)
    version_check(torch, np, pd, optuna, logger=logger)
    try:
        args.project_dir = PROJECT_DIR
        args.logger = logger
        args.device = device
        args.completed = {}
        logger.debug(vars(args))
        logger.debug(f"process id = {args.pid}")
        do_1train(args, logger=logger)

    finally:
        save_param(args)


if __name__ == '__main__':
    main()
