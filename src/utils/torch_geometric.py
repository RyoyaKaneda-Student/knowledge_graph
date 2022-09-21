# torch
import torch
from torch.utils.data import Dataset
from torch import nn
from torch.utils.data.dataloader import DataLoader

from torch_geometric.data import Data

import torch.nn.functional as F


def check_graph(data: Data, *, logger):
    logger.debug(f"グラフ構造:{data}")
    logger.debug(f"グラフのキー: {data.keys}")
    logger.debug(f"ノード数: {data.num_nodes}")
    logger.debug(f"エッジ数: {data.num_edges}")
    logger.debug(f"エッジタイプ数: {data.edge_attr[0].shape[0]}")
    logger.debug(f"ノードの特徴量数: {data.num_node_features}")
    logger.debug(f"孤立したノードの有無: {data.has_isolated_nodes()}")
    if data.has_isolated_nodes():
        nodes = torch.tensor([True for _ in range(data.num_nodes)])
        src_, dst_ = data.edge_index
        for n1, n2 in zip(src_, dst_):
            nodes[n1] = False
            nodes[n2] = False
        logger.debug(torch.nonzero(nodes, as_tuple=True)[0].tolist())
    logger.debug(f"自己ループの有無: {data.has_self_loops()}")
    if data.has_self_loops():
        tmp_list = []
        src_, dst_ = data.edge_index
        [tmp_list.append(n1.item()) for n1, n2 in zip(src_, dst_) if n1 == n2]
        logger.debug(tmp_list)
        del tmp_list
