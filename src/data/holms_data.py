#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path

from typing import List
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import rdflib
import torch

# ラベルエンコーディング（LabelEncoder）
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data as PyG_Data
from torch_geometric.utils import to_networkx

from rdflib import Graph, RDF, Namespace, Literal, FOAF
from rdflib.term import Node

import torch.nn.functional as F
from torch_geometric.nn import GCNConv

PROJECT_DIR = Path(__file__).resolve().parents[2]


def is_integer(n):
    try:
        float(n)
    except ValueError:
        return False
    else:
        return float(n).is_integer()


def make_graph(title: str):
    g = Graph()
    g.parse(f'{PROJECT_DIR}/data/external/2020v2/{title}.ttl')
    g.bind(title, f"http://kgc.knowledge-graph.jp/data/{title}/", override=True)
    return g


def make_df(g: Graph) -> pd.DataFrame:
    SPO = []
    for s, p, o in g.triples((None, None, None)):
        s = s.n3(g.namespace_manager)
        p = p.n3(g.namespace_manager)
        o = o.n3(g.namespace_manager)
        type_s, type_o = None, None
        for _, _, _o in g.triples((s, rdflib.RDF.type, None)):
            type_s = _o.n3(g.namespace_manager)
        for _, _, _o in g.triples((o, rdflib.RDF.type, None)):
            type_o = _o.n3(g.namespace_manager)

        if p != "kgc:source":
            SPO.append([s, p, o, type_s, type_o])
    df = pd.DataFrame(SPO, columns=["S", "P", "O", "TypeS", "TypeO"])
    return df


def encode_SPO(df: pd.DataFrame) -> (pd.DataFrame, List[str]):
    # SとOをつなげてラベル付けする.
    df = df.copy(deep=False)
    df_S = df.loc[:, ['S', "TypeS"]]
    df_S["isS"] = True
    df_S.columns = ['x', 'type_', 'isS']
    df_O = df.loc[:, ['O', "TypeO"]]
    df_O["isS"] = False
    df_O.columns = ['x', 'type_', 'isS']
    df_SO = pd.concat([df_S, df_O])

    le_SO = LabelEncoder()
    encoded_SO = le_SO.fit_transform(df_SO['x'])
    encoded_S = encoded_SO[df_SO['isS']]
    encoded_O = encoded_SO[~df_SO['isS']]

    le_typeSO = LabelEncoder()
    encoded_typeSO = le_typeSO.fit_transform(df_SO['type_'])
    encoded_typeS = encoded_typeSO[df_SO['isS']]
    encoded_typeO = encoded_typeSO[~df_SO['isS']]

    le_P = LabelEncoder()
    encoded_P = le_P.fit_transform(df['P'])

    df['encoded_S'] = encoded_S
    df['encoded_P'] = encoded_P
    df['encoded_O'] = encoded_O
    df['encoded_typeS'] = encoded_typeS
    df['encoded_typeO'] = encoded_typeO
    return df, le_SO.classes_, le_P.classes_, le_typeSO.classes_


def make_PyG(df: pd.DataFrame, classes_: List[str]) -> PyG_Data:
    encoded_S = df['encoded_S']
    encoded_P = df['encoded_P']
    encoded_O = df['encoded_O']
    edge_index = torch.tensor([encoded_S.tolist(), encoded_O.tolist()], dtype=torch.long)
    edge_attr = torch.tensor(encoded_P, dtype=torch.long)

    x = classes_
    y = torch.tensor([0 for _ in x], dtype=torch.long)
    data = PyG_Data(x=x, y=y, num_nodes=len(classes_),
                    edge_index=edge_index, edge_attr=edge_attr
                    )
    return data


def check_graph(data: PyG_Data) -> None:
    print("グラフ構造:", data)
    print("グラフのキー: ", data.keys)
    print("ノード数:", data.num_nodes)
    print("エッジ数:", data.num_edges)
    print("エッジに関して", data.edge_attr)
    print("孤立したノードの有無:", data.has_isolated_nodes())
    print("自己ループの有無:", data.has_self_loops())
    print("====== ノードの特徴量:x ======")
    print(data['x'])
    print("========= エッジ形状 =========")
    print(data['edge_index'])


def check_graph2(data: PyG_Data) -> None:
    print("ノード数:", data.num_nodes)
    print("エッジ数:", data.num_edges)


def show_data(title, data: PyG_Data) -> None:
    # networkxのグラフに変換
    nxg = to_networkx(
        data,
    )
    # print(nxg.degree())
    mapping = {k: v for k, v in zip(nxg.nodes, data['x'])}
    # nxg = nx.relabel_nodes(nxg, mapping)

    # 可視化のためのページランク計算
    pr = nx.pagerank(nxg)
    pr_max = np.array(list(pr.values())).max()

    # 可視化する際のノード位置
    draw_pos = nx.spring_layout(nxg, seed=0)

    plt.figure()
    # ノードの色設定
    cmap = plt.get_cmap('tab10')
    labels = data.y.numpy()
    colors = [cmap(l) for l in labels]

    # 図のサイズ
    plt.figure(figsize=(10, 10))

    # 描画
    nx.draw_networkx_nodes(nxg,
                           draw_pos,
                           node_size=[v / pr_max * 1000 for v in pr.values()],
                           node_color=colors, alpha=0.5,
                           )
    nx.draw_networkx_edges(nxg, draw_pos, arrowstyle='-', alpha=0.2)
    nx.draw_networkx_labels(nxg, draw_pos, font_size=10)

    plt.title(title)
    plt.show()


def function(title, executor):
    g = make_graph(title)
    df = make_df(g)
    df, classes_, _, _ = encode_SPO(df)
    data = make_PyG(df, classes_)
    check_graph2(data)
    executor.submit(show_data, title, data)


def main():
    titles = {
        "僧坊荘園": "AbbeyGrange",
        "花婿失踪事件": "ACaseOfIdentity",
        "背中の曲がった男": "CrookedMan",
        "踊る人形": "DancingMen",
        "悪魔の足": "DevilsFoot",
        "入院患者": "ResidentPatient",
        "白銀号事件": "SilverBlaze",
        "マダラのひも": "SpeckledBand"
    }
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="thread") as executor:
        for jtitle, title in titles.items():
            print(jtitle, title)
            function(title, executor)


    print("END MAIN")


if __name__ == '__main__':
    main()
