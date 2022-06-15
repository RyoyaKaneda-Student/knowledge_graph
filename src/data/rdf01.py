import numpy as np
import matplotlib.pyplot as plt

from rdflib import Graph, RDF, Namespace, Literal, FOAF

from attrdict import AttrDict
from rdflib.paths import Path

from sklearn.manifold import TSNE
from sklearn.preprocessing import OneHotEncoder


def is_integer(n):
    try:
        float(n)
    except ValueError:
        return False
    else:
        return float(n).is_integer()


usetitle = 'SpeckledBand'
g = Graph()
g.parse(f'data/external/2020v2/{usetitle}.ttl')
g.bind(usetitle, f"http://kgc.knowledge-graph.jp/data/{usetitle}/", override=True)

prefix_dict = {item[0]: Namespace(item[1]) for item in g.namespaces()}
prefix_dict = AttrDict(prefix_dict)

situation = [s.n3(g.namespace_manager).replace(f"{usetitle}:", '')
             for s, p, o in g.triples((None, RDF.type, prefix_dict.kgc.Situation))]
situation = [int(n) for n in situation if is_integer(n)]

situation_len = max(situation)
situation = [1 if n in situation else 0 for n in range(situation_len + 1)]

situation_matrix = [[0 for _ in range(situation_len + 1)] for _ in range(situation_len + 1)]

type_list = []


def type_to_index(type_):
    if type_ not in type_list:
        type_list.append(type_)
    return type_list.index(type_)


for s1, p1, o1 in g.triples((None, RDF.type, prefix_dict.kgc.Situation)):
    for s2, p2, o2 in g.triples((None, RDF.type, prefix_dict.kgc.Situation)):
        for s, p, o in g.triples((s1, None, s2)):
            s = s.n3(g.namespace_manager).replace(f"{usetitle}:", '')
            s = int(s) if is_integer(s) else 0
            o = o.n3(g.namespace_manager).replace(f"{usetitle}:", '')
            o = int(o) if is_integer(o) else 0
            if not s * o == 0:
                situation_matrix[s][o] = type_to_index(p.n3(g.namespace_manager))

# print(type_list)
# print(situation_matrix)


for s, _, _ in g.triples((None, RDF.type, prefix_dict.kgc.Situation)):
    for s, p, o in g.triples((s, prefix_dict.kgc.source, None)):
        s = s.n3(g.namespace_manager)
        p = p.n3(g.namespace_manager)
        if o.language == 'ja':
            # print(s, p, o)
            pass

d = {}
for _, p, _ in g.triples((None, None, None)):
    p = p.n3(g.namespace_manager)
    d[p] = d.get(p, 0) + 1

list_ = [p[0] for p in sorted(d.items(), key=lambda x: -x[1])]
print(list_)

SPO = []
for s, p, o in g.triples((None, None, None)):
    s = s.n3(g.namespace_manager)
    p = p.n3(g.namespace_manager)
    o = o.n3(g.namespace_manager)
    if p != "kgc:source":
        SPO.append([s, p, o])

S, P, O = zip(*SPO)
print([len(item) for item in (S, P, O)])

import pandas as pd
import numpy as np

df = pd.DataFrame(SPO, columns=["S", "P", "O"])
print(df)

# ラベルエンコーディング（LabelEncoder）
from sklearn.preprocessing import LabelEncoder
import torch
from torch_geometric.data import Data as PyG_Data

dfS = df.loc[:, ['S']]
dfS["isS"] = True
dfS.columns = ['x', 'isS']
dfO = df.loc[:, ['O']]
dfO["isS"] = False
dfO.columns = ['x', 'isS']
dfSO = pd.concat([dfS, dfO])

le_SO = LabelEncoder()
encoded_SO = le_SO.fit_transform(dfSO['x'])
print(le_SO.classes_)
encoded_S = encoded_SO[dfSO['isS']]
encoded_O = encoded_SO[~dfSO['isS']]
df['encoded_S'] = encoded_S
df['encoded_O'] = encoded_O

le_P = LabelEncoder()
encoded_P = le_P.fit_transform(df['P'])
df['encoded_P'] = encoded_P

edge_index = torch.tensor([encoded_S.tolist(), encoded_O.tolist()], dtype=torch.long)
edge_attr = torch.tensor(encoded_P, dtype=torch.long)

ddf = pd.get_dummies(df.loc[:, ["S"]])
co = ddf.columns
# print(co)

x = le_SO.classes_
y = torch.tensor([0 for _ in x], dtype=torch.long)
data = PyG_Data(x=x, y=y, num_nodes=len(le_SO.classes_), edge_index=edge_index, edge_attr=edge_attr, )


def check_graph(data):
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
    print(data)


check_graph(data)

import torch
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.datasets import KarateClub
# import torch_geometric.transforms as T
from torch_geometric.utils import to_networkx

import networkx as nx
from matplotlib import pyplot as plt
import numpy as np

# networkxのグラフに変換
nxg = to_networkx(data)

# 可視化のためのページランク計算
pr = nx.pagerank(nxg)
pr_max = np.array(list(pr.values())).max()

# 可視化する際のノード位置
draw_pos = nx.spring_layout(nxg, seed=0)

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
                       node_color=colors, alpha=0.5)
nx.draw_networkx_edges(nxg, draw_pos, arrowstyle='-', alpha=0.2)
nx.draw_networkx_labels(nxg, draw_pos, font_size=10)

plt.title('KarateClub')
plt.show()
