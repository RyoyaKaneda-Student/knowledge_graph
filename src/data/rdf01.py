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
ddf = pd.get_dummies(df)
co = ddf.columns
print(ddf.loc[:, co[co.str.contains('S_')]])
