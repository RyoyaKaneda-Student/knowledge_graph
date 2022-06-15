import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import re

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


def function01(usetitle):
    g = Graph()
    g.parse(f'data/external/2020v2/{usetitle}.ttl')
    g.bind(usetitle, f"http://kgc.knowledge-graph.jp/data/{usetitle}/", override=True)

    def change_name(s, p, o):
        return s.n3(g.namespace_manager), p.n3(g.namespace_manager), o.n3(g.namespace_manager)

    prefix_dict = {item[0]: Namespace(item[1]) for item in g.namespaces()}
    prefix_dict = AttrDict(prefix_dict)

    Ef, R, Et = {}, {}, {}

    R["kgc:source"] = 0
    R["rdfs:label"] = 0

    for s, p, o in g.triples((None, None, None)):
        s, p, o = change_name(s, p, o)
        if s not in Ef:
            Ef[s] = len(Ef)
        if p not in R:
            R[p] = len(R) - 2
        if o not in Et:
            Et[o] = len(Et)

    # EtとEfで同じものはまとめる
    lenEf = len(Ef)
    for key, value in Et.items():
        Et[key] = Ef.get(key, value + lenEf)

    # 逆対応dict
    revE = {}
    tmp = list(set(list(Ef.items()) + list(Et.items())))
    for key, value in sorted(tmp, key=lambda x: x[0]):
        revE[value] = key

    # triple に関する処理
    triples = []
    tmp01, tmp02 = {}, {}
    for s, p, o in g.triples((None, None, None)):
        s, p, o = s.n3(g.namespace_manager), p.n3(g.namespace_manager), o.n3(g.namespace_manager)
        _s, _p, _o = Ef[s], R[p], Et[o]

        triples.append((_s, _p, _o))

    for s in range(max(Et.values()) + 1):
        for p in range((max(R.values()) + 1) * 2):
            tmp01[(s, p)] = []
            tmp02[(s, p)] = []

    # source, label に関する処理
    sources = {}
    labels = {}
    for s, p, o in g.triples((None, None, None)):
        s, p, o = s.n3(g.namespace_manager), p.n3(g.namespace_manager), o.n3(g.namespace_manager)
        _s, _p, _o = Ef[s], R[p], Et[o]
        if p == "kgc:source" and '@en' in o:
            sources[_o] = o
        elif p == "rdfs:label":
            labels[_o] = o

    # Story に関する処理
    stories = []
    s_taiou = {}  # [index] = (new s)
    for s, p, o in g.triples((None, RDF.type, None)):
        s, p, o = change_name(s, p, o)
        if o in ("kgc:Situation", "kgc:Statement", "kgc:Thought"):
            # print(s, p, o)
            result = re.findall(fr'({usetitle}:)(\d+)([a-z]?)', s)
            if len(result) == 0:
                continue
            result = result[0]
            s_ = result[0] + result[1].zfill(4) + result[2]
            s_taiou[Ef[s]] = s_
            stories.append(s_)

    stories = sorted(stories)
    print(stories)

    df = pd.DataFrame(triples)
    df.to_csv(f"data/processed/{usetitle}/ids.tsv", header=False, index=False, sep='\t')
    with open(f"data/processed/{usetitle}/to_skip.pickle", "wb") as f:
        pickle.dump({'lhs': tmp01, 'rhs': tmp02}, f)
    with open(f"data/processed/{usetitle}/train.pickle", "wb") as f:
        pickle.dump(df.values, f)
    with open(f"data/processed/{usetitle}/valid.pickle", "wb") as f:
        pickle.dump(df.values, f)
    with open(f"data/processed/{usetitle}/test.pickle", "wb") as f:
        pickle.dump(df.values, f)


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
    for jaName, tagName in titles.items():
        function01(tagName)
    print("complete")


if __name__ == '__main__':
    main()
    pass
