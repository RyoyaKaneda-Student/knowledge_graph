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


def make_g(usetitle):
    g = Graph()
    g.parse(f'KGdata/external/2020v2/{usetitle}.ttl')
    g.bind(usetitle, f"http://kgc.knowledge-graph.jp/data/{usetitle}/", override=True)
    prefix_dict = {item[0]: Namespace(item[1]) for item in g.namespaces()}
    prefix_dict = AttrDict(prefix_dict)
    return g, prefix_dict


def change_name(g, s, p, o):
    s, p, o = s.n3(g.namespace_manager), p.n3(g.namespace_manager), o.n3(g.namespace_manager)
    result = re.findall(fr"<http://kgc.knowledge-graph.jp/data/predicate/(.*)>", s)
    if len(result) > 0:
        s = re.sub(r'([A-Z])', lambda x: "_" + x.group(1).lower(), result[0])
        if s[0] == '_':
            s = s[1:]
    if re.fullmatch(r'.*:Holmes', s) is not None:
        s = ':Holmes'
    if re.fullmatch(r'.*:Watson', s) is not None:
        s = ':Watson'

    return s, p, o


def make_ERdicts(g, Edict=None, Rdict=None):
    if Edict is None:
        Edict = {}
    if Rdict is None:
        Rdict = {}
    Eflist, Rlist, Etlist = [], [], []

    for s, p, o in g.triples((None, None, None)):
        s, p, o = change_name(g, s, p, o)
        Eflist.append(s)
        Rlist.append(p)
        Etlist.append(o)

    Elist = Eflist + Etlist
    # 重複消去
    Elist = sorted(set(Elist), key=Elist.index)
    Rlist = sorted(set(Rlist), key=Rlist.index)

    Elist = [l for l in Elist if l not in Edict.keys()]
    Rlist = [l for l in Rlist if l not in Rdict.keys()]
    # id付
    lenEdict, lenRdict = len(Edict), len(Rdict)

    Edict.update({l: i + lenEdict for i, l in enumerate(Elist)})
    Rdict.update({l: i + lenRdict for i, l in enumerate(Rlist)})

    return Edict, Rdict


def make_Predicate(g, Edict=None, Rdict=None):
    pass
    for s, p, o in g.triples((None, None, None)):
        pass


def make_triple(g, Edict, Rdict):
    # triple に関する処理
    triples = []
    for s, p, o in g.triples((None, None, None)):
        s, p, o = change_name(g, s, p, o)
        _s, _p, _o = Edict[s], Rdict[p], Edict[o]

        triples.append((_s, _p, _o))

    return triples


def make_souce_label(g, Edict, Rdict):
    # source, label に関する処理
    sources = {}
    labels = {}
    for s, p, o in g.triples((None, None, None)):
        s, p, o = change_name(g, s, p, o)
        _s, _p, _o = Edict[s], Rdict[p], Edict[o]
        if p == "kgc:source" and '@en' in o:
            sources[_o] = o
        elif p == "rdfs:label":
            labels[_o] = o

    return sources, labels


def make_story_list(g, usetitle, Edict, Rdict):
    # Story に関する処理
    stories = []
    s_taiou = {}  # [s] = (new s)
    triples = []

    for s, p, o in g.triples((None, RDF.type, None)):
        _s, _p, _o = change_name(g, s, p, o)
        if _o in ("kgc:Situation", "kgc:Statement", "kgc:Thought"):
            # show_all_s(g, s, Edict, Rdict)  # print
            result = re.findall(fr'({usetitle}:)(\d+)([a-z]?)', _s)
            if len(result) == 0:
                continue
            result = result[0]
            s_ = result[0] + result[1].zfill(4) + result[2]
            s_taiou[s] = s_
            stories.append(s_)
            subject, hasPredicate, what = None, None, None
            for _s, _p, _o in g.triples((s, None, None)):
                _s, _p, _o = change_name(g, _s, _p, _o)
                if _p == "kgc:subject":
                    subject = _o
                if _p == "kgc:hasPredicate":
                    hasPredicate = _o
                if _p == "kgc:what":
                    what = _o

            # print((subject, hasPredicate, what))
            if None not in (subject, hasPredicate, what):
                if hasPredicate not in Rdict.keys():
                    Rdict[hasPredicate] = len(Rdict)
                triples.append((subject, hasPredicate, what))

    stories = sorted(stories)

    return stories, s_taiou, triples


def show_all_s(g, s, Edict: dict, Rdict: dict):
    for s, p, o in g.triples((s, None, None)):
        _s, _p, _o = change_name(g, s, p, o)
        print("show SPO.", _s, _p, _o)
        _s = Edict[_s] if _s in Edict.keys() else '_'
        _p = Rdict[_p] if _p in Rdict.keys() else '_'
        _o = Edict[_o] if _o in Edict.keys() else '_'
        # print("show SPO IDs.", _s, _p, _o)


def function01(usetitle):
    g, prefix_dict = make_g(usetitle)

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
    revR = {}
    for key, value in R.items():
        revR[value] = key

    # triple に関する処理
    triples = []
    tmp01, tmp02 = {}, {}
    for s, p, o in g.triples((None, None, None)):
        s, p, o = change_name(g, s, p, o)
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
        s, p, o = change_name(g, s, p, o)
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
    # print(stories)

    df = pd.DataFrame(triples)
    df.to_csv(f"KGdata/processed/{usetitle}/ids.tsv", header=False, index=False, sep='\t')
    with open(f"KGdata/processed/{usetitle}/to_skip.pickle", "wb") as f:
        pickle.dump({'lhs': tmp01, 'rhs': tmp02}, f)
    with open(f"KGdata/processed/{usetitle}/train.pickle", "wb") as f:
        pickle.dump(df.values, f)
    with open(f"KGdata/processed/{usetitle}/valid.pickle", "wb") as f:
        pickle.dump(df.values, f)
    with open(f"KGdata/processed/{usetitle}/test.pickle", "wb") as f:
        pickle.dump(df.values, f)

    return Ef, R, Et, revE, revR, triples


def function02(usetitle):
    g, prefix_dict = make_g(usetitle)

    Edict, Rdict = make_ERdicts(g)

    triples = make_triple(g, Edict, Rdict)

    tmp01, tmp02 = {}, {}
    for s in range(max(Edict.values()) + 1):
        for p in range((max(Rdict.values()) + 1) * 2):
            tmp01[(s, p)] = []
            tmp02[(s, p)] = []

    # source, label に関する処理
    sources, labels = make_souce_label(g, Edict, Rdict)

    stories, s_taiou = make_story_list(g, usetitle)

    df = pd.DataFrame(triples)
    df.to_csv(f"KGdata/processed/{usetitle}/ids.tsv", header=False, index=False, sep='\t')
    with open(f"KGdata/processed/{usetitle}/to_skip.pickle", "wb") as f:
        pickle.dump({'lhs': tmp01, 'rhs': tmp02}, f)
    with open(f"KGdata/processed/{usetitle}/train.pickle", "wb") as f:
        pickle.dump(df.values, f)
    with open(f"KGdata/processed/{usetitle}/valid.pickle", "wb") as f:
        pickle.dump(df.values, f)
    with open(f"KGdata/processed/{usetitle}/test.pickle", "wb") as f:
        pickle.dump(df.values, f)


def function03(titles):
    Edict, Rdict = {}, {}
    triples = []
    for jaName, tagName in titles.items():
        g, prefix_dict = make_g(tagName)

        Edict, Rdict = make_ERdicts(g, Edict, Rdict)

        _triples = make_triple(g, Edict, Rdict)
        triples.extend(_triples)

        # source, label に関する処理
        sources, labels = make_souce_label(g, Edict, Rdict)

        stories, s_taiou, additional_triples = make_story_list(g, tagName, Edict, Rdict)
        _triples.extend(additional_triples)

    args = {}
    args['Edict'] = Edict
    args['Rdict'] = Rdict
    args['triples'] = triples
    args['df'] = pd.DataFrame(triples)

    return args


def main2():
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

    dict_ = {}
    for jaName, tagName in titles.items():
        dict_[tagName] = function01(tagName)
    print("complete")


def main3():
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

    args = function03(titles)
    df = args['df']
    Edict = args['Edict']
    Rdict = args['Rdict']
    df.to_csv(f"KGdata/processed/ALL/ids.tsv", header=False, index=False, sep='\t')

    df.to_csv(f"KGdata/processed/ALL/train", header=False, index=False, sep='\t')
    df.to_csv(f"KGdata/processed/ALL/valid", header=False, index=False, sep='\t')

    print(Edict.keys())

    kill_id = Rdict['<http://kgc.knowledge-graph.jp/data/predicate/kill>']
    hide_id = Rdict['<http://kgc.knowledge-graph.jp/data/predicate/hide>']

    test = (
        (Edict['SpeckledBand:Roylott'], kill_id, Edict['SpeckledBand:Julia']),
        (Edict['DevilsFoot:Mortimer'], kill_id, Edict['DevilsFoot:Brenda']),
        (Edict['DevilsFoot:Standale'], kill_id, Edict['DevilsFoot:Mortimer']),
        (Edict['ACaseOfIdentity:Windybank'], hide_id, Edict['ACaseOfIdentity:hozma'])
    )

    test = np.array(test)

    df = pd.DataFrame(test)
    df.to_csv(f"KGdata/processed/ALL/test", header=False, index=False, sep='\t')

    print("complete")


def main4():
    from sklearn.model_selection import train_test_split
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

    args = function03(titles)
    df: pd.DataFrame = args['df']
    Edict = args['Edict']
    Rdict = args['Rdict']
    train, _test = train_test_split(df.to_numpy(), test_size=0.1)
    _train, _valid = train_test_split(train, test_size=0.2)
    tmpEdict = {value: False for key, value in Edict.items()}
    tmpRdict = {value: False for key, value in Rdict.items()}
    for t in _train:
        e1, r, e2 = t.tolist()
        print(e1, r, e2)
        tmpEdict[e1] = True
        tmpRdict[r] = True
        tmpEdict[e2] = True

    lenRdict = len(Rdict)
    Rdict['special:is'] = lenRdict
    for key, value in tmpEdict.items():
        if not value:
            df.append([key, lenRdict, key])

    df_train = pd.DataFrame(_train)
    df_valid = pd.DataFrame(_valid)
    df__test = pd.DataFrame(_test)

    df.to_csv(f"KGdata/processed/ALL2/ids.tsv", header=False, index=False, sep='\t')

    df_train.to_csv(f"KGdata/processed/ALL2/train", header=False, index=False, sep='\t')
    df_valid.to_csv(f"KGdata/processed/ALL2/valid", header=False, index=False, sep='\t')
    df__test.to_csv(f"KGdata/processed/ALL2/test", header=False, index=False, sep='\t')


def main():
    main4()


if __name__ == '__main__':
    main()
    pass
