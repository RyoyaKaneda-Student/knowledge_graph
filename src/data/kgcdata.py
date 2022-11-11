#!/usr/bin/python
# -*- coding: utf-8 -*-
# ========== python ==========
from logging import Logger
from pathlib import Path
# noinspection PyUnresolvedReferences
from typing import List, Dict, Tuple, Optional, Union, Callable, Final, Literal, get_args
import h5py
import numpy as np
from rdflib import Graph, URIRef, Literal, Namespace, RDF, RDFS
from rdflib.namespace import DefinedNamespace

from utils.setup import easy_logger

PROJECT_DIR = Path(__file__).resolve().parents[2]
print(f"{PROJECT_DIR=}")

URL_DATA: Final = "http://kgc.knowledge-graph.jp/data"
URL_PREDICATE: Final = "http://kgc.knowledge-graph.jp/data/predicate"
DATA_FOLDER_PATH: Final = f"{PROJECT_DIR}/../KGCdata/KGRC-RDF"

PREFIX_PREDICATE: Final = 'word:predicate'

KGC_TYPE: Final = 'kgc:type'
KGC_LABEL: Final = 'kgc:label'

logger = easy_logger(console_level='info')


class KGC(DefinedNamespace):
    source: URIRef
    ActionOption: URIRef
    Not: URIRef
    RelationBetweenScene: URIRef
    SceneObjectProperty: URIRef
    SceneProperty: URIRef
    TargetObjProperty: URIRef
    adjunct: URIRef
    around: URIRef
    at_the_same_time: URIRef
    because: URIRef
    can: URIRef
    canNot: URIRef
    # from: URIRef
    hasPart: URIRef
    hasPredicate: URIRef
    hasProperty: URIRef
    how: URIRef
    # if: URIRef
    infoReceiver: URIRef
    infoSource: URIRef
    left: URIRef
    middle: URIRef
    near: URIRef
    next_to: URIRef
    ofPart: URIRef
    ofWhole: URIRef
    on: URIRef
    opposite: URIRef
    orTarget: URIRef
    otherwise: URIRef
    right: URIRef
    subject: URIRef
    then: URIRef
    therefore: URIRef
    to: URIRef
    what: URIRef
    when: URIRef
    when_during: URIRef
    where: URIRef
    whom: URIRef
    why: URIRef
    LocationProperty: URIRef
    time: URIRef
    AbstractTime: URIRef
    Action: URIRef
    Animal: URIRef
    CanAction: URIRef
    CanNotAction: URIRef
    NotAction: URIRef
    OFobj: URIRef
    ORobj: URIRef
    Object: URIRef
    PhysicalObject: URIRef
    Person: URIRef
    Place: URIRef
    Property: URIRef
    Scene: URIRef
    Situation: URIRef
    Statement: URIRef
    Talk: URIRef
    Thought: URIRef

    _NS = Namespace("http://kgc.knowledge-graph.jp/ontology/kgc.owl#")


title_len_dict: Final = {
    "僧坊荘園": ("AbbeyGrange", 414, 372, 331),
    "花婿失踪事件": ("ACaseOfIdentity", 580, 522, 464),
    "背中の曲がった男": ("CrookedMan", 373, 335, 298),
    "踊る人形": ("DancingMen", 231, 207, 184),
    "悪魔の足": ("DevilsFoot", 489, 440, 391),
    "入院患者": ("ResidentPatient", 324, 291, 259),
    "白銀号事件": ("SilverBlaze", 397, 367, 317),
    "マダラのひも": ("SpeckledBand", 401, 360, 320)
}

OBJECT_SET = {
    KGC.Animal, KGC.OFobj, KGC.ORobj, KGC.Object,
    KGC.PhysicalObject, KGC.Person, KGC.Place,
}

ACTION_SET = {
    KGC.Action, KGC.CanAction, KGC.CanNotAction, KGC.NotAction
}


def get_type_match_people(graph_: Graph, type_set: set[URIRef]):
    match_set = {s for s, _, o in graph_.triples((None, None, None))
                 if o in type_set} - type_set
    return match_set


def make_graph(title: str, data_file_path: Union[str, Path]) -> tuple[Graph, dict[str, Namespace]]:
    g = Graph()
    g.parse(data_file_path)
    g.bind(PREFIX_PREDICATE, Namespace(URL_PREDICATE))
    g.bind(title, Namespace(f"{URL_DATA}/{title}/"))
    namespace_dict = {key: Namespace(value) for key, value in g.namespace_manager.namespaces()}
    return g, namespace_dict


def get_story_list(graph_, namespace_dict, title, l_):
    story_list = []

    prefix_title = namespace_dict[title]
    for i in range(1, l_ + 1):
        iii = str(i).zfill(3)
        for s in ['', 'a', 'b', 'c']:
            iiis = iii + s
            story_ = prefix_title[iiis]
            tmp = list(graph_.triples((story_, RDF.type, None)))
            if len(tmp) == 1:
                story_list.append(iiis)
    return story_list


def write_():
    print(f"{DATA_FOLDER_PATH}/../data/story_list.hdf5")
    with h5py.File(f"{DATA_FOLDER_PATH}/../data/story_list.hdf5", 'a') as f:
        for title_ja, (title, l10, l09, l08) in title_len_dict.items():
            data_file_path = f"{DATA_FOLDER_PATH}/{title}.ttl"
            graph_, namespace_dict = make_graph(title, data_file_path)

            story_list = get_story_list(graph_, namespace_dict, title, l10)
            object_set = get_type_match_people(graph_, OBJECT_SET)
            actions_set = get_type_match_people(graph_, ACTION_SET)
            people_set = get_type_match_people(graph_, {KGC.Person})
            ofobj_set = get_type_match_people(graph_, {KGC.OFobj})
            print(people_set-ofobj_set)
            # f[title] = np.array(story_list, dtype=h5py.special_dtype(vlen=str))


def read_():
    with h5py.File(f"{DATA_FOLDER_PATH}/../data/story_list.hdf5", 'r') as f:
        for title_ja, (title, l10, l09, l08) in title_len_dict.items():
            logger.info(f[title][:])


def main():
    write_()


if __name__ == '__main__':
    main()
