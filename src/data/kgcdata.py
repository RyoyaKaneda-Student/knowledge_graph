#!/usr/bin/python
# -*- coding: utf-8 -*-
# ========== python ==========
# noinspection PyUnresolvedReferences
from logging import Logger
from pathlib import Path
from urllib.parse import urlparse
# noinspection PyUnresolvedReferences
from typing import List, Dict, Tuple, Optional, Union, Callable, Final, Literal, Iterable, get_args, cast
import h5py
# noinspection PyUnresolvedReferences
from h5py import Group
import numpy as np
# noinspection PyUnresolvedReferences
from rdflib import Graph, URIRef, Literal, Namespace, RDF, RDFS
from rdflib.namespace import DefinedNamespace
from rdflib.term import Node

from utils.hdf5 import del_data_if_exist
from utils.setup import easy_logger

PROJECT_DIR = Path(__file__).resolve().parents[2]
print(f"{PROJECT_DIR=}")

URL_DATA: Final = "http://kgc.knowledge-graph.jp/data"
URL_PREDICATE: Final = "http://kgc.knowledge-graph.jp/data/predicate/"
DATA_FOLDER_PATH: Final = f"{PROJECT_DIR}/../KGCdata/KGRC-RDF"

PREFIX_PREDICATE: Final = 'word.predicate'

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

    @classmethod
    def from_(cls) -> URIRef:
        return cls._NS['from']

    @classmethod
    def if_(cls) -> URIRef:
        return cls._NS['if']


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
    match_list = [s for s, _, o in graph_.triples((None, None, None))
                  if o in type_set]
    match_list = sorted(set(match_list), key=match_list.index)
    for type_ in type_set:
        if type_ in match_list:
            del match_list[match_list.index(type_)]
            pass
    return match_list


def make_graph(title: str, data_file_path: Union[str, Path]) -> tuple[Graph, dict[str, Namespace]]:
    g = Graph()
    g.parse(data_file_path)
    g.bind(PREFIX_PREDICATE, Namespace(URL_PREDICATE))
    g.bind(title, Namespace(f"{URL_DATA}/{title}/"))
    namespace_dict = {key: Namespace(value) for key, value in g.namespace_manager.namespaces()}
    return g, namespace_dict


def get_story_list(graph_, namespace_dict, title, story_length):
    story_list = []

    prefix_title = namespace_dict[title]
    for i in range(1, story_length + 1):
        iii = str(i).zfill(3)
        for s in ['', 'a', 'b', 'c']:
            iiis = iii + s
            story_ = prefix_title[iiis]
            tmp = list(graph_.triples((story_, RDF.type, None)))
            if len(tmp) == 1:
                story_list.append(iiis)
    return story_list


def get_svos_triples(graph_: Graph, namespace_dict: dict[str, Namespace], title: str, story_list: list[str]) \
        -> tuple[dict[str, dict[URIRef, list[Node]]], list[Node], list[str], list[URIRef], list[Node]]:
    """

    """
    p_list: list[URIRef] = [
        KGC.subject, KGC.hasProperty, KGC.hasPredicate, KGC.what, KGC.whom, KGC.to, KGC.where,
        KGC.on, KGC.from_(), KGC.when, KGC.time, KGC.why
    ]

    prefix_title = namespace_dict[title]
    triple_dict: dict[str, dict[URIRef, list[URIRef]]] = {iiis: {r: None for r in p_list} for iiis in story_list}

    for iiis in story_list:
        story_ = prefix_title[iiis]
        for r in p_list:
            triple_dict[iiis][r] = cast(list[URIRef], list(graph_.objects(story_, r)))
        if len(triple_dict[iiis][KGC.subject]) != 1:
            logger.info(f"{title}, {iiis}, {[x.n3(graph_.namespace_manager) for x in triple_dict[iiis][KGC.subject]]}")
    spo = [(iiis, subject, p, o)
           for iiis, p_os in triple_dict.items()
           for subject in p_os[KGC.subject]
           for p in p_os.keys()
           for o in p_os[p]]
    story_list, s_list, p_list, o_list = [[x[i] for x in spo] for i in range(4)]
    return triple_dict, story_list, s_list, p_list, o_list


class ConstNameForHDF:
    FILE_NAME: Final = f"{DATA_FOLDER_PATH}/../data/story_list.hdf5"
    LENGTH_10: Final = 'length_10'
    LENGTH_09: Final = 'length_09'
    LENGTH_08: Final = 'length_08'
    STORIES: Final = 'stories'
    OBJECTS: Final = 'objects'
    ACTIONS: Final = 'actions'
    PEOPLE: Final = 'people'
    TRIPLE: Final = 'triple'

    @classmethod
    def all_list(cls):
        return [cls.LENGTH_10, cls.LENGTH_09, cls.LENGTH_08, cls.STORIES,
                cls.OBJECTS, cls.ACTIONS, cls.PEOPLE, cls.TRIPLE]


def write_():
    # print(f"{DATA_FOLDER_PATH}/../data/story_list.hdf5")
    with h5py.File(ConstNameForHDF.FILE_NAME, 'a') as f:
        people_dict: dict[str, list] = dict()
        for title_ja, (title, l10, l09, l08) in title_len_dict.items():
            data_file_path = f"{DATA_FOLDER_PATH}/{title}.ttl"
            title_group = f.require_group(title)

            graph_, namespace_dict = make_graph(title, data_file_path)
            namespace_manager = graph_.namespace_manager

            story_list = get_story_list(graph_, namespace_dict, title, l10)
            objects_node_list = get_type_match_people(graph_, OBJECT_SET)
            actions_node_list = get_type_match_people(graph_, ACTION_SET)
            people_node_list = get_type_match_people(graph_, {KGC.Person})
            # ofobj_set = get_type_match_people(graph_, {KGC.OFobj})

            triple_dict, spo_story_list, s_list, p_list, o_list = \
                get_svos_triples(graph_, namespace_dict, title, story_list)

            def make_shape_list(_node_list: Iterable[Node]):
                _str_list = [URIRef(urlparse(str(p)).geturl()).n3(namespace_manager) for p in _node_list]
                _tuple_list = [tuple(p.split(':', 1)) for p in _str_list]
                _item_list = [x[1] for x in _tuple_list]
                return _str_list, _tuple_list, _item_list

            _, _, objects_list = make_shape_list(objects_node_list)
            _, _, actions_list = make_shape_list(actions_node_list)
            _, _, people_list = make_shape_list(people_node_list)
            objects_list = [l_ for l_ in objects_list if l_ not in people_list]
            del people_list[people_list.index('Watson')], people_list[people_list.index('Holmes')]

            s_list, p_list, o_list = map(lambda x: make_shape_list(x)[2], (s_list, p_list, o_list))
            triple = np.array(list(zip(s_list, p_list, o_list)), dtype=h5py.special_dtype(vlen=str))
            print(triple)

            del_data_if_exist(title_group, ConstNameForHDF.all_list())

            [title_group.create_dataset(name, data=value) for name, value in (
                (ConstNameForHDF.LENGTH_10, l10),
                (ConstNameForHDF.LENGTH_09, l09),
                (ConstNameForHDF.LENGTH_08, l08),
                (ConstNameForHDF.STORIES, np.array(story_list, dtype=h5py.special_dtype(vlen=str))),
                (ConstNameForHDF.OBJECTS, np.array(objects_list, dtype=h5py.special_dtype(vlen=str))),
                (ConstNameForHDF.ACTIONS, np.array(actions_list, dtype=h5py.special_dtype(vlen=str))),
                (ConstNameForHDF.PEOPLE, np.array(people_list, dtype=h5py.special_dtype(vlen=str))),
                (ConstNameForHDF.TRIPLE, triple),
            )]

            people_dict[title] = people_list

        # print(people_dict)


def read_():
    with h5py.File(ConstNameForHDF.FILE_NAME, 'r') as f:
        for title_ja, (title, l10, l09, l08) in title_len_dict.items():
            title_group = f.require_group(title)
            [logger.info(title_group[name][()]) for name in ConstNameForHDF.all_list()]


def main():
    write_()
    read_()


if __name__ == '__main__':
    main()
