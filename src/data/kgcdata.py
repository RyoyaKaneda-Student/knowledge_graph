#!/usr/bin/python
# -*- coding: utf-8 -*-
# ========== python ==========
# noinspection PyUnresolvedReferences
import os
from logging import Logger
from pathlib import Path
from urllib.parse import quote, unquote
# noinspection PyUnresolvedReferences
from datetime import datetime
# noinspection PyUnresolvedReferences
from typing import List, Dict, Tuple, Optional, Union, Callable, Final, Literal, Iterable, get_args, cast
import h5py
# noinspection PyUnresolvedReferences
from h5py import Group
import numpy as np
# noinspection PyUnresolvedReferences
from rdflib import Graph, URIRef, Literal, Namespace, RDF, RDFS, BNode
from rdflib.namespace import DefinedNamespace, NamespaceManager
from rdflib.term import Node

from models.datasets.data_helper import INFO_INDEX, ALL_TAIL_INDEX
from utils.hdf5 import del_data_if_exist, str_list_for_hdf5
from utils.setup import easy_logger
from utils.utils import replace_list_value, remove_duplicate_order_save

PROJECT_DIR = Path(__file__).resolve().parents[2]
print(f"{PROJECT_DIR=}")

URL_DATA: Final = "http://kgc.knowledge-graph.jp/data"
URL_PREDICATE: Final = "http://kgc.knowledge-graph.jp/data/predicate/"
DATA_FOLDER_PATH: Final = f"{PROJECT_DIR}/../KGCdata/KGRC-RDF"

PREFIX_PREDICATE: Final = 'word.predicate'

KGC_TYPE: Final = 'kgc:type'
KGC_LABEL: Final = 'kgc:label'

logger: Logger = easy_logger(console_level='info')


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


ALL_RELATION = [
    'rdf:type', 'kgc:subject', 'kgc:ActionOption', 'kgc:Not', 'kgc:RelationBetweenScene', 'kgc:SceneObjectProperty',
    'kgc:SceneProperty', 'kgc:TargetObjProperty', 'kgc:adjunct', 'kgc:around', 'kgc:at_the_same_time',
    'kgc:because', 'kgc:can', 'kgc:canNot', 'kgc:from', 'kgc:hasPart', 'kgc:hasPredicate', 'kgc:hasProperty', 'kgc:how',
    'kgc:if', 'kgc:infoReceiver', 'kgc:infoSource', 'kgc:left', 'kgc:middle', 'kgc:near', 'kgc:next_to', 'kgc:ofPart',
    'kgc:ofWhole', 'kgc:on', 'kgc:opposite', 'kgc:orTarget', 'kgc:otherwise', 'kgc:right', 'kgc:subject', 'kgc:then',
    'kgc:therefore', 'kgc:to', 'kgc:what', 'kgc:when', 'kgc:when_during', 'kgc:where', 'kgc:whom', 'kgc:why',
    'kgc:LocationProperty', 'kgc:time', 'kgc:AbstractTime', 'kgc:Action', 'kgc:Animal', 'kgc:CanAction',
    'kgc:CanNotAction', 'kgc:NotAction', 'kgc:OFobj', 'kgc:ORobj', 'kgc:Object', 'kgc:PhysicalObject',
    'kgc:Person', 'kgc:Place', 'kgc:Property', 'kgc:Scene', 'kgc:Situation', 'kgc:Statement', 'kgc:Talk', 'kgc:Thought',
]

ALL_TITLE: Final = 'All'
ALL_WITHOUT_TITLE: Final = 'AllWithoutTitle'
GENERAL: Final = 'General'

HOLMES_ALT_NAME: Final = 'AllTitle:Holmes'
WATSON_ALT_NAME: Final = 'AllTitle:Watson'
HOLMES_TITLE_NAME: Final = lambda title: f'{title}:Holmes'
WATSON_TITLE_NAME: Final = lambda title: f'{title}:Watson'

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

NodeList = list[Node]
URIRefList = list[URIRef]
NamespaceDict = dict[str, Namespace]
StoryList = list[str]
TripleDict = dict[str, dict[URIRef, URIRefList]]
StorySPO = tuple[StoryList, URIRefList, URIRefList, URIRefList, URIRefList]
StoryPO = tuple[StoryList, URIRefList, URIRefList]


def get_type_match_people(graph_: Graph, type_set: set[URIRef]) -> URIRefList:
    match_list = remove_duplicate_order_save(
        [s for s, _, o in graph_.triples((None, None, None)) if o in type_set])
    for type_ in type_set:
        if type_ in match_list:
            del match_list[match_list.index(type_)]
            pass
    return cast(URIRefList, match_list)


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


def get_svos_triples(graph_: Graph, namespace_dict: NamespaceDict, title: str, story_list: StoryList) \
        -> tuple[TripleDict, StorySPO, StoryPO]:
    """

    """
    kgc_p_list: list[URIRef] = [
        KGC.subject, KGC.hasPredicate, KGC.hasProperty, KGC.what, KGC.whom, KGC.to, KGC.where,
        KGC.on, KGC.from_(), KGC.when, KGC.time, KGC.why
    ]

    prefix_title = namespace_dict[title]
    triple_dict: dict[str, dict[URIRef, URIRefList]] = {iiis: {r: None for r in kgc_p_list} for iiis in story_list}

    for iiis in story_list:
        story_ = prefix_title[iiis]
        triple_dict[iiis] = {
            p: list(graph_.objects(story_, p)) for p in kgc_p_list
        }
        if len(triple_dict[iiis][KGC.subject]) != 1:
            logger.debug(f"{title}, {iiis}, {[x.n3(graph_.namespace_manager) for x in triple_dict[iiis][KGC.subject]]}")
    # logger.info([triple_dict[iiis][p] for iiis in story_list for p in kgc_p_list[:]])
    storynum_p_subject_predicate_o = [(iiis, p, subject, predicate, o)
                                      for iiis in story_list
                                      for subject in triple_dict[iiis][KGC.subject]
                                      for predicate in triple_dict[iiis][KGC.hasPredicate]
                                      for p in kgc_p_list[2:]  # delete
                                      for o in triple_dict[iiis][p]
                                      ]

    # logger.info(f"{len(story_list)=}")

    storynum_p_o = [(iiis, p, o)
                    for iiis in story_list
                    for p in kgc_p_list
                    for o in triple_dict[iiis][p]
                    ]

    return (
        triple_dict,
        cast(StorySPO, tuple(zip(*storynum_p_subject_predicate_o))),
        cast(StoryPO, tuple(zip(*storynum_p_o))))


class ConstName1:
    FILE_NAME: Final = f"{DATA_FOLDER_PATH}/../data/story_list.hdf5"
    LENGTH_10: Final = 'length_10'
    LENGTH_09: Final = 'length_09'
    LENGTH_08: Final = 'length_08'
    STORIES_IIIS: Final = 'stories_iiis'
    STORIES: Final = 'stories'
    OBJECTS: Final = 'objects'
    ACTIONS: Final = 'actions'
    PEOPLE: Final = 'people'
    SPO_TRIPLE: Final = 'story_property_subject_predicate_object'
    PO_TRIPLE: Final = 'story_property_object'

    RELATION: Final = 'relation'

    @classmethod
    def all_list(cls):
        return [cls.LENGTH_10, cls.LENGTH_09, cls.LENGTH_08, cls.STORIES, cls.STORIES_IIIS,
                cls.OBJECTS, cls.ACTIONS, cls.PEOPLE, cls.SPO_TRIPLE, cls.PO_TRIPLE]


def change_sentence(s: URIRef, namespace_manager: NamespaceManager):
    if type(s.toPython()) is str:
        url_, value = str(s).rsplit('/', maxsplit=1)
        rev = unquote(URIRef(url_ + '/' + quote(value, safe='#')).n3(namespace_manager))
    else:
        rev = "DateTime:{}".format(str(s))
    return rev


def replace_holmes_and_watson(list_, title):
    return replace_list_value(list_, (
        (HOLMES_TITLE_NAME(title), HOLMES_ALT_NAME),
        (WATSON_TITLE_NAME(title), WATSON_ALT_NAME),
    ))


def write_one_title(f, title_ja, title, l10, l09, l08):
    logger.debug(f"{title_ja=}")
    data_file_path = f"{DATA_FOLDER_PATH}/{title}.ttl"
    title_group = f.require_group(title)

    graph_, namespace_dict = make_graph(title, data_file_path)
    namespace_manager = graph_.namespace_manager

    story_iiis_list = get_story_list(graph_, namespace_dict, title, l10)
    story_name_list = [f'{title}:{iiis}' for iiis in story_iiis_list]
    objects_node_list = get_type_match_people(graph_, OBJECT_SET)
    actions_node_list = get_type_match_people(graph_, ACTION_SET)
    people_node_list = get_type_match_people(graph_, {KGC.Person})
    # ofobj_set = get_type_match_people(graph_, {KGC.OFobj})

    triple_dict, svo_items, story_po_items = get_svos_triples(graph_, namespace_dict, title, story_iiis_list)

    def make_shape_list(_node_list: Iterable[URIRef], del_prefix=False):
        _str_list = [change_sentence(p, namespace_manager) for p in _node_list]
        _tuple_list = [tuple(p.split(':', 1)) for p in _str_list]
        _item_list = [x[1] if del_prefix else ':'.join(x) for x in _tuple_list]
        return _item_list

    objects_list = make_shape_list(objects_node_list)
    actions_list = make_shape_list(actions_node_list)
    people_list = make_shape_list(people_node_list)

    del people_list[people_list.index(HOLMES_TITLE_NAME(title))]
    del people_list[people_list.index(WATSON_TITLE_NAME(title))]
    # story_spo
    spo_story_list = svo_items[0]
    property_list, s_list, p_list, o_list = map(make_shape_list, svo_items[1:])
    spo_list = [replace_holmes_and_watson(_list, title) for _list
                in (spo_story_list, property_list, s_list, p_list, o_list)]
    # story_po
    po_story_list = [f"{title}:{iiis}" for iiis in story_po_items[0]]
    p_list, o_list = map(make_shape_list, story_po_items[1:])
    po_list = [replace_holmes_and_watson(_list, title) for _list
               in (po_story_list, p_list, o_list)]
    objects_list = [
        o for o in remove_duplicate_order_save(objects_list + o_list)
        if (o not in people_list) and (o not in people_list) and (o not in actions_list) and (o not in story_name_list)
    ]

    spo_array: np.ndarray = str_list_for_hdf5(spo_list).T
    po_array: np.ndarray = str_list_for_hdf5(po_list).T

    del_data_if_exist(title_group, ConstName1.all_list())
    [title_group.create_dataset(name, data=value) for name, value in (
        (ConstName1.LENGTH_10, l10),
        (ConstName1.LENGTH_09, l09),
        (ConstName1.LENGTH_08, l08),
        (ConstName1.STORIES_IIIS, str_list_for_hdf5(story_iiis_list)),
        (ConstName1.STORIES, str_list_for_hdf5(story_name_list)),
        (ConstName1.OBJECTS, str_list_for_hdf5(objects_list)),
        (ConstName1.ACTIONS, str_list_for_hdf5(actions_list)),
        (ConstName1.PEOPLE, str_list_for_hdf5(people_list)),
        (ConstName1.SPO_TRIPLE, spo_array),
        (ConstName1.PO_TRIPLE, po_array),
    )]
    return objects_list, actions_list, people_list, story_name_list, spo_array, po_array


def write_():
    # print(f"{DATA_FOLDER_PATH}/../data/story_list.hdf5")
    with h5py.File(ConstName1.FILE_NAME, 'a') as f:
        objects_all_set = set()
        objects_all_set_split_title = set()
        actions_all_set = set()
        actions_all_set_split_title = set()
        people_all_set = set()
        people_all_set_split_title = set()
        story_all_name_list = list()
        spo_all_array = np.array([[0, 0, 0, 0, 0]])  # dammy
        po_all_array = np.array([[0, 0, 0]])  # dammy

        for title_ja, (title, l10, l09, l08) in title_len_dict.items():
            objects_list, actions_list, people_list, story_name_list, spo_array, po_array = (
                write_one_title(f, title_ja, title, l10, l09, l08))
            objects_all_set |= set(objects_list)
            objects_all_set_split_title |= {o.replace(f'{title}:', '') for o in objects_list}
            actions_all_set |= set(actions_list)
            actions_all_set_split_title |= {a.replace(f'{title}:', '') for a in actions_all_set}
            people_all_set |= set(people_list)
            people_all_set_split_title |= {p.replace(f'{title}:', '') for p in people_list}
            story_all_name_list = story_all_name_list + story_name_list
            spo_all_array = np.concatenate((spo_all_array, spo_array))
            po_all_array = np.concatenate((po_all_array, po_array))
            del objects_list, actions_list, people_list, story_name_list, spo_array, po_array

        spo_all_array = spo_all_array[1:]
        po_all_array = po_all_array[1:]
        # general
        title_group = f.require_group(GENERAL)
        del_data_if_exist(title_group, [ConstName1.PEOPLE, ConstName1.RELATION])
        title_group.create_dataset(ConstName1.PEOPLE, data=str_list_for_hdf5([HOLMES_ALT_NAME, WATSON_ALT_NAME]))
        title_group.create_dataset(ConstName1.RELATION, data=str_list_for_hdf5(ALL_RELATION))
        # all
        title_group = f.require_group(ALL_TITLE)
        del_data_if_exist(title_group, [
            ConstName1.OBJECTS, ConstName1.ACTIONS, ConstName1.PEOPLE,
            ConstName1.STORIES, ConstName1.SPO_TRIPLE, ConstName1.PO_TRIPLE])
        [title_group.create_dataset(_name, data=_list) for _name, _list in (
            (ConstName1.OBJECTS, str_list_for_hdf5(list(objects_all_set))),
            (ConstName1.ACTIONS, str_list_for_hdf5(list(actions_all_set))),
            (ConstName1.PEOPLE, str_list_for_hdf5(list(people_all_set))),
            (ConstName1.STORIES, str_list_for_hdf5(story_all_name_list)),
            (ConstName1.SPO_TRIPLE, spo_all_array),
            (ConstName1.PO_TRIPLE, po_all_array),
        )]
        # all without title info
        title_group = f.require_group(ALL_WITHOUT_TITLE)
        del_data_if_exist(title_group, [ConstName1.OBJECTS, ConstName1.ACTIONS, ConstName1.PEOPLE])
        [title_group.create_dataset(_name, data=_list) for _name, _list in (
            (ConstName1.OBJECTS, str_list_for_hdf5(list(objects_all_set_split_title))),
            (ConstName1.ACTIONS, str_list_for_hdf5(list(actions_all_set_split_title))),
            (ConstName1.PEOPLE, str_list_for_hdf5(list(people_all_set_split_title))),
        )]


def read_():
    with h5py.File(ConstName1.FILE_NAME, 'r') as f:
        for title_ja, (title, l10, l09, l08) in title_len_dict.items():
            title_group = f.require_group(title)
            [logger.debug(title_group[name][()]) for name in ConstName1.all_list()]


class ConstName2:
    WRITE_FILE: Final = f"{PROJECT_DIR}/data/processed/KGCdata"
    keyREVERSE: Final = lambda key: f'{key}_REVERSE'


def write2_write_triples(fw_info, fw_train, entity_list, relation_list, is_rev_list, triple, triple_raw=None):
    #
    del_data_if_exist(fw_info, INFO_INDEX.all_index())
    fw_info.create_dataset(INFO_INDEX.E_LEN, data=len(entity_list))
    fw_info.create_dataset(INFO_INDEX.R_LEN, data=len(relation_list))
    fw_info.create_dataset(INFO_INDEX.ENTITIES, data=entity_list)
    fw_info.create_dataset(INFO_INDEX.RELATIONS, data=relation_list)
    fw_info.create_dataset(INFO_INDEX.IS_REV_RELATION, data=is_rev_list)
    fw_info.create_dataset(INFO_INDEX.ID2COUNT_ENTITY,
                           data=np.bincount(triple[:, (0, 2)].flatten(), minlength=len(entity_list)))
    fw_info.create_dataset(INFO_INDEX.ID2COUNT_RELATION,
                           data=np.bincount(triple[:, 1], minlength=len(relation_list)))
    # triple
    del_data_if_exist(fw_train, [INFO_INDEX.TRIPLE, f'{INFO_INDEX.TRIPLE}_raw'])
    fw_train.create_dataset(INFO_INDEX.TRIPLE, data=triple)
    if triple_raw is not None: fw_train.create_dataset(f'{INFO_INDEX.TRIPLE}_raw', data=triple_raw)


def write2_svo(title: str, read_group: Group):
    entity_list = [HOLMES_ALT_NAME, WATSON_ALT_NAME] + [b.decode('utf-8') for b in (
        *(read_group[ConstName1.PEOPLE][()]),
        *(read_group[ConstName1.STORIES][()]),
        *(read_group[ConstName1.OBJECTS][()]),
    )]
    relation_list = [b.decode('utf-8') for b in read_group[ConstName1.ACTIONS][()]]
    entity_dict = {e: i for i, e in enumerate(entity_list)}
    relation_dict = {r: i for i, r in enumerate(relation_list)}

    svo_triple = np.array([
        [entity_dict[s.decode('utf-8')], relation_dict[v.decode('utf-8')], entity_dict[o.decode('utf-8')]]
        for _, _, s, v, o in read_group[ConstName1.SPO_TRIPLE][()]
    ])
    # add reverse
    relation_len_no_reverse = len(relation_list)
    relation_list = relation_list + [ConstName2.keyREVERSE(r) for r in relation_list]
    is_rev_list = np.concatenate(
        [np.zeros(relation_len_no_reverse, dtype=bool), np.ones(relation_len_no_reverse, dtype=bool)]
    )
    assert len(relation_list) == len(is_rev_list) and len(is_rev_list)==2*relation_len_no_reverse
    svo_triple_reverse = svo_triple[:, (2, 1, 0)]
    svo_triple_reverse[:, 1] += relation_len_no_reverse
    # concat
    svo_triple = np.concatenate([svo_triple, svo_triple_reverse])

    with (h5py.File(f"{ConstName2.WRITE_FILE}/{title}/SVO/info.hdf5", 'a') as fw_info,
          h5py.File(f"{ConstName2.WRITE_FILE}/{title}/SVO/train.hdf5", 'a') as fw_train):
        write2_write_triples(fw_info, fw_train, entity_list, relation_list, is_rev_list, svo_triple)
        pass


def write2_sro(title: str, read_group: Group, general_read_group: Group):
    entity_list = [HOLMES_ALT_NAME, WATSON_ALT_NAME] + [b.decode('utf-8') for b in (
        *(read_group[ConstName1.PEOPLE][()]),
        *(read_group[ConstName1.STORIES][()]),
        *(read_group[ConstName1.ACTIONS][()]),
        *(read_group[ConstName1.OBJECTS][()]),
    )]
    entity_dict = {e: i for i, e in enumerate(entity_list)}
    relation_list = [b.decode('utf-8') for b in general_read_group[ConstName1.RELATION][()]]
    relation_dict = {r: i for i, r in enumerate(relation_list)}
    sro_triple = np.array([
        [entity_dict[s.decode('utf-8')], relation_dict[r.decode('utf-8')], entity_dict[o.decode('utf-8')]]
        for s, r, o in read_group[ConstName1.PO_TRIPLE][()]
    ])
    # add reverse
    """
    relation_len_no_reverse = len(relation_list)
    relation_list = relation_list + [ConstName2.keyREVERSE(r) for r in relation_list]
    is_rev_list = np.concatenate(
        [np.zeros(relation_len_no_reverse, dtype=bool), np.ones(relation_len_no_reverse, dtype=bool)]
    )
    sro_triple_reverse = sro_triple[:, (2, 1, 0)]
    sro_triple_reverse[:, 1] += relation_len_no_reverse

    sro_triple = np.concatenate([sro_triple, sro_triple_reverse])
    """
    sro_triple_raw = [
        [entity_list[s], relation_list[r], entity_list[o]]
        for s, r, o in sro_triple
    ]

    with (h5py.File(f"{ConstName2.WRITE_FILE}/{title}/SRO/info.hdf5", 'a') as fw_info,
          h5py.File(f"{ConstName2.WRITE_FILE}/{title}/SRO/train.hdf5", 'a') as fw_train):
        write2_write_triples(fw_info, fw_train, entity_list, relation_list,
                             np.zeros(len(relation_list), dtype=bool), sro_triple,
                             str_list_for_hdf5(sro_triple_raw))


def write2_():
    with (h5py.File(ConstName1.FILE_NAME, 'r') as fr, ):
        general_read_group = fr[GENERAL]
        for title_ja, (title, _, _, _) in title_len_dict.items():
            # read
            read_group = fr[title]
            # svo
            write2_svo(title, read_group)
            # spo
            write2_sro(title, read_group, general_read_group)
            del title_ja, title
        read_group = fr[ALL_TITLE]
        write2_svo(ALL_TITLE, read_group)
        write2_sro(ALL_TITLE, read_group, general_read_group)


def write3_write_title(fr_train, fw_all_tail):
    triples: np.ndarray = fr_train[INFO_INDEX.TRIPLE][()]
    triples_list = cast(list[tuple[int, int, int]], triples.tolist())

    er_list = list({(h, r) for (h, r, t) in triples_list})
    er_dict = {key: [] for key in er_list}
    [er_dict[(h, r)].append(t) for (h, r, t) in triples_list]
    er_list = sorted(list(er_list), key=lambda _er: -len(er_dict[_er]))
    all_tail = np.array([
        (i, t, 1) for i, _er in enumerate(er_list) for t in er_dict[_er]
    ])
    del_data_if_exist(fw_all_tail, ALL_TAIL_INDEX.all_index())
    # write
    [fw_all_tail.create_dataset(_name, data=_data) for _name, _data in (
        (ALL_TAIL_INDEX.ER_LENGTH, len(er_list)),
        (ALL_TAIL_INDEX.ERS, er_list),
        (ALL_TAIL_INDEX.ID2ALL_TAIL_ROW, all_tail[:, 0]),
        (ALL_TAIL_INDEX.ID2ALL_TAIL_ENTITY, all_tail[:, 1]),
        (ALL_TAIL_INDEX.ID2ALL_TAIL_MODE, all_tail[:, 2])
    )]


def write3_():
    for title_ja, (title, l10, l09, l08) in title_len_dict.items():
        with (h5py.File(f"{ConstName2.WRITE_FILE}/{title}/SVO/train.hdf5", 'r') as fr_train,
              h5py.File(f"{ConstName2.WRITE_FILE}/{title}/SVO/all_tail.hdf5", 'a') as fw_all_tail, ):
            write3_write_title(fr_train, fw_all_tail)

    with (h5py.File(f"{ConstName2.WRITE_FILE}/{ALL_TITLE}/SVO/train.hdf5", 'r') as fr_train,
          h5py.File(f"{ConstName2.WRITE_FILE}/{ALL_TITLE}/SVO/all_tail.hdf5", 'a') as fw_all_tail, ):
        write3_write_title(fr_train, fw_all_tail)


def main():
    write_()
    read_()
    write2_()
    write3_()


if __name__ == '__main__':
    main()
    pass
