#!/usr/bin/python
# -*- coding: utf-8 -*-
# ========== python ==========
# noinspection PyUnresolvedReferences
import collections
from logging import Logger
from pathlib import Path
from urllib.parse import quote, unquote
# noinspection PyUnresolvedReferences
from datetime import datetime
# noinspection PyUnresolvedReferences
from typing import List, Dict, Tuple, Optional, Union, Callable, Final, Literal, Iterable, get_args, cast
import h5py
# noinspection PyUnresolvedReferences
import rdflib
from h5py import Group
import numpy as np
# noinspection PyUnresolvedReferences
from rdflib import Graph, URIRef, Namespace, RDF, RDFS, BNode
from rdflib.namespace import DefinedNamespace, NamespaceManager
from rdflib.term import Node

from models.datasets.data_helper import INFO_INDEX, ALL_TAIL_INDEX
from utils.hdf5 import del_data_if_exist, str_list_for_hdf5
from utils.setup import easy_logger
from utils.utils import replace_list_value

PROJECT_DIR = Path(__file__).resolve().parents[2]
print(f"{PROJECT_DIR=}")

URL_DATA: Final = "http://kgc.knowledge-graph.jp/data"
URL_PREDICATE: Final = "http://kgc.knowledge-graph.jp/data/predicate/"
DATA_FOLDER_PATH: Final = f"{PROJECT_DIR}/../KGCdata/KGRC-RDF"

PREFIX_PREDICATE: Final = 'word.predicate'

KGC_TYPE: Final = 'kgc:type'
KGC_LABEL: Final = 'kgc:label'

logger: Logger = easy_logger(console_level='debug')


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
    "僧坊荘園": ("AbbeyGrange", 414, 372, 331, 310),
    "花婿失踪事件": ("ACaseOfIdentity", 580, 522, 464, 435),
    "背中の曲がった男": ("CrookedMan", 373, 335, 298, 279),
    "踊る人形": ("DancingMen", 231, 207, 184, 173),
    "悪魔の足": ("DevilsFoot", 489, 440, 391, 366),
    "入院患者": ("ResidentPatient", 324, 291, 259, 243),
    "白銀号事件": ("SilverBlaze", 397, 367, 317, 297),
    "マダラのひも": ("SpeckledBand", 401, 360, 320, 300)
}

OBJECT_SET = {
    KGC.Animal, KGC.OFobj, KGC.ORobj, KGC.Object,
    KGC.PhysicalObject, KGC.Person, KGC.Place,
}

ACTION_SET = {
    KGC.Action, KGC.CanAction, KGC.CanNotAction, KGC.NotAction
}

PEOPLE_SET = {
    KGC.Person,
}

SCENE_SET = {
    KGC.Scene, KGC.Situation, KGC.Thought, KGC.Statement, KGC.Talk
}

NodeList = list[Node]
URIRefList = list[URIRef]
NamespaceDict = dict[str, Namespace]
StoryList = list[str]
TripleDict = dict[str, dict[URIRef, URIRefList]]
StorySPO = tuple[StoryList, URIRefList, URIRefList, URIRefList, URIRefList]
StoryPO = tuple[StoryList, URIRefList, URIRefList]


def get_pure_path(_path):
    return Path(_path).resolve().as_posix()


def get_type_match_items(graph_: Graph, type_set: set[URIRef]) -> URIRefList:
    # noinspection PyTypeChecker
    match_list: URIRefList = list(dict.fromkeys(
        [s for s, _, o in graph_.triples((None, None, None)) if o in type_set]))
    match_list = [_l for _l in match_list if True not in [_l == type_ for type_ in type_set]]
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


def get_triples(graph_: Graph, namespace_dict: NamespaceDict, title: str, story_list: StoryList) \
        -> tuple[TripleDict, StorySPO, StoryPO]:
    """

    Args:
        graph_:
        namespace_dict:
        title:
        story_list:

    Returns:

    """
    kgc_p_list: list[URIRef] = [
        KGC.subject, KGC.hasPredicate, KGC.hasProperty, KGC.what, KGC.whom, KGC.to, KGC.where, KGC.infoSource,
        KGC.on, KGC.from_(), KGC.when, KGC.time, KGC.why
    ]

    prefix_title = namespace_dict[title]
    triple_dict: dict[str, dict[URIRef, URIRefList]] = dict()

    for iiis in story_list:
        story_ = prefix_title[iiis]
        triple_dict[iiis] = {p: list(graph_.objects(story_, p)) for p in kgc_p_list}

    storynum_p_subject_predicate_o = [(URIRef(f'{title}:{iiis}'), p, subject, predicate, o)
                                      for iiis in story_list
                                      for subject in triple_dict[iiis][KGC.subject]
                                      for predicate in triple_dict[iiis][KGC.hasPredicate]
                                      for p in kgc_p_list[2:]  # delete
                                      for o in triple_dict[iiis][p]]

    # logger.info(f"{len(story_list)=}")

    storynum_p_o = [(URIRef(f'{title}:{iiis}'), p, o)
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
    WRITE_FILE: Final = f"{PROJECT_DIR}/data/processed/KGCdata"

    LENGTH_100: Final = 'length_100'
    LENGTH_090: Final = 'length_090'
    LENGTH_080: Final = 'length_080'
    LENGTH_075: Final = 'length_075'
    STORIES_IIIS: Final = 'stories_iiis'
    STORIES: Final = 'stories'
    OBJECTS: Final = 'objects'
    PURE_OBJECTS: Final = 'pure_objects'
    PEOPLE: Final = 'people'
    ACTIONS: Final = 'actions'
    OBJECTS_LABEL: Final = 'objects_label'
    ACTIONS_LABEL: Final = 'actions_label'
    PEOPLE_LABEL: Final = 'people_label'
    PURE_OBJECTS_LABEL: Final = 'pure_objects_label'

    SPO_TRIPLE: Final = 'story_property_subject_predicate_object'
    PO_TRIPLE: Final = 'story_property_object'

    ENTITY: Final = 'entity'
    RELATION: Final = 'relation'

    keyREVERSE: Final = lambda key: f'{key}_REVERSE'

    @classmethod
    def all_list(cls):
        return [cls.LENGTH_100, cls.LENGTH_090, cls.LENGTH_080, cls.LENGTH_075, cls.STORIES, cls.STORIES_IIIS,
                cls.OBJECTS, cls.PURE_OBJECTS, cls.ACTIONS, cls.PEOPLE,
                cls.OBJECTS_LABEL, cls.PURE_OBJECTS_LABEL, cls.PEOPLE_LABEL, cls.ACTIONS_LABEL,
                cls.SPO_TRIPLE, cls.PO_TRIPLE]


def change_sentence(s: URIRef, namespace_manager: NamespaceManager):
    if type(s.toPython()) is str:
        x = str(s).rsplit('/', maxsplit=1)
        if len(x) == 2:
            url_, value = x
            rev = unquote(URIRef(url_ + '/' + quote(value, safe='#')).n3(namespace_manager))
        else:
            rev = s.toPython()
    else:
        rev = "DateTime:{}".format(str(s))
    return rev


def save_info_list(
        title_group, l100, l090, l080, l075,
        story_iiis_list, story_name_list, objects_list, pure_objects_list, people_list, actions_list,
        objects_label_list, pure_objects_label_list, people_label_list, actions_label_list, spo_array, po_array,
):
    """

    Args:
        pure_objects_label_list:
        pure_objects_list:
        title_group:
        l100:
        l090:
        l080:
        l075:
        story_iiis_list:
        story_name_list:
        objects_list:
        actions_list:
        people_list:
        objects_label_list:
        actions_label_list:
        people_label_list:
        spo_array:
        po_array:

    Returns:

    """

    def change_to_np_array(_list):
        if _list is None: return None
        return str_list_for_hdf5(_list) if type(story_iiis_list) is not np.ndarray else _list

    [title_group.create_dataset(name, data=value) for name, value in (
        (ConstName1.LENGTH_100, l100),
        (ConstName1.LENGTH_090, l090),
        (ConstName1.LENGTH_080, l080),
        (ConstName1.LENGTH_075, l075),
        (ConstName1.STORIES_IIIS, change_to_np_array(story_iiis_list)),
        (ConstName1.STORIES, change_to_np_array(story_name_list)),
        (ConstName1.OBJECTS, change_to_np_array(objects_list)),
        (ConstName1.PURE_OBJECTS, change_to_np_array(pure_objects_list)),
        (ConstName1.PEOPLE, change_to_np_array(people_list)),
        (ConstName1.ACTIONS, change_to_np_array(actions_list)),
        (ConstName1.OBJECTS_LABEL, change_to_np_array(objects_label_list)),
        (ConstName1.PURE_OBJECTS_LABEL, change_to_np_array(pure_objects_label_list)),
        (ConstName1.PEOPLE_LABEL, change_to_np_array(people_label_list)),
        (ConstName1.ACTIONS_LABEL, change_to_np_array(actions_label_list)),
        (ConstName1.SPO_TRIPLE, spo_array),
        (ConstName1.PO_TRIPLE, po_array),
    ) if value is not None]


def write_one_title(f, title_ja, title, l100, l090, l080, l075):
    logger.debug(f"{title_ja=}")
    data_file_path = f"{DATA_FOLDER_PATH}/{title}.ttl"
    title_group = f.require_group(title)

    graph_, namespace_dict = make_graph(title, data_file_path)
    namespace_manager = graph_.namespace_manager

    story_iiis_list = get_story_list(graph_, namespace_dict, title, l100)
    story_name_list = [f'{title}:{iiis}' for iiis in story_iiis_list]

    # ofobj_set = get_type_match_people(graph_, {KGC.OFobj})

    def make_label_list(_node_list: Iterable[URIRef]):
        # noinspection PyTypeChecker
        _label_list_list: list[list[rdflib.term.Literal]] = [
            [o for _, _, o in graph_.triples((_n, RDFS.label, None)) if cast(rdflib.term.Literal, o).language == 'en']
            for _n in _node_list
        ]
        _label_list = [labels[0].value if len(labels) > 0 else '' for labels in _label_list_list]
        return _label_list

    def replace_holmes_and_watson(_list):
        return replace_list_value(_list, (
            (HOLMES_TITLE_NAME(title), HOLMES_ALT_NAME),
            (WATSON_TITLE_NAME(title), WATSON_ALT_NAME),
        ))

    def make_shape_str(_node_list: Iterable[URIRef]):
        _str_list = [change_sentence(_n, namespace_manager) for _n in _node_list]
        return replace_holmes_and_watson(_str_list)

    triple_dict, svo_items, story_po_items = get_triples(graph_, namespace_dict, title, story_iiis_list)
    # story_spo
    svo_items_shaped = [make_shape_str(item) for item in svo_items]
    # story_po
    story_po_items_shaped = [make_shape_str(item) for item in story_po_items]

    def make_str_label_list(_list: list[URIRef]):
        _node_list = _list
        _str_list = make_shape_str(_list)
        _label_list = make_label_list(_list)
        return _list, _str_list, _label_list

    objects_node_list = get_type_match_items(graph_, OBJECT_SET)
    people_node_list = get_type_match_items(graph_, PEOPLE_SET)
    actions_node_list = get_type_match_items(graph_, ACTION_SET)

    scene_set = get_type_match_items(graph_, SCENE_SET)
    objects_node_list = list(dict.fromkeys(objects_node_list + list(story_po_items[2])))

    pure_objects_list = [o for o in objects_node_list
                         if o not in (people_node_list + actions_node_list + scene_set)]
    logger.debug(actions_node_list)
    logger.debug(pure_objects_list)

    objects_node_list, objects_str_list, objects_label_list = make_str_label_list(objects_node_list)
    pure_objects_node_list, pure_objects_str_list, pure_objects_label_list = make_str_label_list(pure_objects_list)
    people_node_list, people_str_list, people_label_list = make_str_label_list(people_node_list)
    actions_node_list, actions_str_list, actions_label_list = make_str_label_list(actions_node_list)
    # logger.debug(pure_objects_str_list)

    spo_array = str_list_for_hdf5(svo_items_shaped).T
    po_array = str_list_for_hdf5(story_po_items_shaped).T

    del_data_if_exist(title_group, ConstName1.all_list())
    save_info_list(
        title_group, l100, l090, l080, l075,
        story_iiis_list, story_name_list,
        objects_str_list, pure_objects_str_list, people_str_list, actions_str_list,
        objects_label_list, pure_objects_label_list, people_label_list, actions_label_list,
        spo_array, po_array
    )
    return (
        objects_str_list, pure_objects_str_list, people_str_list, actions_str_list, story_name_list,
        objects_label_list, pure_objects_label_list, people_label_list, actions_label_list,
        spo_array, po_array
    )


def write_():
    # print(f"{DATA_FOLDER_PATH}/../data/story_list.hdf5")
    print("save at: {}".format(get_pure_path(ConstName1.FILE_NAME)))
    with h5py.File(ConstName1.FILE_NAME, 'a') as f:
        all_objects_str_list, all_pure_objects_str_list, all_people_str_list, all_actions_str_list = \
            list[str](), list[str](), list[str](), list[str]()
        all_objects_label_list, all_pure_objects_label_list, all_people_label_list, all_actions_label_list = \
            list[str](), list[str](), list[str](), list[str]()
        all_story_name_list = list[str]()
        all_spo_array_list, all_po_array_list = list[np.ndarray](), list[np.ndarray]()

        for title_ja, (title, l100, l090, l080, l075) in title_len_dict.items():
            (
                objects_str_list, pure_objects_str_list, people_str_list, actions_str_list, story_name_list,
                objects_label_list, pure_objects_label_list, people_label_list, actions_label_list,
                spo_array, po_array
            ) = (write_one_title(f, title_ja, title, l100, l090, l080, l075))
            all_objects_str_list += objects_str_list
            all_objects_label_list += objects_label_list
            all_pure_objects_str_list += pure_objects_str_list
            all_pure_objects_label_list += pure_objects_label_list
            all_people_str_list += people_str_list
            all_people_label_list += people_label_list
            all_actions_str_list += actions_str_list
            all_actions_label_list += actions_label_list
            all_story_name_list += story_name_list
            all_spo_array_list.append(spo_array)
            all_po_array_list.append(po_array)

        #
        def del_duplication(_str_list, _label_list):
            _dict = {key: value for key, value in zip(_str_list, _label_list)}
            return list(_dict.keys()), list(_dict.values())

        all_objects_str_list, all_objects_label_list = del_duplication(all_objects_str_list, all_objects_label_list)
        all_pure_objects_str_list, all_pure_objects_label_list = del_duplication(
            all_pure_objects_str_list, all_pure_objects_label_list)
        all_people_str_list, all_people_label_list = del_duplication(all_people_str_list, all_people_label_list)
        all_actions_str_list, all_actions_label_list = del_duplication(all_actions_str_list, all_actions_label_list)
        #
        all_spo_array = np.concatenate(all_spo_array_list)
        all_po_array = np.concatenate(all_po_array_list)
        # general
        title_group = f.require_group(GENERAL)
        del_data_if_exist(title_group, [ConstName1.PEOPLE, ConstName1.RELATION])
        title_group.create_dataset(ConstName1.PEOPLE, data=str_list_for_hdf5([HOLMES_ALT_NAME, WATSON_ALT_NAME]))
        title_group.create_dataset(ConstName1.RELATION, data=str_list_for_hdf5(ALL_RELATION))
        # all
        title_group = f.require_group(ALL_TITLE)
        del_data_if_exist(title_group, ConstName1.all_list())

        save_info_list(
            title_group, None, None, None, None,
            None, all_story_name_list,
            all_objects_str_list, all_pure_objects_str_list, all_people_str_list, all_actions_str_list,
            all_objects_label_list, all_pure_objects_label_list, all_people_label_list, all_actions_label_list,
            all_spo_array, all_po_array,
        )


def read_():
    with h5py.File(ConstName1.FILE_NAME, 'r') as f:
        for title_ja, (title, l100, l090, l080, l075) in title_len_dict.items():
            title_group = f.require_group(title)
            logger.debug(get_pure_path(title_group.file.filename))
            # [logger.debug(title_group[name][()]) for name in ConstName1.all_list()]


class ConstName2:
    WRITE_SVO_INFO_FILE: Final = lambda title: f"{PROJECT_DIR}/data/processed/KGCdata/{title}/SVO/info.hdf5"
    WRITE_SVO_TRAIN_FILE: Final = lambda title: f"{PROJECT_DIR}/data/processed/KGCdata/{title}/SVO/train.hdf5"
    WRITE_SRO_INFO_FILE: Final = lambda title: f"{PROJECT_DIR}/data/processed/KGCdata/{title}/SRO/info.hdf5"
    WRITE_SRO_TRAIN_FILE: Final = lambda title: f"{PROJECT_DIR}/data/processed/KGCdata/{title}/SRO/train.hdf5"
    WRITE_SRO_ALL_TAIL_FILE: Final = lambda title: f"{PROJECT_DIR}/data/processed/KGCdata/{title}/SVO/all_tail.hdf5"
    keyREVERSE: Final = lambda key: f'{key}_REVERSE'


def write2_write_triples(fw_info, fw_train, entity_list, entity_label_list, relation_list, relation_label_list,
                         is_rev_list, triple, triple_raw=None):
    #
    logger.debug(f"info: {get_pure_path(fw_info.filename)}")
    logger.debug(f"triple: {get_pure_path(fw_train.filename)}")
    del_data_if_exist(fw_info, INFO_INDEX.all_index())
    fw_info.create_dataset(INFO_INDEX.E_LEN, data=len(entity_list))
    fw_info.create_dataset(INFO_INDEX.R_LEN, data=len(relation_list))
    fw_info.create_dataset(INFO_INDEX.ENTITIES, data=entity_list)
    fw_info.create_dataset(INFO_INDEX.ENTITIES_LABEL, data=entity_label_list)
    fw_info.create_dataset(INFO_INDEX.RELATIONS, data=relation_list)
    fw_info.create_dataset(INFO_INDEX.RELATIONS_LABEL, data=relation_label_list)
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
        *(read_group[ConstName1.PURE_OBJECTS][()]),
    )]
    # logger.debug(entity_list)
    entity_label_list = ['holmes', 'watson'] + [b.decode('utf-8') for b in (
        *(read_group[ConstName1.PEOPLE_LABEL][()]),
        *([b'' for _ in read_group[ConstName1.STORIES][()]]),
        *(read_group[ConstName1.OBJECTS_LABEL][()]),
    )]
    relation_list = [b.decode('utf-8') for b in read_group[ConstName1.ACTIONS][()]]
    relation_label_list = [b.decode('utf-8') for b in read_group[ConstName1.ACTIONS_LABEL][()]]
    entity_dict = {e: i for i, e in enumerate(entity_list)}
    relation_dict = {r: i for i, r in enumerate(relation_list)}

    svo_triple = np.array([
        [entity_dict[s.decode('utf-8')], relation_dict[v.decode('utf-8')], entity_dict[o.decode('utf-8')]]
        for _, _, s, v, o in read_group[ConstName1.SPO_TRIPLE][()]
        # if print(s, v, o) or True
    ])
    # add reverse
    relation_len_no_reverse = len(relation_list)
    relation_list = relation_list + [ConstName2.keyREVERSE(r) for r in relation_list]
    is_rev_list = np.concatenate(
        [np.zeros(relation_len_no_reverse, dtype=bool), np.ones(relation_len_no_reverse, dtype=bool)]
    )
    assert len(relation_list) == len(is_rev_list) and len(is_rev_list) == 2 * relation_len_no_reverse
    svo_triple_reverse = svo_triple[:, (2, 1, 0)]
    svo_triple_reverse[:, 1] += relation_len_no_reverse
    # concat
    svo_triple = np.concatenate([svo_triple, svo_triple_reverse])

    with (h5py.File(ConstName2.WRITE_SVO_INFO_FILE(title), 'a') as fw_info,
          h5py.File(ConstName2.WRITE_SVO_TRAIN_FILE(title), 'a') as fw_train):
        write2_write_triples(fw_info, fw_train, entity_list, entity_label_list, relation_list, relation_label_list,
                             is_rev_list, svo_triple)
        pass


def write2_sro(title: str, read_group: Group, general_read_group: Group):
    entity_list = [HOLMES_ALT_NAME, WATSON_ALT_NAME] + [b.decode('utf-8') for b in (
        *(read_group[ConstName1.PEOPLE][()]),
        *(read_group[ConstName1.PURE_OBJECTS][()]),
        *(read_group[ConstName1.STORIES][()]),
        *(read_group[ConstName1.ACTIONS][()]),
    )]
    entity_label_list = ['Holmes', 'Watson'] + [b.decode('utf-8') for b in (
        *(read_group[ConstName1.PEOPLE_LABEL][()]),
        *(read_group[ConstName1.PURE_OBJECTS_LABEL][()]),
        *[b'' for _ in read_group[ConstName1.STORIES][()]],
        *(read_group[ConstName1.ACTIONS_LABEL][()]),
    )]
    entity_dict = {e: i for i, e in enumerate(entity_list)}
    entity_label_dict = {e: l for e, l in zip(entity_list, entity_label_list)}
    assert len(entity_dict) == len(entity_label_dict)
    relation_list = [b.decode('utf-8') for b in general_read_group[ConstName1.RELATION][()]]
    relation_label_list = ['' for _ in general_read_group[ConstName1.RELATION][()]]
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

    with (h5py.File(ConstName2.WRITE_SRO_INFO_FILE(title), 'a') as fw_info,
          h5py.File(ConstName2.WRITE_SRO_TRAIN_FILE(title), 'a') as fw_train):
        write2_write_triples(fw_info, fw_train,
                             entity_list, entity_label_list, relation_list, relation_label_list,
                             np.zeros(len(relation_list), dtype=bool), sro_triple,
                             str_list_for_hdf5(sro_triple_raw))


def write2_():
    with (h5py.File(ConstName1.FILE_NAME, 'r') as fr, ):
        logger.debug(f"save at: {get_pure_path(ConstName1.FILE_NAME)}")
        general_read_group = fr[GENERAL]
        for title_ja, (title, _, _, _, _) in title_len_dict.items():
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
    # noinspection PyTypeChecker
    triples_list: list[tuple[int, int, int]] = triples.tolist()
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
    for title in [_t for _, (_t, _, _, _, _) in title_len_dict.items()] + [ALL_TITLE]:
        with (h5py.File(ConstName2.WRITE_SVO_TRAIN_FILE(title), 'r') as fr_train,
              h5py.File(ConstName2.WRITE_SRO_ALL_TAIL_FILE(title), 'a') as fw_all_tail, ):
            write3_write_title(fr_train, fw_all_tail)


def main():
    write_()
    read_()
    write2_()
    write3_()


if __name__ == '__main__':
    main()
    pass
