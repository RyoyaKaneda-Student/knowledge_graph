#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Change and save the data using Knowledge Graph Challenge.

* This file is used for changing and saving the data of Knowledge Graph Challenge for useful.

Todo:
    * なんか色々あるきがする. Maybe I have many problem, but I can't write by my words.

"""
# ========== python ==========
from logging import Logger
from pathlib import Path
from urllib.parse import quote, unquote
# noinspection PyUnresolvedReferences
from typing import List, Dict, Tuple, Optional, Union, Callable, Final, Literal, Iterable, get_args, cast
import h5py
from h5py import Group
import numpy as np
import rdflib
from rdflib import Graph, URIRef, Namespace, RDF, RDFS
from rdflib.namespace import DefinedNamespace, NamespaceManager
from rdflib.term import Node
# my data
from models.datasets.data_helper import INFO_INDEX, TRAIN_INDEX
from utils.hdf5 import del_data_if_exist, str_list_for_hdf5
from utils.setup import easy_logger
from utils.typing import ConstMeta
from utils.utils import get_pure_path
# get PROJECT_DIR
from const.const_values import PROJECT_DIR


DATA_FOLDER_PATH: Final = f"{PROJECT_DIR}/data/external/KGRC-RDF"
WRITE1_FILE_PATH: Final = f"{PROJECT_DIR}/data/external/KGCdata/story_list.hdf5"
WRITE2_SVO_INFO_FILE: Final = lambda title: f"{PROJECT_DIR}/data/processed/KGCdata/{title}/SVO/info.hdf5"
WRITE2_SVO_TRAIN_FILE: Final = lambda title: f"{PROJECT_DIR}/data/processed/KGCdata/{title}/SVO/train.hdf5"
WRITE2_SRO_INFO_FILE: Final = lambda title: f"{PROJECT_DIR}/data/processed/KGCdata/{title}/SRO/info.hdf5"
WRITE2_SRO_TRAIN_FILE: Final = lambda title: f"{PROJECT_DIR}/data/processed/KGCdata/{title}/SRO/train.hdf5"

URL_DATA: Final = "http://kgc.knowledge-graph.jp/data"
URL_PREDICATE: Final = "http://kgc.knowledge-graph.jp/data/predicate/"
PREFIX_PREDICATE: Final = 'word.predicate'

KGC_TYPE: Final = 'kgc:type'
KGC_LABEL: Final = 'kgc:label'

logger: Logger = easy_logger(console_level='debug')


class KGC(DefinedNamespace):
    """Knowledge Graph Challenge's Namespaces.

    * The word "from" and "if" are Reserved words in Python, So use not just words but function.

    """
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
        """URIRef Item(the word "from" can't use naturally)
        """
        return cls._NS['from']

    @classmethod
    def if_(cls) -> URIRef:
        """URIRef Item(the word "if" can't use naturally)
        """
        return cls._NS['if']


ALL_RELATION: Final = [
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

OBJECT_SET: Final = {
    KGC.Animal, KGC.OFobj, KGC.ORobj, KGC.Object,
    KGC.PhysicalObject, KGC.Person, KGC.Place,
}
ACTION_SET: Final = {
    KGC.Action, KGC.CanAction, KGC.CanNotAction, KGC.NotAction
}
PEOPLE_SET: Final = {
    KGC.Person,
}
SCENE_SET: Final = {
    KGC.Scene, KGC.Situation, KGC.Thought, KGC.Statement, KGC.Talk
}
KGC_PREDICATE_LIST: Final[tuple[URIRef, ...]] = (KGC.subject, KGC.hasPredicate, KGC.hasProperty, KGC.what, KGC.whom,
                                                 KGC.to, KGC.where, KGC.infoSource, KGC.on, KGC.from_(),
                                                 KGC.when, KGC.time, KGC.why)

NodeList = list[Node]
URIRefList = list[URIRef]
NamespaceDict = dict[str, Namespace]
StoryList = list[str]
TripleDict = dict[str, dict[URIRef, URIRefList]]
SubjHaspredObj = tuple[StoryList, URIRefList, URIRefList, URIRefList, URIRefList]
StoryPredObj = tuple[StoryList, URIRefList, URIRefList]


def make_get_graph(title: str, data_file_path: Union[str, Path]):
    """make and get rdf graph

    * make the rdf graph and bind the words 'predicate' and the title.

    Args:
        title(str): the title.
        data_file_path(Union[str, Path]): file path.

    Returns:
        tuple[Graph, NamespaceManager, dict[str, Namespace]]: return (graph ,NamespaceManager, namespace dict)

    """
    g = Graph()
    g.parse(data_file_path)
    g.bind(PREFIX_PREDICATE, Namespace(URL_PREDICATE))
    g.bind(title, Namespace(f"{URL_DATA}/{title}/"))
    namespace_dict = {key: Namespace(value) for key, value in g.namespace_manager.namespaces()}
    return g, g.namespace_manager, namespace_dict


def change_sentence(_uriref: URIRef, namespace_manager: NamespaceManager) -> str:
    """Change one sentence.

    * If the uriref is string item, this item change to short sentence one by using namespace_manager.
    * Else if the uriref is time item, this item change to time item.
    Args:
        _uriref(URIRef): uriref item.
        namespace_manager(NamespaceManager): namespace manager.

    Returns:
        str: changed item.

    """
    if type(_uriref.toPython()) is str:
        x = str(_uriref).rsplit('/', maxsplit=1)
        if len(x) == 2:
            url_, value = x
            item = unquote(URIRef(url_ + '/' + quote(value, safe='#')).n3(namespace_manager))
        else:
            item = _uriref.toPython()
    else:
        item = "DateTime:{}".format(str(_uriref))
        pass
    return item


def replace_holmes_and_watson(title: str, item_list: Iterable[str]):
    """replace the holmes and watson tags.

    * Delete specific title and set the general title for all holmes and watson tags.

    Args:
        title(str): The specific title.
        item_list(Iterable[str]): The item list.

    Returns:
        list[str]: The item list. the items about Holmes and Watson in this list are changed to no specific title items.

    """
    return [HOLMES_ALT_NAME if item == HOLMES_TITLE_NAME(title) else
            WATSON_ALT_NAME if item == WATSON_TITLE_NAME(title) else
            item for item in item_list]


def get_shaped_str_list(title, namespace_manager, node_list: Iterable[URIRef]):
    """get shaped str list

    * Get shaped string list. shape means change_sentence and replace name.

    Args:
        title(str): title
        namespace_manager(NamespaceManager): namespace_manager
        node_list(Iterable[URIRef]): the list of node.

    Returns:
        list[str]: the list of strings.

    """
    str_list = [change_sentence(_n, namespace_manager) for _n in node_list]
    alt_name_list = replace_holmes_and_watson(title, str_list)
    return alt_name_list


# noinspection PyTypeChecker
def get_type_match_list(graph_: Graph, type_set: set[URIRef]) -> URIRefList:
    """get type match list

    * get the list of URIRef item which type is in the type_set from rdf graph.

    Args:
        graph_(Graph): graph
        type_set(set[URIRef]):

    Returns:
        list[URIRef]: URIRefList

    """
    match_list: URIRefList = list(dict.fromkeys(
        [s for s, _, o in graph_.triples((None, None, None)) if o in type_set]))
    match_list = [_l for _l in match_list if _l not in type_set]
    return match_list


def get_story_list(graph_: Graph, namespace_dict: NamespaceDict, title: str, story_length: int):
    """get story list.

    * get story item list in graph.

    Args:
        graph_(Graph):
        namespace_dict(dict[str, Namespace]):
        title(str):
        story_length(int):

    Returns:
        list[str]: the string iiis item. iiis means triple int(iii) and char(s).

    """
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


def get_triples(graph_: Graph, namespace_dict: NamespaceDict, title: str, story_list: StoryList):
    """get triples

    * get some triples from rdf graph.

    Args:
        graph_(Graph): rdf graph
        namespace_dict(dict[str, Namespace]): namespace_dict
        title(str): title
        story_list(list[str]): the list of stories

    Returns:
        tuple[TripleDict, SubjHaspredObj, StoryPredObj]:
            TripleDict is the dict. dict[iiis] == dict[_predicate][object_list]
            SubjHaspredObj is list. list[i] == []

    """

    prefix_title = namespace_dict[title]
    triple_dict: dict[str, dict[URIRef, URIRefList]] = {
        iiis: {p: list(graph_.objects(prefix_title[iiis], p)) for p in KGC_PREDICATE_LIST}
        for iiis in story_list
    }

    subj_haspred_obj = [(URIRef(f'{title}:{iiis}'), _predicate, _subject, _has_predicate, _object)
                        for iiis in story_list
                        for _subject in triple_dict[iiis][KGC.subject]  # almost always one item
                        for _has_predicate in triple_dict[iiis][KGC.hasPredicate]  # almost always one item
                        for _predicate in KGC_PREDICATE_LIST[2:]  # delete KGC.subject and predicate
                        for _object in triple_dict[iiis][_predicate]]  # almost always one item

    story_pred_obj = [(URIRef(f'{title}:{iiis}'), _predicate, _object)
                      for iiis in story_list
                      for _predicate in KGC_PREDICATE_LIST
                      for _object in triple_dict[iiis][_predicate]]

    return triple_dict, tuple(zip(*subj_haspred_obj)), tuple(zip(*story_pred_obj))


# noinspection PyTypeChecker
def get_label_list(graph_: Graph, node_list: Iterable[URIRef]):
    """get label list

    * get label list from node_list

    Args:
        graph_(Graph):
        node_list(Iterable[URIRef]):

    Returns:
        list[str]: the label list.

    """

    _label_list_list: list[list[rdflib.term.Literal]] = [
        [o for o in list(graph_.objects(_n, RDFS.label)) if o.language == 'en']
        for _n in node_list
    ]
    _label_list = [labels[0].value if len(labels) > 0 else '' for labels in _label_list_list]
    return _label_list


class TAGS1(metaclass=ConstMeta):
    """TAGS1

    * This is only const parameters about tags1.

    """
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

    @classmethod
    def ALL_LIST(cls):
        """all list of tags in TAGS1.

        Returns:
            tuple[str]: the tuple of tags in TAGS1.

        """
        return (cls.LENGTH_100, cls.LENGTH_090, cls.LENGTH_080, cls.LENGTH_075, cls.STORIES, cls.STORIES_IIIS,
                cls.OBJECTS, cls.PURE_OBJECTS, cls.ACTIONS, cls.PEOPLE,
                cls.OBJECTS_LABEL, cls.PURE_OBJECTS_LABEL, cls.PEOPLE_LABEL, cls.ACTIONS_LABEL,
                cls.SPO_TRIPLE, cls.PO_TRIPLE)


def save_info_list(
        title_group, l100, l090, l080, l075,
        story_iiis_list, story_name_list, objects_list, pure_objects_list, people_list, actions_list,
        objects_label_list, pure_objects_label_list, people_label_list, actions_label_list, spo_array, po_array,
):
    """Save function.

    * Save its items to title_group.

    Args:
        title_group(Group): title group.
        l100(int): int about l100.
        l090(int): int about l090.
        l080(int): int about l080.
        l075(int): int about l075.
        story_iiis_list(list[str]): the list of iiis.
        story_name_list(list[str]): the list of story_name(not iiis).
        objects_list(list[str]): the list of objects.
        actions_list(list[str]): the list of actions.
        people_list(list[str]): the list of people.
        objects_label_list(list[str]): the list of object labels.
        actions_label_list(list[str]): the list of action labels.
        people_label_list(list[str]): the list of people labels.
        pure_objects_list(list[str]): the list of pure object. pure means all items not used in any other lists.
        pure_objects_label_list(list[str]): the list of pure object label.
            pure means all items not used in any other lists.
        spo_array(np.ndarray): spo_array. It has (Story, predicate, subject, hasPredicate, object) in one sequence.
        po_array(np.ndarray): po_array. It has (storySubject, predicate, object) in one sequence.

    """

    def change_to_np_array(_list: Union[list, np.ndarray, None]) -> Union[list, np.ndarray, None]:
        """change items to np.ndarray if the _list is not np.ndarray

        Args:
            _list(Union[list, np.ndarray]):
                If its type is np.ndarray, return itself.
                If its type is None, return None.
                If its type is list, this list change to np.ndarray items.

        Returns:
            Union[list, np.ndarray, None]: The item which type is np.ndarray or None.

        """
        if _list is None: return None
        return str_list_for_hdf5(_list) if type(story_iiis_list) is not np.ndarray else _list

    [title_group.create_dataset(name, data=value) for name, value in (
        (TAGS1.LENGTH_100, l100),
        (TAGS1.LENGTH_090, l090),
        (TAGS1.LENGTH_080, l080),
        (TAGS1.LENGTH_075, l075),
        (TAGS1.STORIES_IIIS, change_to_np_array(story_iiis_list)),
        (TAGS1.STORIES, change_to_np_array(story_name_list)),
        (TAGS1.OBJECTS, change_to_np_array(objects_list)),
        (TAGS1.PURE_OBJECTS, change_to_np_array(pure_objects_list)),
        (TAGS1.PEOPLE, change_to_np_array(people_list)),
        (TAGS1.ACTIONS, change_to_np_array(actions_list)),
        (TAGS1.OBJECTS_LABEL, change_to_np_array(objects_label_list)),
        (TAGS1.PURE_OBJECTS_LABEL, change_to_np_array(pure_objects_label_list)),
        (TAGS1.PEOPLE_LABEL, change_to_np_array(people_label_list)),
        (TAGS1.ACTIONS_LABEL, change_to_np_array(actions_label_list)),
        (TAGS1.SPO_TRIPLE, spo_array),
        (TAGS1.PO_TRIPLE, po_array),
    ) if value is not None]


def write1_one_title(f, title_ja, title, l100, l090, l080, l075):
    """write one title

    * write one title in write1

    Args:
        f(File):
        title_ja(str):
        title(str):
        l100(int):
        l090(int):
        l080(int):
        l075(int):

    Returns:
        tuple[list[str], list[str], list[str], list[str], list[str], list[str], list[str], list[str], list[str],
            np.ndarray, np.ndarray]:
            (objects_str_list, pure_objects_str_list, people_str_list, actions_str_list, story_name_list,
            objects_label_list, pure_objects_label_list, people_label_list, actions_label_list,
            subj_haspred_obj_array, story_pred_obj_array)

    """
    logger.debug(f"{title_ja=}")
    data_file_path = f"{DATA_FOLDER_PATH}/{title}.ttl"
    title_group = f.require_group(title)

    graph_, namespace_manager, namespace_dict = make_get_graph(title, data_file_path)

    story_iiis_list = get_story_list(graph_, namespace_dict, title, l100)
    story_name_list = [f'{title}:{iiis}' for iiis in story_iiis_list]
    triple_dict, subj_haspred_obj, story_pred_obj = get_triples(graph_, namespace_dict, title, story_iiis_list)

    subj_haspred_obj_shaped = [get_shaped_str_list(title, namespace_manager, item) for item in subj_haspred_obj]
    story_pred_obj_shaped = [get_shaped_str_list(title, namespace_manager, item) for item in story_pred_obj]

    def make_str_label_list(_list: list[URIRef]):
        """make_str_label_list

        """
        _str_list = get_shaped_str_list(title, namespace_manager, _list)
        _label_list = get_label_list(graph_, _list)
        return _list, _str_list, _label_list

    objects_node_list = get_type_match_list(graph_, OBJECT_SET)
    people_node_list = get_type_match_list(graph_, PEOPLE_SET)
    actions_node_list = get_type_match_list(graph_, ACTION_SET)

    scene_set = get_type_match_list(graph_, SCENE_SET)
    objects_node_list = list(dict.fromkeys(objects_node_list + list(story_pred_obj[2])))

    node_3set_mix = (people_node_list + actions_node_list + scene_set)
    pure_objects_list = [o for o in objects_node_list if o not in node_3set_mix]

    objects_node_list, objects_str_list, objects_label_list = make_str_label_list(objects_node_list)
    pure_objects_node_list, pure_objects_str_list, pure_objects_label_list = make_str_label_list(pure_objects_list)
    people_node_list, people_str_list, people_label_list = make_str_label_list(people_node_list)
    actions_node_list, actions_str_list, actions_label_list = make_str_label_list(actions_node_list)

    subj_haspred_obj_array = str_list_for_hdf5(subj_haspred_obj_shaped).T
    story_pred_obj_array = str_list_for_hdf5(story_pred_obj_shaped).T

    del_data_if_exist(title_group, TAGS1.ALL_LIST())
    save_info_list(
        title_group, l100, l090, l080, l075,
        story_iiis_list, story_name_list,
        objects_str_list, pure_objects_str_list, people_str_list, actions_str_list,
        objects_label_list, pure_objects_label_list, people_label_list, actions_label_list,
        subj_haspred_obj_array, story_pred_obj_array
    )
    return (
        objects_str_list, pure_objects_str_list, people_str_list, actions_str_list, story_name_list,
        objects_label_list, pure_objects_label_list, people_label_list, actions_label_list,
        subj_haspred_obj_array, story_pred_obj_array
    )


def write_():
    """Write function 1

    """
    # print(f"{DATA_FOLDER_PATH}/../data/story_list.hdf5")
    print("save at: {}".format(get_pure_path(WRITE1_FILE_PATH)))
    with h5py.File(WRITE1_FILE_PATH, 'a') as f:
        all_objects_str_list, all_pure_objects_str_list, all_people_str_list, all_actions_str_list = \
            list[str](), list[str](), list[str](), list[str]()
        all_objects_label_list, all_pure_objects_label_list, all_people_label_list, all_actions_label_list = \
            list[str](), list[str](), list[str](), list[str]()
        all_story_name_list = list[str]()
        all_subj_haspred_obj_array_list, all_story_pred_obj_array_list = list[np.ndarray](), list[np.ndarray]()

        for title_ja, (title, l100, l090, l080, l075) in title_len_dict.items():
            (
                objects_str_list, pure_objects_str_list, people_str_list, actions_str_list, story_name_list,
                objects_label_list, pure_objects_label_list, people_label_list, actions_label_list,
                spo_array, po_array
            ) = (write1_one_title(f, title_ja, title, l100, l090, l080, l075))
            all_objects_str_list += objects_str_list
            all_objects_label_list += objects_label_list
            all_pure_objects_str_list += pure_objects_str_list
            all_pure_objects_label_list += pure_objects_label_list
            all_people_str_list += people_str_list
            all_people_label_list += people_label_list
            all_actions_str_list += actions_str_list
            all_actions_label_list += actions_label_list
            all_story_name_list += story_name_list
            all_subj_haspred_obj_array_list.append(spo_array)
            all_story_pred_obj_array_list.append(po_array)

        def del_duplication(_str_list: list[str], _label_list: list[str]):
            """delete str_list duplication and reflected in label_list.

            Args:
                _str_list: list[str]: item names list.
                _label_list: list[str]: item labels list.

            Returns:
                tuple[list[str], list[str]]: not duplication str_list and label_list.
                    Note that sometimes label_list is duplicate but it is OK.

            """
            _dict = {key: value for key, value in zip(_str_list, _label_list)}
            return list(_dict.keys()), list(_dict.values())

        all_objects_str_list, all_objects_label_list = del_duplication(all_objects_str_list, all_objects_label_list)
        all_pure_objects_str_list, all_pure_objects_label_list = del_duplication(
            all_pure_objects_str_list, all_pure_objects_label_list)
        all_people_str_list, all_people_label_list = del_duplication(all_people_str_list, all_people_label_list)
        all_actions_str_list, all_actions_label_list = del_duplication(all_actions_str_list, all_actions_label_list)
        #
        all_subj_haspred_obj_array = np.concatenate(all_subj_haspred_obj_array_list)
        all_story_pred_obj_array = np.concatenate(all_story_pred_obj_array_list)
        # general
        title_group = f.require_group(GENERAL)
        del_data_if_exist(title_group, [TAGS1.PEOPLE, TAGS1.RELATION])
        title_group.create_dataset(TAGS1.PEOPLE, data=str_list_for_hdf5([HOLMES_ALT_NAME, WATSON_ALT_NAME]))
        title_group.create_dataset(TAGS1.RELATION, data=str_list_for_hdf5(ALL_RELATION))
        # all
        title_group = f.require_group(ALL_TITLE)
        del_data_if_exist(title_group, TAGS1.ALL_LIST())

        # noinspection PyTypeChecker
        save_info_list(
            title_group, None, None, None, None,
            None, all_story_name_list,
            all_objects_str_list, all_pure_objects_str_list, all_people_str_list, all_actions_str_list,
            all_objects_label_list, all_pure_objects_label_list, all_people_label_list, all_actions_label_list,
            all_subj_haspred_obj_array, all_story_pred_obj_array,
        )


def read_():
    """read write1 items

    """
    with h5py.File(WRITE1_FILE_PATH, 'r') as f:
        for title_ja, (title, l100, l090, l080, l075) in title_len_dict.items():
            title_group = f.require_group(title)
            logger.debug(get_pure_path(title_group.file.filename))
            # [logger.debug(title_group[name][()]) for name in ConstName1.all_list()]


keyREVERSE: Final = lambda key: f'{key}_REVERSE'


def write2_write_triples(fw_info, fw_train, entity_list, entity_label_list, relation_list, relation_label_list,
                         is_rev_list, triple, triple_raw=None):
    """write triples by using train file.

    Args:
        fw_info(File): fw_info
        fw_train(File): fw_train
        entity_list(list[str]): entity_list
        entity_label_list(list[str]): entity_label_list
        relation_list(list[str]): relation_list
        relation_label_list(list[str]): relation_label_list
        is_rev_list(np.ndarray): is_rev_list
        triple(np.ndarray): triple
        triple_raw(np.ndarray): triple_raw

    """
    #
    logger.debug(f"info: {get_pure_path(fw_info.filename)}")
    logger.debug(f"triple: {get_pure_path(fw_train.filename)}")

    del_data_if_exist(fw_info, INFO_INDEX.ALL_INDEXES())
    fw_info.create_dataset(INFO_INDEX.ENTITY_NUM, data=len(entity_list))
    fw_info.create_dataset(INFO_INDEX.RELATION_NUM, data=len(relation_list))
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
    del_data_if_exist(fw_train, TRAIN_INDEX.ALL_INDEXES())
    fw_train.create_dataset(TRAIN_INDEX.TRIPLE, data=triple)
    if triple_raw is not None: fw_train.create_dataset(TRAIN_INDEX.TRIPLE_RAW, data=triple_raw)


def write2_svo(title: str, read_group: Group):
    """write2 about (subject, hasPredicate, objects).

    Args:
        title(str): title
        read_group(Group): read_group

    """
    entity_list = [HOLMES_ALT_NAME, WATSON_ALT_NAME] + [b.decode('utf-8') for b in (
        *(read_group[TAGS1.PEOPLE][()]),
        *(read_group[TAGS1.STORIES][()]),
        *(read_group[TAGS1.PURE_OBJECTS][()]),
    )]
    # logger.debug(entity_list)
    entity_label_list = ['holmes', 'watson'] + [b.decode('utf-8') for b in (
        *(read_group[TAGS1.PEOPLE_LABEL][()]),
        *([b'' for _ in read_group[TAGS1.STORIES][()]]),
        *(read_group[TAGS1.OBJECTS_LABEL][()]),
    )]
    relation_list = [b.decode('utf-8') for b in read_group[TAGS1.ACTIONS][()]]
    relation_label_list = [b.decode('utf-8') for b in read_group[TAGS1.ACTIONS_LABEL][()]]
    entity_dict = {e: i for i, e in enumerate(entity_list)}
    relation_dict = {r: i for i, r in enumerate(relation_list)}

    svo_triple = np.array([
        [entity_dict[s.decode('utf-8')], relation_dict[v.decode('utf-8')], entity_dict[o.decode('utf-8')]]
        for _, _, s, v, o in read_group[TAGS1.SPO_TRIPLE][()]
        # if print(s, v, o) or True
    ])
    # add reverse
    relation_len_no_reverse = len(relation_list)
    relation_list = relation_list + [keyREVERSE(r) for r in relation_list]
    is_rev_list = np.concatenate(
        [np.zeros(relation_len_no_reverse, dtype=bool), np.ones(relation_len_no_reverse, dtype=bool)]
    )
    assert len(relation_list) == len(is_rev_list) and len(is_rev_list) == 2 * relation_len_no_reverse
    svo_triple_reverse = svo_triple[:, (2, 1, 0)]
    svo_triple_reverse[:, 1] += relation_len_no_reverse
    # concat
    svo_triple = np.concatenate([svo_triple, svo_triple_reverse])

    with (h5py.File(WRITE2_SVO_INFO_FILE(title), 'a') as fw_info,
          h5py.File(WRITE2_SVO_TRAIN_FILE(title), 'a') as fw_train):
        write2_write_triples(fw_info, fw_train, entity_list, entity_label_list, relation_list, relation_label_list,
                             is_rev_list, svo_triple)
        pass


def write2_sro(title: str, read_group: Group, general_read_group: Group):
    """write2 about (story, predicate, objects).

    Args:
        title(str): title
        read_group(Group): read_group
        general_read_group(Group): general_read_group

    """
    entity_list = [HOLMES_ALT_NAME, WATSON_ALT_NAME] + [b.decode('utf-8') for b in (
        *(read_group[TAGS1.PEOPLE][()]),
        *(read_group[TAGS1.PURE_OBJECTS][()]),
        *(read_group[TAGS1.STORIES][()]),
        *(read_group[TAGS1.ACTIONS][()]),
    )]
    entity_label_list = ['Holmes', 'Watson'] + [b.decode('utf-8') for b in (
        *(read_group[TAGS1.PEOPLE_LABEL][()]),
        *(read_group[TAGS1.PURE_OBJECTS_LABEL][()]),
        *[b'' for _ in read_group[TAGS1.STORIES][()]],
        *(read_group[TAGS1.ACTIONS_LABEL][()]),
    )]
    entity_dict = {e: i for i, e in enumerate(entity_list)}
    entity_label_dict = {e: l for e, l in zip(entity_list, entity_label_list)}
    assert len(entity_dict) == len(entity_label_dict)
    relation_list = [b.decode('utf-8') for b in general_read_group[TAGS1.RELATION][()]]
    relation_label_list = ['' for _ in general_read_group[TAGS1.RELATION][()]]
    relation_dict = {r: i for i, r in enumerate(relation_list)}
    sro_triple = np.array([
        [entity_dict[s.decode('utf-8')], relation_dict[r.decode('utf-8')], entity_dict[o.decode('utf-8')]]
        for s, r, o in read_group[TAGS1.PO_TRIPLE][()]
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

    with (h5py.File(WRITE2_SRO_INFO_FILE(title), 'a') as fw_info,
          h5py.File(WRITE2_SRO_TRAIN_FILE(title), 'a') as fw_train):
        write2_write_triples(fw_info, fw_train,
                             entity_list, entity_label_list, relation_list, relation_label_list,
                             np.zeros(len(relation_list), dtype=bool), sro_triple,
                             str_list_for_hdf5(sro_triple_raw))


def write2_():
    """ Write function 2

    """
    with (h5py.File(WRITE1_FILE_PATH, 'r') as fr, ):
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


def main():
    """Main function

    """
    # write 1
    write_()
    # write 2
    write2_()


if __name__ == '__main__':
    main()
    pass
