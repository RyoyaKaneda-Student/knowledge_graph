#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Const Values

"""
from pathlib import Path
from typing import Final, Callable

PROJECT_DIR = Path(__file__).resolve().parents[2]


# general words
CPU: Final[str] = 'cpu'
CUDA: Final[str] = 'cuda'
TRAIN: Final[str] = 'train'
TEST: Final[str] = 'test'
VALID: Final[str] = 'valid'
PRE_TRAIN: Final[str] = 'pre_train'
PRE_VALID: Final[str] = 'pre_valid'
STUDY: Final[str] = 'study'
MODEL: Final[str] = 'model'
PARAMS: Final[str] = 'params'
LR: Final[str] = 'lr'
OPTIMIZER: Final[str] = 'optimizer'
# about training tags
LOSS: Final[str] = 'loss'
PRED: Final[str] = 'pred'
ANS: Final[str] = 'ans'
# about lr
LR_STORY: Final[str] = 'lr_story'
LR_RELATION: Final[str] = 'lr_relation'
LR_ENTITY: Final[str] = 'lr_entity'
LOSS_FUNCTION: Final[str] = 'loss_function'
# about training loss tags especially for triple.
HEAD_LOSS: Final[str] = 'story_loss'
RELATION_LOSS: Final[str] = 'relation_loss'
TAIL_LOSS: Final[str] = 'entity_loss'
LOSS_NAME3: Final[tuple[str, str, str]] = (HEAD_LOSS, RELATION_LOSS, TAIL_LOSS)
# about predicate loss tags especially for triple.
HEAD_PRED: Final[str] = 'story_pred'
RELATION_PRED: Final[str] = 'relation_pred'
TAIL_PRED: Final[str] = 'entity_pred'
PRED_NAME3: Final[tuple[str, str, str]] = (HEAD_PRED, RELATION_PRED, TAIL_PRED)
# about answer loss tags especially for triple.
HEAD_ANS: Final[str] = 'story_ans'
RELATION_ANS: Final[str] = 'relation_ans'
TAIL_ANS: Final[str] = 'object_ans'
ANS_NAME3: Final[tuple[str, str, str]] = (HEAD_ANS, RELATION_ANS, TAIL_ANS)
# about accuracy loss tags especially for triple.
STORY_ACCURACY: Final[str] = 'story_accuracy'
RELATION_ACCURACY: Final[str] = 'relation_accuracy'
ENTITY_ACCURACY: Final[str] = 'entity_accuracy'
ACCURACY_NAME3: Final[tuple[str, str, str]] = (STORY_ACCURACY, RELATION_ACCURACY, ENTITY_ACCURACY)
# about metric tags
HEAD_METRIC_NAMES: Final[tuple[str, str]] = (HEAD_LOSS, STORY_ACCURACY)
RELATION_METRIC_NAMES: Final[tuple[str, str]] = (RELATION_LOSS, RELATION_ACCURACY)
TAIL_METRIC_NAMES: Final[tuple[str, str]] = (TAIL_LOSS, ENTITY_ACCURACY)

# about train tags
TRAIN_SCALER_TAG_GETTER: Final[Callable[[str], str]] = lambda _name: f"{TRAIN}/{_name}"
TRAIN_MODEL_WEIGHT_TAG_GETTER: Final[Callable[[str], str]] = lambda _name: f"{TRAIN}/model_weight/{_name}"
VALID_SCALER_TAG_GETTER: Final[Callable[[str], str]] = lambda _name: f"{VALID}/{_name}"
# about pre-train tags
PRE_TRAIN_SCALER_TAG_GETTER: Final[Callable[[str], str]] = lambda _name: f"{PRE_TRAIN}/{_name}"
PRE_TRAIN_MODEL_WEIGHT_TAG_GETTER: Final[Callable[[str], str]] = lambda _name: f"{PRE_TRAIN}/model_weight/{_name}"
PRE_VALID_SCALER_TAG_GETTER: Final[Callable[[str], str]] = lambda _name: f"{PRE_VALID}/{_name}"

# about data_helper and dataloader
DATASETS: Final[str] = 'datasets'
DATA_HELPER: Final[str] = 'data_helper'
DATA_LOADERS: Final[str] = 'data_loaders'
TRAIN_RETURNS: Final[str] = 'train_returns'

# about all title
ACaseOfIdentity: Final[str] = 'ACaseOfIdentity'
AbbeyGrange: Final[str] = 'AbbeyGrange'
CrookedMan: Final[str] = 'CrookedMan'
DancingMen: Final = 'DancingMen'
DevilsFoot: Final[str] = 'DevilsFoot'
ResidentPatient: Final[str] = 'ResidentPatient'
SilverBlaze: Final[str] = 'SilverBlaze'
SpeckledBand: Final[str] = 'SpeckledBand'
ALL_TITLE_LIST: Final = (
    ACaseOfIdentity, AbbeyGrange, CrookedMan, DancingMen, DevilsFoot, ResidentPatient, SilverBlaze, SpeckledBand
)
# about wards
ABOUT_KILL_WORDS: Final[tuple[str, str, str]] = (
    'word.predicate:kill', 'word.predicate:notKill', 'word.predicate:beKilled')
# save folder
MOST_GOOD_CHECKPOINT_PATH: Final[str] = '{}/most_good/'
LATEST_CHECKPOINT_PATH: Final[str] = '{}/most_good/'

# about sro file and folder path
SRO_FOLDER: Final[str] = "data/processed/KGCdata/All/SRO"
SRO_ALL_INFO_FILE: Final[str] = f"{SRO_FOLDER}/info.hdf5"
SRO_ALL_TRAIN_FILE: Final[str] = f"{SRO_FOLDER}/train.hdf5"
TITLE2SRO_FILE090: Final[dict[str, str]] = {
    ACaseOfIdentity: f"{SRO_FOLDER}/train_AbbeyGrange_l090.hdf5",
    AbbeyGrange: f"{SRO_FOLDER}/train_ACaseOfIdentity_l090.hdf5",
    CrookedMan: f"{SRO_FOLDER}/train_CrookedMan_l090.hdf5",
    DancingMen: f"{SRO_FOLDER}/train_DancingMen_l090.hdf5",
    DevilsFoot: f"{SRO_FOLDER}/train_DevilsFoot_l090.hdf5",
    ResidentPatient: f"{SRO_FOLDER}/train_ResidentPatient_l090.hdf5",
    SilverBlaze: f"{SRO_FOLDER}/train_SilverBlaze_l090.hdf5",
    SpeckledBand: f"{SRO_FOLDER}/train_SpeckledBand_l090.hdf5",
}
TITLE2SRO_FILE075: Final[dict[str, str]] = {
    'AbbeyGrange': f"{SRO_FOLDER}/train_AbbeyGrange_l075.hdf5",
    'ACaseOfIdentity': f"{SRO_FOLDER}/train_ACaseOfIdentity_l075.hdf5",
    'CrookedMan': f"{SRO_FOLDER}/train_CrookedMan_l075.hdf5",
    'DancingMen': f"{SRO_FOLDER}/train_DancingMen_l075.hdf5",
    'DevilsFoot': f"{SRO_FOLDER}/train_DevilsFoot_l075.hdf5",
    'ResidentPatient': f"{SRO_FOLDER}/train_ResidentPatient_l075.hdf5",
    'SilverBlaze': f"{SRO_FOLDER}/train_SilverBlaze_l075.hdf5",
    'SpeckledBand': f"{SRO_FOLDER}/train_SpeckledBand_l075.hdf5"
}

# about svo file and folder path
SVO_FOLDER: Final[str] = "data/processed/KGCdata/All/SVO"
SVO_ALL_INFO_FILE: Final[str] = f"{SRO_FOLDER}/info.hdf5"
SVO_ALL_TRAIN_FILE: Final[str] = f"{SRO_FOLDER}/train.hdf5"
TITLE2SVO_FILE090: Final[dict[str, str]] = {
    ACaseOfIdentity: f"{SRO_FOLDER}/train_AbbeyGrange_l090.hdf5",
    AbbeyGrange: f"{SRO_FOLDER}/train_ACaseOfIdentity_l090.hdf5",
    CrookedMan: f"{SRO_FOLDER}/train_CrookedMan_l090.hdf5",
    DancingMen: f"{SRO_FOLDER}/train_DancingMen_l090.hdf5",
    DevilsFoot: f"{SRO_FOLDER}/train_DevilsFoot_l090.hdf5",
    ResidentPatient: f"{SRO_FOLDER}/train_ResidentPatient_l090.hdf5",
    SilverBlaze: f"{SRO_FOLDER}/train_SilverBlaze_l090.hdf5",
    SpeckledBand: f"{SRO_FOLDER}/train_SpeckledBand_l090.hdf5",
}
TITLE2SVO_FILE075: Final[dict[str, str]] = {
    'AbbeyGrange': f"{SRO_FOLDER}/train_AbbeyGrange_l075.hdf5",
    'ACaseOfIdentity': f"{SRO_FOLDER}/train_ACaseOfIdentity_l075.hdf5",
    'CrookedMan': f"{SRO_FOLDER}/train_CrookedMan_l075.hdf5",
    'DancingMen': f"{SRO_FOLDER}/train_DancingMen_l075.hdf5",
    'DevilsFoot': f"{SRO_FOLDER}/train_DevilsFoot_l075.hdf5",
    'ResidentPatient': f"{SRO_FOLDER}/train_ResidentPatient_l075.hdf5",
    'SilverBlaze': f"{SRO_FOLDER}/train_SilverBlaze_l075.hdf5",
    'SpeckledBand': f"{SRO_FOLDER}/train_SpeckledBand_l075.hdf5"
}

JA_TITLE2LEN_INFO = {
    "僧坊荘園": ("AbbeyGrange", 414, 372, 331, 310),
    "花婿失踪事件": ("ACaseOfIdentity", 580, 522, 464, 435),
    "背中の曲がった男": ("CrookedMan", 373, 335, 298, 279),
    "踊る人形": ("DancingMen", 231, 207, 184, 173),
    "悪魔の足": ("DevilsFoot", 489, 440, 391, 366),
    "入院患者": ("ResidentPatient", 324, 291, 259, 243),
    "白銀号事件": ("SilverBlaze", 397, 367, 317, 297),
    "マダラのひも": ("SpeckledBand", 401, 360, 320, 300)
}

EN_TITLE2LEN_INFO = {
    AbbeyGrange: (414, 372, 331, 310),
    ACaseOfIdentity: (580, 522, 464, 435),
    CrookedMan: (373, 335, 298, 279),
    DancingMen: (231, 207, 184, 173),
    DevilsFoot: (489, 440, 391, 366),
    ResidentPatient: (324, 291, 259, 243),
    SilverBlaze: (397, 367, 317, 297),
    SpeckledBand: (401, 360, 320, 300)
}
