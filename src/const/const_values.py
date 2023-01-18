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
# about training loss tags especially for triple.
STORY_LOSS: Final[str] = 'story_loss'
RELATION_LOSS: Final[str] = 'relation_loss'
OBJECT_LOSS: Final[str] = 'entity_loss'
LOSS_NAME3: Final[tuple[str, str, str]] = (STORY_LOSS, RELATION_LOSS, OBJECT_LOSS)
# about predicate loss tags especially for triple.
STORY_PRED: Final[str] = 'story_pred'
RELATION_PRED: Final[str] = 'relation_pred'
ENTITY_PRED: Final[str] = 'entity_pred'
PRED_NAME3: Final[tuple[str, str, str]] = (STORY_PRED, RELATION_PRED, ENTITY_PRED)
# about answer loss tags especially for triple.
STORY_ANS: Final[str] = 'story_ans'
RELATION_ANS: Final[str] = 'relation_ans'
OBJECT_ANS: Final[str] = 'object_ans'
ANS_NAME3: Final[tuple[str, str, str]] = (STORY_ANS, RELATION_ANS, OBJECT_ANS)
# about accuracy loss tags especially for triple.
STORY_ACCURACY: Final[str] = 'story_accuracy'
RELATION_ACCURACY: Final[str] = 'relation_accuracy'
ENTITY_ACCURACY: Final[str] = 'entity_accuracy'
ACCURACY_NAME3: Final[tuple[str, str, str]] = (STORY_ACCURACY, RELATION_ACCURACY, ENTITY_ACCURACY)
# about metric tags
METRIC_NAMES: Final[tuple[str, str, str, str, str, str, str]] = (
    LOSS, STORY_LOSS, RELATION_LOSS, OBJECT_LOSS, STORY_ACCURACY, RELATION_ACCURACY, ENTITY_ACCURACY)

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
