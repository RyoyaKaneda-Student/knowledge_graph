#!/usr/bin/python
# -*- coding: utf-8 -*-
from argparse import Namespace
# noinspection PyUnresolvedReferences
from collections import namedtuple
# ========== python ==========
from itertools import chain
from logging import Logger
from pathlib import Path
# noinspection PyUnresolvedReferences
from typing import List, Dict, Tuple, Optional, Union, Callable, Final, Literal, get_args, cast
# Machine learning
import h5py
import numpy as np
import optuna
import pandas as pd
# torch
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import Timer, Checkpoint, global_step_from_engine, DiskSaver
from ignite.metrics import Average, Accuracy

# My Models
from models.KGModel.kg_story_transformer import (
    KgStoryTransformer01, KgStoryTransformer02, add_bos, KgStoryTransformer03, KgStoryTransformer03preInit,
    KgStoryTransformer00, KgStoryTransformer)
from models.datasets.data_helper import MyDataHelper, DefaultTokens, DefaultIds, SpecialTokens01 as SpecialTokens, \
    MyDataLoaderHelper
from models.datasets.datasets import (
    StoryTriple, StoryTripleForValid,
)
# My utils
from utils.torch import save_model, torch_fix_seed, DeviceName, force_cpu_decorator
from utils.typing import ConstMeta
from utils.utils import version_check, elapsed_time_str

PROJECT_DIR = Path(__file__).resolve().parents[1]

# About words used as tags
CPU: Final[str] = 'cpu'
TRAIN: Final[str] = 'train'
PRE_TRAIN: Final[str] = 'pre_train'
TEST: Final[str] = 'test'
MRR: Final[str] = 'mrr'
HIT_: Final[str] = 'hit_'
STUDY: Final[str] = 'study'
MODEL: Final[str] = 'model'
PARAMS: Final[str] = 'params'
LR: Final[str] = 'lr'
OPTIMIZER: Final[str] = 'optimizer'
TRAINER: Final[str] = 'trainer'
EVALUATOR: Final[str] = 'evaluator'
CHECKPOINTER: Final[str] = 'checkpointer'
CHECKPOINTER_GOOD_LOSS: Final[str] = 'checkpointer_good_loss'
CHECKPOINTER_LAST: Final[str] = 'checkpointer_last'
DATA_HELPER: Final[str] = 'data_helper'
DATA_LOADERS: Final[str] = 'data_loaders'

ALL_TAIL: Final[str] = 'all_tail'
TRIPLE: Final[str] = 'triple'
DATASETS: Final[str] = 'datasets'
TRAIN_ITEMS: Final[str] = 'train_items'

STORY_RELATION_ENTITY: Final[tuple[str, str, str]] = ('story', 'relation', 'entity')

LOSS: Final[str] = 'loss'
STORY_LOSS: Final[str] = 'story_loss'
RELATION_LOSS: Final[str] = 'relation_loss'
OBJECT_LOSS: Final[str] = 'entity_loss'
LOSS_NAME3: Final[tuple[str, str, str]] = (STORY_LOSS, RELATION_LOSS, OBJECT_LOSS)

STORY_PRED: Final[str] = 'story_pred'
RELATION_PRED: Final[str] = 'relation_pred'
ENTITY_PRED: Final[str] = 'entity_pred'
PRED_NAME3: Final[tuple[str, str, str]] = (STORY_PRED, RELATION_PRED, ENTITY_PRED)

STORY_ANS: Final[str] = 'story_ans'
RELATION_ANS: Final[str] = 'relation_ans'
OBJECT_ANS: Final[str] = 'object_ans'
ANS_NAME3: Final[tuple[str, str, str]] = (STORY_ANS, RELATION_ANS, OBJECT_ANS)

STORY_ACCURACY: Final[str] = 'story_accuracy'
RELATION_ACCURACY: Final[str] = 'relation_accuracy'
ENTITY_ACCURACY: Final[str] = 'entity_accuracy'
ACCURACY_NAME3: Final[tuple[str, str, str]] = (STORY_ACCURACY, RELATION_ACCURACY, ENTITY_ACCURACY)

METRIC_NAMES: Final[tuple[str, str, str, str, str, str, str]] = (
    LOSS, STORY_LOSS, RELATION_LOSS, OBJECT_LOSS, STORY_ANS, RELATION_ANS, OBJECT_ANS)

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

ABOUT_KILL_WORDS: Final[tuple[str, str, str]] = (
    'word.predicate:kill', 'word.predicate:notKill', 'word.predicate:beKilled')

SRO_FOLDER: Final[str] = "data/processed/KGCdata/All/SRO"

SRO_ALL_INFO_FILE: Final[str] = f"{SRO_FOLDER}/info.hdf5"
SRO_ALL_TRAIN_FILE: Final[str] = f"{SRO_FOLDER}/train.hdf5"

TITLE2FILE090: Final[dict[str, str]] = {
    ACaseOfIdentity: f"{SRO_FOLDER}/train_AbbeyGrange_l090.hdf5",
    AbbeyGrange: f"{SRO_FOLDER}/train_ACaseOfIdentity_l090.hdf5",
    CrookedMan: f"{SRO_FOLDER}/train_CrookedMan_l090.hdf5",
    DancingMen: f"{SRO_FOLDER}/train_DancingMen_l090.hdf5",
    DevilsFoot: f"{SRO_FOLDER}/train_DevilsFoot_l090.hdf5",
    ResidentPatient: f"{SRO_FOLDER}/train_ResidentPatient_l090.hdf5",
    SilverBlaze: f"{SRO_FOLDER}/train_SilverBlaze_l090.hdf5",
    SpeckledBand: f"{SRO_FOLDER}/train_SpeckledBand_l090.hdf5",
}

TITLE2FILE075: Final[dict[str, str]] = {
    'AbbeyGrange': f"{SRO_FOLDER}/train_AbbeyGrange_l075.hdf5",
    'ACaseOfIdentity': f"{SRO_FOLDER}/train_ACaseOfIdentity_l075.hdf5",
    'CrookedMan': f"{SRO_FOLDER}/train_CrookedMan_l075.hdf5",
    'DancingMen': f"{SRO_FOLDER}/train_DancingMen_l075.hdf5",
    'DevilsFoot': f"{SRO_FOLDER}/train_DevilsFoot_l075.hdf5",
    'ResidentPatient': f"{SRO_FOLDER}/train_ResidentPatient_l075.hdf5",
    'SilverBlaze': f"{SRO_FOLDER}/train_SilverBlaze_l075.hdf5",
    'SpeckledBand': f"{SRO_FOLDER}/train_SpeckledBand_l075.hdf5"
}

MOST_GOOD_CHECKPOINT_PATH: Final[str] = '{}/most_good/'
LATEST_CHECKPOINT_PATH: Final[str] = '{}/most_good/'


class ModelVersion(metaclass=ConstMeta):
    V01: Final = '01'
    V02: Final = '02'
    V03: Final = '03'
    V03a: Final = '03a'

    @classmethod
    def ALL_LIST(cls) -> tuple:
        return cls.V01, cls.V02, cls.V03, cls.V03a


SEED: Final = 42


def setup_parser(args: Namespace = None) -> Namespace:
    """
    Args:
        args:

    Returns:

    """
    import argparse  # 1. argparseをインポート
    parser = argparse.ArgumentParser(description='This is make and training source code for KGC.')
    paa = parser.add_argument
    paa('--notebook', help='if use notebook, use this argument.', action='store_true')
    paa('--console-level', help='log level on console', type=str, default='debug', choices=['info', 'debug'])
    paa('--logfile', help='the path of saving log', type=str, default='log/test.log')
    paa('--param-file', help='the path of saving param', type=str, default='log/param.pkl')
    paa('--device-name', help=DeviceName.ALL_INFO, type=str, default=DeviceName.CPU, choices=DeviceName.ALL_LIST)
    paa('--train-anyway', help='It will not be reproducible, but it could be faster.', action='store_true')
    # save dir setting
    parser_group01 = parser.add_argument_group('dir and path', 'There are the setting of training setting dir or path.')
    paa1 = parser_group01.add_argument
    paa1('--tensorboard-dir', type=str, default='log/tensorboard/', help='tensorboard direction')
    paa1('--checkpoint-dir', type=str, default='log/checkpoint/', help='tensorboard direction')
    paa1('--model-path', type=str, required=True, help='model path')
    paa1('--resume-from-checkpoint', action='store_true', help='if use checkpoint, use this argument.')
    paa1('--resume-from-last-point', action='store_true', help='if use checkpoint, use this argument.')
    paa1('--only-load-trainer-evaluator', action='store_true',
         help='load only mode. not training. use it for valid model.', )
    paa1('--resume-checkpoint-path', type=str, help='if use checkpoint, use this argument.')
    # use title setting
    parser_group02 = parser.add_argument_group('use title setting', 'There are the setting of training title.')
    paa2 = parser_group02.add_argument
    paa2('--pre-train', help="Put on if you are doing pre-training", action='store_true')
    paa2('--train-valid-test', help='', action='store_true')
    paa2('--only-train', help='', action='store_true')
    paa2('--use-for-challenge100', help='', action='store_true')
    paa2('--use-for-challenge090', help='', action='store_true')
    paa2('--use-for-challenge075', help='', action='store_true')
    paa2('--use-title', help=' or '.join(ALL_TITLE_LIST), type=str, choices=ALL_TITLE_LIST)
    # optuna setting
    parser_group03 = parser.add_argument_group('optuna setting', 'There are the setting of optuna.')
    paa3 = parser_group03.add_argument
    paa3('--do-optuna', action='store_true', help="do optuna")
    paa3('--optuna-file', type=str, help='optuna file')
    paa3('--study-name', type=str, help='optuna study-name')
    paa3('--n-trials', type=int, help='optuna n-trials')
    # special num count
    parser_group04 = parser.add_argument_group('special token setting', 'There are the setting of special token.')
    paa4 = parser_group04.add_argument
    paa4('--story-special-num', help='story special num', type=int, default=5)
    paa4('--relation-special-num', help='relation special num', type=int, default=5)
    paa4('--entity-special-num', help='entity special num', type=int, default=5)
    # e special
    parser_group041 = parser.add_argument_group(
        'special (tail) embedding token setting', 'There are the setting of special (tail) embedding token setting.')
    paa41 = parser_group041.add_argument
    paa41('--padding-token-e', help='padding', type=int, default=DefaultIds.PAD_E_DEFAULT_ID)
    paa41('--cls-token-e', help='cls', type=int, default=DefaultIds.CLS_E_DEFAULT_ID)
    paa41('--mask-token-e', help='mask', type=int, default=DefaultIds.MASK_E_DEFAULT_ID)
    paa41('--sep-token-e', help='sep', type=int, default=DefaultIds.SEP_E_DEFAULT_ID)
    paa41('--bos-token-e', help='bos', type=int, default=DefaultIds.BOS_E_DEFAULT_ID)
    # r special
    parser_group042 = parser.add_argument_group(
        'special (tail) embedding token setting', 'There are the setting of special relation embedding token setting.')
    paa42 = parser_group042.add_argument
    paa42('--padding-token-r', help='padding', type=int, default=DefaultIds.PAD_R_DEFAULT_ID)
    paa42('--cls-token-r', help='cls', type=int, default=DefaultIds.CLS_R_DEFAULT_ID)
    paa42('--mask-token-r', help='mask', type=int, default=DefaultIds.MASK_R_DEFAULT_ID)
    paa42('--sep-token-r', help='sep', type=int, default=DefaultIds.SEP_R_DEFAULT_ID)
    paa42('--bos-token-r', help='bos', type=int, default=DefaultIds.BOS_R_DEFAULT_ID)
    # story
    parser_group043 = parser.add_argument_group(
        'special (tail) embedding token setting', 'There are the setting of special (head) embedding token setting.')
    paa43 = parser_group043.add_argument
    paa43('--padding-token-s', help='padding', type=int, default=DefaultIds.PAD_E_DEFAULT_ID)
    paa43('--cls-token-s', help='cls', type=int, default=DefaultIds.CLS_E_DEFAULT_ID)
    paa43('--mask-token-s', help='mask', type=int, default=DefaultIds.MASK_E_DEFAULT_ID)
    paa43('--sep-token-s', help='sep', type=int, default=DefaultIds.SEP_E_DEFAULT_ID)
    paa43('--bos-token-s', help='bos', type=int, default=DefaultIds.BOS_E_DEFAULT_ID)
    # model
    parser_group05 = parser.add_argument_group('model setting', 'There are the setting of model params.')
    paa5 = parser_group05.add_argument
    paa5('--model-version', type=str, choices=ModelVersion.ALL_LIST(), help='model version.')
    paa5('--embedding-dim', help='The embedding dimension. Default: 128', type=int, default=128)
    paa5('--entity-embedding-dim', help='The embedding dimension. Default: 128', type=int, default=128)
    paa5('--relation-embedding-dim', help='The embedding dimension. Default: 128', type=int, default=128)
    paa5('--separate-head-and-tail', action='store_true', default=False,
         help='If True, it head Embedding and tail Embedding are different.')
    paa5('--batch-size', help='batch size', type=int, default=4)
    paa5('--max-len', metavar='MAX-LENGTH', help='max length of 1 batch. default: 256', type=int, default=256)
    paa5('--no-use-pe', action='store_true', help='to check pe(position encoding) power, we have to make no pe model')
    paa5('--init-embedding-using-bert', action='store_true',
         help='if it is set and the model is 03a, it will be pre_init by bert')
    # mask percent
    parser_group051 = parser.add_argument_group(
        'model setting of mask-percent', 'MUST mask-mask + mask-random + mask-nomask == 1.00.')
    paa51 = parser_group051.add_argument
    paa51('--mask-percent', help='default: 0.15', metavar='mask-rate', type=float, default=0.15)
    paa51('--mask-mask-percent', help='default: 0.80', metavar='mask-rate', type=float, default=0.80)
    paa51('--mask-random-percent', help='default: 0.10', metavar='random-rate', type=float, default=0.10)
    paa51('--mask-nomask-percent', help='default: 0.10', metavar='nomask-rate', type=float, default=0.10)
    # transformer
    parser_group052 = parser.add_argument_group(
        'model setting of transformer', 'There are the setting of transformer params in model.')
    paa52 = parser_group052.add_argument
    paa52('--nhead', type=int, default=4, metavar='N', help='nhead. Default: 4.')
    paa52('--num-layers', type=int, default=4, metavar='NUM', help='num layers. Default: 4.')
    paa52('--dim-feedforward', type=int, default=1028, metavar='DIM', help='dim of feedforward. Default: 1028.')
    paa52('--transformer-drop', type=float, default=0.1, metavar='DROP_RATE', help='transformer-drop. Default: 0.1.')
    paa52('--position-encoder-drop', type=float, default=0.1, metavar='DROP_RATE',
          help='position-encoder-drop. Default: 0.1.')
    # optimizer
    parser_group06 = parser.add_argument_group('model optimizer setting',
                                               'There are the setting of model optimizer params.')
    paa6 = parser_group06.add_argument
    paa6('--lr', type=float, default=0.003, help='learning rate (default: 0.003)')
    paa6('--lr-story', type=float, help='learning rate (default: same as --lr)')
    paa6('--lr-relation', type=float, help='learning rate (default: same as --lr)')
    paa6('--lr-entity', type=float, help='learning rate (default: same as --lr)')
    paa6('--valid-interval', type=int, default=1, help='valid-interval', )
    paa6('--loss-weight-story', type=float, default=1., help='loss-weight-story')
    paa6('--loss-weight-relation', type=float, default=1., help='loss-weight-relation')
    paa6('--loss-weight-entity', type=float, default=1., help='loss-weight-entity')
    paa6('--epoch', help='max epoch', type=int, default=2)

    args = parser.parse_args(args=args)
    return args


def pre_training(args: Namespace, hyper_params, data_helper, data_loaders, model, *, logger: Logger,
                 summary_writer: SummaryWriter):
    """

    Args:
        args(Namespace):
        hyper_params(tuple):
        data_helper(MyDataHelper):
        data_loaders(MyDataLoaderHelper):
        model(KgStoryTransformer01):
        summary_writer(SummaryWriter):
        logger(Logger):

    Returns:

    """
    lr, lr_story, lr_relation, lr_entity, loss_weight_story, loss_weight_relation, loss_weight_entity = hyper_params
    device: torch.device = args.device
    non_blocking = True
    model.to(device)
    max_len = args.max_len

    entity_num, relation_num = len(data_helper.processed_entities), len(data_helper.processed_relations)
    max_epoch = args.epoch
    mask_token_e, mask_token_r = args.mask_token_e, args.mask_token_r
    logger.debug(f"{entity_num=}, {relation_num=}")
    modules = {_name: _module for _name, _module in model.named_children()}
    logger.debug("model modules: " + ', '.join(list(modules.keys())))
    del modules['head_maskdlm'], modules['relation_maskdlm'], modules['tail_maskdlm']
    optim_list = [
                     {PARAMS: _module.parameters(), LR: lr} for _name, _module in modules.items()
                 ] + [
                     {PARAMS: model.head_maskdlm.parameters(), LR: lr_story},
                     {PARAMS: model.relation_maskdlm.parameters(), LR: lr_relation},
                     {PARAMS: model.tail_maskdlm.parameters(), LR: lr_entity},
                 ]

    opt = torch.optim.Adam(optim_list)
    loss_fn_entity = torch.nn.CrossEntropyLoss(weight=torch.ones(entity_num).to(device))
    loss_fn_relation = torch.nn.CrossEntropyLoss(weight=torch.ones(relation_num).to(device))
    checkpoint_dir = args.checkpoint_dir
    # checkpoint_dir = CHECKPOINT_DIR.format(line_up_key_value(pid=args.pid, uid=uid))
    train = data_loaders.train_dataloader
    valid = data_loaders.valid_dataloader if args.train_valid_test else None
    train_triple = train.dataset.triple
    # mask percents
    mask_percent = args.mask_percent
    mask_mask_percent = mask_percent * args.mask_mask_percent
    mask_nomask_percent = mask_percent * args.mask_nomask_percent
    mask_random_percent = mask_percent * args.mask_random_percent
    logger.debug(f"{mask_percent=}, {mask_mask_percent=}, {mask_nomask_percent=}, {mask_random_percent=}")
    assert mask_mask_percent + mask_nomask_percent + mask_random_percent + (1 - mask_percent) == 1.

    index2count_head = torch.bincount(train_triple[:, 0], minlength=entity_num).to(torch.float).to(device)
    index2count_relation = torch.bincount(train_triple[:, 1], minlength=relation_num).to(torch.float).to(device)
    index2count_tail = torch.bincount(train_triple[:, 2], minlength=entity_num).to(torch.float).to(device)

    # torch.from_numpy(data_helper.processed_id2count_entity).to(device, non_blocking=non_blocking)
    # torch.from_numpy(data_helper.processed_id2count_relation).to(device, non_blocking=non_blocking)

    def cpu_deep_copy_or_none(_tensor: Optional[torch.Tensor]):
        return _tensor.to(CPU, non_blocking=non_blocking).detach().clone() if _tensor is not None else None

    def mask_function(_random_all, _value, _mask_token, weights) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _mask_filter = torch.lt(_random_all, mask_percent)
        _mask_ans = _value[_mask_filter].detach().clone()
        _mask_value = _value[_mask_filter]

        _random = _random_all[_mask_filter]
        _mask_random_filter = torch.lt(_random, mask_random_percent)  # <
        _mask_mask_filter = torch.ge(_random, (mask_nomask_percent + mask_random_percent))  # >=
        _mask_value[_mask_random_filter] = torch.multinomial(
            weights, torch.count_nonzero(_mask_random_filter).item(), replacement=True)
        _mask_value[_mask_mask_filter] = _mask_token
        return _mask_filter, _mask_ans, _mask_value

    def train_step(_, batch) -> dict:
        model.train()
        triple = batch
        batch_size = triple.shape[0]
        assert triple.shape == (batch_size, max_len, 3)
        # train start
        opt.zero_grad()
        triple: torch.Tensor = triple.to(device, non_blocking=non_blocking)

        mask_filter_story, mask_ans_story, mask_value_story = mask_function(
            torch.rand((batch_size, max_len)), triple[:, :, 0], mask_token_e, index2count_head)
        mask_filter_relation, mask_ans_relation, mask_value_relation = mask_function(
            torch.rand((batch_size, max_len)), triple[:, :, 1], mask_token_r, index2count_relation)
        mask_filter_object, mask_ans_object, mask_value_object = mask_function(
            torch.rand((batch_size, max_len)), triple[:, :, 2], mask_token_e, index2count_tail)

        triple[:, :, 0][mask_filter_story] = mask_value_story
        triple[:, :, 1][mask_filter_relation] = mask_value_relation
        triple[:, :, 2][mask_filter_object] = mask_value_object

        _, (story_pred, relation_pred, entity_pred) = \
            model(triple, mask_filter_story, mask_filter_relation, mask_filter_object)

        loss: torch.Tensor = torch.tensor(0, dtype=torch.float).to(device)
        story_loss, relation_loss, object_loss = None, None, None
        if len(mask_ans_story) > 0:
            story_loss = loss_fn_entity(story_pred, mask_ans_story)
            loss += story_loss * loss_weight_story
        if len(mask_ans_relation) > 0:
            relation_loss = loss_fn_relation(relation_pred, mask_ans_relation)
            loss += relation_loss * loss_weight_relation
        if len(mask_ans_object) > 0:
            object_loss = loss_fn_entity(entity_pred, mask_ans_object)
            loss += object_loss * loss_weight_entity

        loss.backward()
        opt.step()

        # return values
        return_dict = {
            STORY_ANS: cpu_deep_copy_or_none(mask_ans_story),
            RELATION_ANS: cpu_deep_copy_or_none(mask_ans_relation),
            OBJECT_ANS: cpu_deep_copy_or_none(mask_ans_object),
            STORY_PRED: cpu_deep_copy_or_none(story_pred),
            RELATION_PRED: cpu_deep_copy_or_none(relation_pred),
            ENTITY_PRED: cpu_deep_copy_or_none(entity_pred),
            STORY_LOSS: cpu_deep_copy_or_none(story_loss),
            RELATION_LOSS: cpu_deep_copy_or_none(relation_loss),
            OBJECT_LOSS: cpu_deep_copy_or_none(object_loss),
            LOSS: cpu_deep_copy_or_none(loss),
        }
        return return_dict

    @torch.no_grad()
    def valid_step(_, batch) -> dict:
        model.eval()
        triple: torch.Tensor = batch[0].to(device, non_blocking=non_blocking)
        valid_filter: torch.Tensor = batch[1].to(device, non_blocking=non_blocking)
        # triple.shape == (batch, max_len, 3)
        # valid_filter.shape == (batch, max_len)

        valid_ans_story = triple[:, :, 0][valid_filter]
        valid_ans_relation = triple[:, :, 1][valid_filter]
        valid_ans_object = triple[:, :, 2][valid_filter]

        triple_for_valid = triple.clone()
        triple_for_valid[:, :, 0][valid_filter] = mask_token_e
        _, (story_pred, _, _) = model(triple_for_valid, valid_filter, None, None)

        triple_for_valid = triple.clone()
        triple_for_valid[:, :, 1][valid_filter] = mask_token_r
        _, (_, relation_pred, _) = model(triple_for_valid, None, valid_filter, None)

        triple_for_valid = triple.clone()
        triple_for_valid[:, :, 2][valid_filter] = mask_token_e
        _, (_, _, entity_pred) = model(triple_for_valid, None, None, valid_filter)

        loss: torch.Tensor = torch.tensor(0, dtype=torch.float).to(device)
        story_loss, relation_loss, object_loss = None, None, None
        if len(valid_ans_story) > 0:
            story_loss = loss_fn_entity(story_pred, valid_ans_story)
            loss += story_loss  # * valid_ans_story
            relation_loss = loss_fn_relation(relation_pred, valid_ans_relation)
            loss += relation_loss  # * valid_ans_relation
            object_loss = loss_fn_entity(entity_pred, valid_ans_object)
            if object_loss < 0: raise ValueError("error")
            loss += object_loss  # * valid_ans_object

        # return dict
        return_dict = {
            STORY_ANS: cpu_deep_copy_or_none(valid_ans_story),
            RELATION_ANS: cpu_deep_copy_or_none(valid_ans_relation),
            OBJECT_ANS: cpu_deep_copy_or_none(valid_ans_object),
            STORY_PRED: cpu_deep_copy_or_none(story_pred),
            RELATION_PRED: cpu_deep_copy_or_none(relation_pred),
            ENTITY_PRED: cpu_deep_copy_or_none(entity_pred),
            LOSS: cpu_deep_copy_or_none(loss),
            STORY_LOSS: cpu_deep_copy_or_none(story_loss),
            RELATION_LOSS: cpu_deep_copy_or_none(relation_loss),
            OBJECT_LOSS: cpu_deep_copy_or_none(object_loss),
        }
        return return_dict

    trainer, evaluator = Engine(train_step), Engine(valid_step)
    [ProgressBar().attach(_e) for _e in (trainer, evaluator)]

    # loss and average of trainer
    trainer_matrix = {
        LOSS: Average(lambda x: x[LOSS]),
        STORY_LOSS: Average(lambda x: x[STORY_LOSS]),
        RELATION_LOSS: Average(lambda x: x[RELATION_LOSS]),
        OBJECT_LOSS: Average(lambda x: x[OBJECT_LOSS]),
        STORY_ACCURACY: Accuracy(lambda x: (x[STORY_PRED], x[STORY_ANS])),
        RELATION_ACCURACY: Accuracy(lambda x: (x[RELATION_PRED], x[RELATION_ANS])),
        ENTITY_ACCURACY: Accuracy(lambda x: (x[ENTITY_PRED], x[OBJECT_ANS]))
    }
    logger.debug(f"----- add trainer matrix start -----")
    for key, value in trainer_matrix.items():
        value.attach(trainer, key)
        logger.debug(f"add trainer matrix: {key}")
    logger.debug(f"----- add trainer matrix end -----")
    # loss and average of evaluator
    if args.train_valid_test:
        valid_matrix = {
            LOSS: Average(lambda x: x[LOSS]),
            STORY_LOSS: Average(lambda x: x[STORY_LOSS]),
            RELATION_LOSS: Average(lambda x: x[RELATION_LOSS]),
            OBJECT_LOSS: Average(lambda x: x[OBJECT_LOSS]),
            STORY_ACCURACY: Accuracy(lambda x: (x[STORY_PRED], x[STORY_ANS])),
            RELATION_ACCURACY: Accuracy(lambda x: (x[RELATION_PRED], x[RELATION_ANS])),
            ENTITY_ACCURACY: Accuracy(lambda x: (x[ENTITY_PRED], x[OBJECT_ANS]))
        }
        logger.debug(f"----- add evaluator matrix start -----")
        for key, value in valid_matrix.items():
            value.attach(evaluator, key)
            logger.debug(f"add evaluator matrix: {key}")
        logger.debug(f"----- add evaluator matrix end -----")

    @trainer.on(Events.EPOCH_STARTED)
    def start_epoch_func(engine: Engine):
        epoch = engine.state.epoch
        logger.debug("----- epoch: {:>5} start -----".format(epoch))
        train.dataset.shuffle_per_1scene()

    @trainer.on(Events.EPOCH_COMPLETED)
    def end_epoch_func(engine: Engine):
        epoch = engine.state.epoch
        metrics = engine.state.metrics
        for _name in METRIC_NAMES:
            _value = metrics[_name]
            logger.debug(f"----- train metrics[{_name}]={_value} -----")
            if summary_writer is not None:
                summary_writer.add_scalar(f"{PRE_TRAIN}/{_name}", _value, global_step=epoch)
        if summary_writer is not None and hasattr(model, 'weight_head'):
            summary_writer.add_scalar(f"{PRE_TRAIN}/model_weight/story", model.weight_head.data, global_step=epoch)
            summary_writer.add_scalar(
                f"{PRE_TRAIN}/model_weight/relation", model.weight_relation.data, global_step=epoch)
            summary_writer.add_scalar(f"{PRE_TRAIN}/model_weight/entity", model.weight_tail.data, global_step=epoch)

    @trainer.on(Events.EPOCH_COMPLETED(every=args.valid_interval))
    def valid_func(engine: Engine):
        epoch = engine.state.epoch
        if args.train_valid_test:
            logger.info(f"----- valid start ({epoch=}) -----")
            evaluator.run(valid)
            metrics = evaluator.state.metrics
            for _name in METRIC_NAMES:
                _value = metrics[_name]
                logger.debug(f"----- valid metrics[{_name}]={_value} -----")
                if summary_writer is not None:
                    summary_writer.add_scalar(f"pre_valid/{_name}", _value, global_step=epoch)
            logger.info("----- valid end -----")

    total_timer = Timer(average=False)

    @trainer.on(Events.STARTED)
    def start_train(engine: Engine):
        total_timer.reset()
        logger.info("pre training start. epoch length = {}".format(engine.state.max_epochs))

    @trainer.on(Events.COMPLETED)
    def complete_train(engine: Engine):
        epoch = engine.state.epoch
        time_str = elapsed_time_str(total_timer.value())
        logger.info("pre training complete. finish epoch: {:>5}, time: {:>7}".format(epoch, time_str))

    @trainer.on(Events.EPOCH_COMPLETED)
    def print_time_per_epoch(engine: Engine):
        epoch = engine.state.epoch
        logger.info(
            "----- epoch: {:>5} complete. time: {:>8.2f}. total time: {:>7} -----".format(
                epoch, engine.state.times['EPOCH_COMPLETED'], elapsed_time_str(total_timer.value()))
        )

    @trainer.on(Events.ITERATION_COMPLETED(every=100))
    def print_info_per_some_iter(engine: Engine):
        output = engine.state.output
        epoch = engine.state.epoch
        logger.debug("----- epoch: {:>5} iter {:>6} complete. total time: {:>7} -----".format(
            epoch, engine.state.iteration, elapsed_time_str(total_timer.value())))
        logger.debug(f"loss={output[LOSS].item()}")

    if args.only_load_trainer_evaluator:
        logger.info("load trainer and evaluator. then end")
        return model, {TRAINER: trainer, EVALUATOR: evaluator}

    # about checkpoint
    to_save = {MODEL: model, OPTIMIZER: opt, TRAINER: trainer}
    good_checkpoint = Checkpoint(
        to_save, DiskSaver(MOST_GOOD_CHECKPOINT_PATH.format(checkpoint_dir), require_empty=False),
        global_step_transform=global_step_from_engine(trainer), include_self=True,
        score_name=LOSS, score_function=Checkpoint.get_default_score_fn(LOSS, -1.0))

    to_save = to_save | {CHECKPOINTER_GOOD_LOSS: good_checkpoint}
    last_checkpoint = Checkpoint(
        to_save, DiskSaver(LATEST_CHECKPOINT_PATH.format(checkpoint_dir), require_empty=False),
        include_self=True,
    )

    if args.resume_from_checkpoint and args.resume_from_last_point:
        raise ValueError("resume-from-checkpoint or resume-from-last-point can be 'True', not both.")
        pass
    elif args.resume_from_checkpoint:
        load_path = args.resume_checkpoint_path
        if load_path is None: raise "checkpoint_path must not None."
        to_load = {MODEL: model, OPTIMIZER: opt, TRAINER: trainer}
        logger.info(f"----- resume from path: {load_path}")
        Checkpoint.load_objects(to_load=to_load, checkpoint=load_path)
        good_checkpoint.save_handler = DiskSaver(MOST_GOOD_CHECKPOINT_PATH.format(checkpoint_dir), require_empty=True)
        last_checkpoint.save_handler = DiskSaver(LATEST_CHECKPOINT_PATH.format(checkpoint_dir), require_empty=True)
    elif args.resume_from_last_point:
        load_path = args.resume_checkpoint_path
        if load_path is None:
            raise ValueError("--checkpoint-path must not None.")
            pass
        to_load = {MODEL: model, OPTIMIZER: opt, TRAINER: trainer,
                   CHECKPOINTER_GOOD_LOSS: good_checkpoint, CHECKPOINTER: last_checkpoint}
        logger.info(f"----- resume from last. last_path: {load_path}")
        checkpoint = torch.load(load_path)
        to_load = {key: value for key, value in to_load.items() if key in checkpoint.keys()}
        Checkpoint.load_objects(to_load=to_load, checkpoint=checkpoint)
        logger.info(f"----- load objects keys: {to_load.keys()}")
        good_checkpoint.save_handler = DiskSaver(MOST_GOOD_CHECKPOINT_PATH.format(checkpoint_dir), require_empty=False)
        last_checkpoint.save_handler = DiskSaver(LATEST_CHECKPOINT_PATH.format(checkpoint_dir), require_empty=False)
    else:
        good_checkpoint.save_handler = DiskSaver(MOST_GOOD_CHECKPOINT_PATH.format(checkpoint_dir), require_empty=True)
        last_checkpoint.save_handler = DiskSaver(LATEST_CHECKPOINT_PATH.format(checkpoint_dir), require_empty=True)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, last_checkpoint)
    evaluator.add_event_handler(
        Events.COMPLETED, force_cpu_decorator(model, device, non_blocking=non_blocking)(good_checkpoint))

    if max_epoch > trainer.state.epoch:
        valid_func(trainer)  # first valid
        trainer.run(train, max_epochs=max_epoch)
    return model, {TRAINER: trainer, EVALUATOR: evaluator,
                   CHECKPOINTER_GOOD_LOSS: good_checkpoint, CHECKPOINTER_LAST: last_checkpoint}


def get_all_tokens(args: Namespace):
    """
    Args:
        args(Namespace):

    Returns:
        tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]:
            (
                (pad_token_e, pad_token_r), (cls_token_e, cls_token_r), (mask_token_e, mask_token_r),
                (sep_token_e, sep_token_r), (bos_token_e, bos_token_r)
            )

    """
    pad_token_e, pad_token_r = args.padding_token_e, args.padding_token_r
    cls_token_e, cls_token_r = args.cls_token_e, args.cls_token_r
    mask_token_e, mask_token_r = args.mask_token_e, args.mask_token_r
    sep_token_e, sep_token_r = args.sep_token_e, args.sep_token_r
    bos_token_e, bos_token_r = args.bos_token_e, args.bos_token_r
    return (
        (pad_token_e, pad_token_r), (cls_token_e, cls_token_r), (mask_token_e, mask_token_r),
        (sep_token_e, sep_token_r), (bos_token_e, bos_token_r)
    )


def make_get_data_helper(args: Namespace, *, logger: Logger):
    """

    Args:
        args(Namespace):
        logger(Logger):

    Returns:

    """
    ((pad_token_e, pad_token_r), (cls_token_e, cls_token_r), (mask_token_e, mask_token_r),
     (sep_token_e, sep_token_r), (bos_token_e, bos_token_r)) = get_all_tokens(args)
    is_challenge090, is_challenge075 = args.use_for_challenge090, args.use_for_challenge075

    if is_challenge090 or is_challenge075:
        if not args.only_train: raise ValueError("If use for challenge, --only-train must True")
        if args.use_title is None: raise ValueError("--use-title must not None.")

    train_file = TITLE2FILE090[args.use_title] if is_challenge090 \
        else TITLE2FILE075[args.use_title] if is_challenge075 else SRO_ALL_TRAIN_FILE

    entity_special_dicts = {
        pad_token_e: DefaultTokens.PAD_E, cls_token_e: DefaultTokens.CLS_E, mask_token_e: DefaultTokens.MASK_E,
        sep_token_e: DefaultTokens.SEP_E, bos_token_e: DefaultTokens.BOS_E
    }
    relation_special_dicts = {
        pad_token_r: DefaultTokens.PAD_R, cls_token_r: DefaultTokens.CLS_R, mask_token_r: DefaultTokens.MASK_R,
        sep_token_r: DefaultTokens.SEP_R, bos_token_r: DefaultTokens.BOS_R
    }
    data_helper = MyDataHelper(SRO_ALL_INFO_FILE, train_file, None, None, logger=logger,
                               entity_special_dicts=entity_special_dicts, relation_special_dicts=relation_special_dicts)
    data_helper.show(logger)
    return data_helper


def make_get_datasets(args: Namespace, *, data_helper: MyDataHelper, logger: Logger):
    """
    
    Args:
        args: 
        data_helper: 
        logger: 

    Returns:

    """
    # get from args
    ((pad_token_e, pad_token_r), _, _, (sep_token_e, sep_token_r), (bos_token_e, bos_token_r)) = get_all_tokens(args)
    entity_special_num = args.entity_special_num
    max_len = args.max_len
    train_valid_test, only_train = args.train_valid_test, args.only_train
    # get from data_helper
    entities, relations = data_helper.processed_entities, data_helper.processed_relations
    triple = data_helper.processed_train_triple
    # Check the number of data before including special tokens, as you will need them when creating the validation data.
    len_of_default_triple = len(triple)
    # add bos token in triples
    triple = add_bos(triple, bos_token_e, bos_token_r, bos_token_e)
    dataset_train, dataset_valid, dataset_test = None, None, None
    if train_valid_test:
        # These words cannot be left out in the search for the criminal.
        kill_entity, notKill_entity, beKilled_entity = [entities.index(_entity) for _entity in ABOUT_KILL_WORDS]
        # Filters to detect words essential for training.
        cannot_valid_filter = (triple[:, 0] < entity_special_num) | (triple[:, 2] == kill_entity) | \
                              (triple[:, 2] == notKill_entity) | (triple[:, 2] == beKilled_entity)
        cannot_valid_filter = cast(np.ndarray, cannot_valid_filter)
        # Filters to detect words essential for training.
        prob = (~cannot_valid_filter).astype(float)
        valid_test_indices = np.sort(
            np.random.choice(
                np.arange(len(triple)), size=np.floor(0.1 * len_of_default_triple).astype(int) * 2,
                p=(prob / np.sum(prob)), replace=False
            )
        )

        valid_indices, test_indices = valid_test_indices[0::2], valid_test_indices[1::2]
        assert len(valid_indices) == len(test_indices)

        valid_test_filter = np.zeros(len(triple), dtype=bool)
        test_filter = np.zeros(len(triple), dtype=bool)
        valid_filter = np.zeros(len(triple), dtype=bool)
        [np.put(_filter, _indices, True) for _filter, _indices in [
            (valid_test_filter, valid_test_indices), (test_filter, test_indices), (valid_filter, valid_indices)
        ]]
        assert np.array_equal(valid_test_filter, (test_filter | valid_filter))
        assert np.count_nonzero(test_filter) + np.count_nonzero(valid_filter) == np.count_nonzero(valid_test_filter)

        triple_train, triple_valid, triple_test = triple[~valid_test_filter], triple[~test_filter], triple
        valid_filter = valid_filter[~test_filter]
        assert np.count_nonzero(test_filter) + np.count_nonzero(valid_filter) == np.count_nonzero(valid_test_filter)

        # region debug area
        logger.debug("----- show all triple(no remove) -----")
        for i in range(30):
            logger.debug(f"example: {entities[triple[i][0]]}, {relations[triple[i][1]]}, {entities[triple[i][2]]}")
        logger.debug("----- show train triple -----")
        for i in range(30):
            logger.debug(
                f"example: "
                f"{entities[triple_train[i][0]]}, {relations[triple_train[i][1]]}, {entities[triple_train[i][2]]}")
        logger.debug("----- show example -----")
        # endregion

        # region debug area2
        entities_label, relations_label = data_helper.processed_entities_label, data_helper.processed_relations_label
        logger.debug("----- show all triple(no remove) label-----")
        for i in range(30):
            logger.debug(f"example: "
                         f"{entities_label[triple[i][0]]}, {relations_label[triple[i][1]]}, "
                         f"{entities_label[triple[i][2]]}")
        logger.debug("----- show example -----")
        # endregion

        dataset_train = StoryTriple(triple_train, np.where(triple_train[:, 0] == bos_token_e)[0], max_len,
                                    pad_token_e, pad_token_r, pad_token_e,
                                    sep_token_e, sep_token_r, sep_token_e)
        dataset_valid = StoryTripleForValid(triple_valid,
                                            np.where(triple_valid[:, 0] == bos_token_e)[0], valid_filter, max_len,
                                            pad_token_e, pad_token_r, pad_token_e,
                                            sep_token_e, sep_token_r, sep_token_e)
        dataset_test = StoryTripleForValid(triple_test,
                                           np.where(triple_test[:, 0] == bos_token_e)[0], test_filter, max_len,
                                           pad_token_e, pad_token_r, pad_token_e,
                                           sep_token_e, sep_token_r, sep_token_e)
    elif only_train:
        dataset_train = StoryTriple(triple, np.where(triple[:, 0] == bos_token_e)[0], max_len,
                                    pad_token_e, pad_token_r, pad_token_e,
                                    sep_token_e, sep_token_r, sep_token_e)
    else:
        raise ValueError("Either --train-valid-test or --only-train is required.")
        pass

    return dataset_train, dataset_valid, dataset_test


def make_get_model(args: Namespace, *, data_helper: MyDataHelper, logger: Logger):
    """

    Args:
        args:
        data_helper:
        logger:

    Returns:

    """
    # get from args

    all_tokens = [_t for _t in chain.from_iterable(get_all_tokens(args))]
    # get from data_helper
    num_entities, num_relations = len(data_helper.processed_entities), len(data_helper.processed_relations)
    logger.debug("----- make_model start -----")
    if 'model_version' not in args or args.model_version is None:
        raise ValueError("")
        pass
    version_ = args.model_version

    # noinspection PyTypeChecker
    Model_: KgStoryTransformer = (
        None if version_ not in ModelVersion.ALL_LIST()
        else KgStoryTransformer01 if version_ == ModelVersion.V01
        else KgStoryTransformer02 if version_ == ModelVersion.V02
        else KgStoryTransformer03 if version_ == ModelVersion.V03
        else KgStoryTransformer03preInit if version_ == ModelVersion.V03a
        else KgStoryTransformer00
    )

    if Model_ is None: raise f"model-version '{version_}' is not defined."

    model = Model_(args, num_entities, num_relations, special_tokens=SpecialTokens(*all_tokens))
    model.assert_check()
    model.init(args, data_helper=data_helper)
    logger.info(model)

    return model


def make_get_dataloader(args: Namespace, *, datasets: tuple[Dataset, Dataset, Dataset], logger: Logger):

    batch_size = args.batch_size
    dataset_train, dataset_valid, dataset_test = datasets
    dataloader_train = DataLoader(
        dataset_train, shuffle=True, batch_size=batch_size, num_workers=2, pin_memory=True)
    dataloader_valid = None if dataset_valid is None else DataLoader(
        dataset_valid, shuffle=False, batch_size=batch_size * 2, num_workers=2, pin_memory=True)
    dataloader_test = None if dataset_test is None else DataLoader(
        dataset_test, shuffle=False, batch_size=batch_size * 2, num_workers=2, pin_memory=True)
    data_loaders = MyDataLoaderHelper(datasets, dataloader_train, None, dataloader_valid, dataloader_test)
    logger.debug(f"{dataloader_train=}, {dataloader_valid=}, {dataloader_test=}")
    return data_loaders


def do_train_test_ect(args: Namespace, *, data_helper, data_loaders, model, logger: Logger):
    """

    Args:
        args(Namespace):
        data_helper(MyDataHelper):
        data_loaders(MyDataLoaderHelper):
        model(KgStoryTransformer01):
        logger(Logger):

    Returns:

    """
    # Now we are ready to start except for the hyper parameters.
    summary_writer = SummaryWriter(log_dir=args.tensorboard_dir) if args.tensorboard_dir is not None else None
    train_items = {TRAINER: None, EVALUATOR: None, CHECKPOINTER_LAST: None, CHECKPOINTER_GOOD_LOSS: None}

    # default mode
    if args.pre_train:
        # setting hyper parameter
        hyper_param = (args.lr, args.lr_story or args.lr, args.lr_relation or args.lr, args.lr_entity or args.lr,
                       args.loss_weight_story, args.loss_weight_relation, args.loss_weight_story)
        # setting path
        model_path = args.model_path
        if model_path is None:
            raise ValueError("model path must not None")
        # training.
        model, info_dict = pre_training(
            args, hyper_param, data_helper, data_loaders, model, summary_writer=summary_writer, logger=logger)
        # check the output of the training.
        good_checkpoint: Checkpoint = info_dict[CHECKPOINTER_GOOD_LOSS]
        last_checkpoint: Checkpoint = info_dict[CHECKPOINTER_LAST]
        logger.info(f"good model path: {good_checkpoint.last_checkpoint}")
        logger.info(f"last model path: {last_checkpoint.last_checkpoint}")
        checkpoint_ = last_checkpoint.last_checkpoint if args.only_train else good_checkpoint.last_checkpoint
        Checkpoint.load_objects(to_load={MODEL: model}, checkpoint=checkpoint_)
        # re-save as cpu model
        save_model(model, args.model_path, device=args.device)
        logger.info(f"save model path: {args.model_path}")
        # update training item for check the output.
        train_items.update(info_dict)
    # if checking the trained items, use this mode.
    elif args.only_load_trainer_evaluator:
        hyper_param = (0., 0., 0., 0., 1., 1., 1.)
        model, info_dict = pre_training(
            args, hyper_param, data_helper, data_loaders, model, summary_writer=summary_writer, logger=logger, )
        train_items.update(info_dict)

    return train_items


def main_function(args: Namespace, *, logger: Logger):
    # load raw data and make datahelper. Support for special-tokens by datahelper.
    logger.info('----- make datahelper start. -----')
    data_helper = make_get_data_helper(args, logger=logger)
    logger.info('----- make datahelper complete. -----')
    # make dataset.
    logger.info('----- make datasets start. -----')
    datasets = make_get_datasets(args, data_helper=data_helper, logger=logger)
    logger.info('----- make datasets complete. -----')
    # make dataloader.
    logger.info('----- make dataloader start. -----')
    data_loaders = make_get_dataloader(args, datasets=datasets, logger=logger)
    logger.info('----- make dataloader complete. -----')
    # make model
    logger.info('----- make model start -----')
    model = make_get_model(args, data_helper=data_helper, logger=logger)
    logger.info('----- make model complete. -----')
    # train test ect
    logger.info('----- do train start -----')
    train_items = do_train_test_ect(
        args, data_helper=data_helper, data_loaders=data_loaders, model=model, logger=logger)
    logger.info('----- do train complete -----')
    # return some value
    return {
        MODEL: model, DATA_HELPER: data_helper, DATASETS: datasets, DATA_LOADERS: data_loaders, TRAIN_ITEMS: train_items
    }


def main(args=None):
    from utils.setup import setup, save_param
    torch_fix_seed(seed=SEED)
    args, logger, device = setup(setup_parser, PROJECT_DIR, parser_args=args)
    if args.train_anyway:
        logger.warning("This process do not have reproducible.")
        torch.backends.cudnn.benchmark = True
    version_check(torch, np, pd, h5py, optuna, logger=logger)
    try:
        args.project_dir = PROJECT_DIR
        args.logger = logger
        args.device = device
        logger.debug(vars(args))
        logger.info(f"process id = {args.pid}")
        save_param(args)
        main_function(args, logger=logger)
    finally:
        save_param(args)
        pass


if __name__ == '__main__':
    main()
    pass
