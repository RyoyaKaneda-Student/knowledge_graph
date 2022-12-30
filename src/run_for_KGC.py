#!/usr/bin/python
# -*- coding: utf-8 -*-
from argparse import Namespace
# noinspection PyUnresolvedReferences
from collections import namedtuple
# ========== python ==========
from logging import Logger
from pathlib import Path
import gc
# noinspection PyUnresolvedReferences
from typing import List, Dict, Tuple, Optional, Union, Callable, Final, Literal, get_args, cast

# Machine learning
import h5py
import numpy as np
import optuna
import pandas as pd
# torch
import torch
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import Timer, Checkpoint, global_step_from_engine, DiskSaver
from ignite.metrics import Average, Accuracy
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

# Made by myself
from models.KGModel.kg_story_transformer import (
    KgStoryTransformer01, KgStoryTransformer02, add_bos, SpecialTokens01 as SpecialTokens, KgStoryTransformer03)
from models.datasets.data_helper import MyDataHelper
from models.datasets.datasets import StoryTriple, StoryTripleForValid
from utils.torch import save_model, torch_fix_seed, DeviceName, force_cpu_decorator
from utils.utils import force_gc_after_function, version_check, elapsed_time_str

PROJECT_DIR = Path(__file__).resolve().parents[1]

CPU: Final = 'cpu'
TRAIN: Final = 'train'
TEST: Final = 'test'
MRR: Final = 'mrr'
HIT_: Final = 'hit_'
STUDY: Final = 'study'
MODEL: Final = 'model'
OPTIMIZER: Final = 'optimizer'

TRAINER: Final = 'trainer'
EVALUATOR: Final = 'evaluator'
CHECKPOINTER: Final = 'checkpointer'
CHECKPOINTER_GOOD_LOSS: Final = 'checkpointer_good_loss'
CHECKPOINTER_LAST: Final = 'checkpointer_last'
DATA_HELPER: Final = 'data_helper'

PAD_E: Final = '<pad_e>'
CLS_E: Final = '<cls_e>'
MASK_E: Final = '<mask_e>'
SEP_E: Final = '<sep_e>'
BOS_E: Final = '<bos_e>'
PAD_R: Final = '<pad_r>'
CLS_R: Final = '<cls_r>'
MASK_R: Final = '<mask_r>'
SEP_R: Final = '<sep_r>'
BOS_R: Final = '<bos_r>'

ALL_TAIL = 'all_tail'
TRIPLE = 'triple'
DATASETS = 'datasets'

LOSS: Final = 'loss'
STORY_RELATION_ENTITY = ('story', 'relation', 'entity')
STORY_LOSS: Final = 'story_loss'
RELATION_LOSS: Final = 'relation_loss'
OBJECT_LOSS: Final = 'entity_loss'
LOSS_NAME3 = (STORY_LOSS, RELATION_LOSS, OBJECT_LOSS)

STORY_PRED: Final = 'story_pred'
RELATION_PRED: Final = 'relation_pred'
ENTITY_PRED: Final = 'entity_pred'
PRED_NAME3 = (STORY_PRED, RELATION_PRED, ENTITY_PRED)

STORY_ANS: Final = 'story_ans'
RELATION_ANS: Final = 'relation_ans'
OBJECT_ANS: Final = 'object_ans'
ANS_NAME3 = (STORY_ANS, RELATION_ANS, OBJECT_ANS)

STORY_ACCURACY: Final = 'story_accuracy'
RELATION_ACCURACY: Final = 'relation_accuracy'
ENTITY_ACCURACY: Final = 'entity_accuracy'
ACCURACY_NAME3 = (STORY_ACCURACY, RELATION_ACCURACY, ENTITY_ACCURACY)

METRIC_NAMES = [LOSS, *LOSS_NAME3, *ACCURACY_NAME3]

ALL_TITLE_LIST = [
    'ACaseOfIdentity', 'AbbeyGrange', 'CrookedMan', 'DancingMen',
    'DevilsFoot', 'ResidentPatient', 'SilverBlaze', 'SpeckledBand'
]

ABOUT_KILL_WORDS: Final = ['word.predicate:kill', 'word.predicate:notKill', 'word.predicate:beKilled']

SRO_ALL_INFO_FILE = "data/processed/KGCdata/All/SRO/info.hdf5"
SRO_ALL_TRAIN_FILE = "data/processed/KGCdata/All/SRO/train.hdf5"

SRO_AbbeyGrange075_TRAIN_FILE = "data/processed/KGCdata/All/SRO/train_AbbeyGrange_075.hdf5"
SRO_ACaseOfIdentity075_TRAIN_FILE = "data/processed/KGCdata/All/SRO/train_ACaseOfIdentity_075.hdf5"
SRO_CrookedMan075_TRAIN_FILE = "data/processed/KGCdata/All/SRO/train_CrookedMan_075.hdf5"
SRO_DancingMen075_TRAIN_FILE = "data/processed/KGCdata/All/SRO/train_DancingMen_075.hdf5"
SRO_DevilsFoot075_TRAIN_FILE = "data/processed/KGCdata/All/SRO/train_DevilsFoot_075.hdf5"
SRO_ResidentPatient075_TRAIN_FILE = "data/processed/KGCdata/All/SRO/train_ResidentPatient_075.hdf5"
SRO_SilverBlaze075_TRAIN_FILE = "data/processed/KGCdata/All/SRO/train_SilverBlaze_075.hdf5"
SRO_SpeckledBand075_TRAIN_FILE = "data/processed/KGCdata/All/SRO/train_SpeckledBand_075.hdf5"

TITLE2FILE075 = {
    'ACaseOfIdentity': SRO_AbbeyGrange075_TRAIN_FILE,
    'AbbeyGrange': SRO_ACaseOfIdentity075_TRAIN_FILE,
    'CrookedMan': SRO_CrookedMan075_TRAIN_FILE,
    'DancingMen': SRO_DancingMen075_TRAIN_FILE,
    'DevilsFoot': SRO_DevilsFoot075_TRAIN_FILE,
    'ResidentPatient': SRO_ResidentPatient075_TRAIN_FILE,
    'SilverBlaze': SRO_SilverBlaze075_TRAIN_FILE,
    'SpeckledBand': SRO_SpeckledBand075_TRAIN_FILE
}

CHECKPOINT_DIR: Final = 'saved_models/.tmp/check-point.{}'
MOST_GOOD_CHECKPOINT_PATH: Final = '{}/most_good/'
LATEST_CHECKPOINT_PATH: Final = '{}/most_good/'

SEED: Final = 42


def setup_parser(args: Namespace = None) -> Namespace:
    """
    Args:
        args:

    Returns:

    """
    import argparse  # 1. argparseをインポート
    parser = argparse.ArgumentParser(description='データの初期化')
    paa = parser.add_argument
    parser.add_argument_group()
    paa('--notebook', help='if use notebook, use this argument.', action='store_true')
    paa('--train-anyway', help='It will not be reproducible, but it could be faster.', action='store_true')
    paa('--logfile', help='the path of saving log', type=str, default='log/test.log')
    paa('--param-file', help='the path of saving param', type=str, default='log/param.pkl')
    paa('--tensorboard-dir', help='tensorboard direction', type=str, default='log/tensorboard/')
    paa('--checkpoint-dir', help='tensorboard direction', type=str, default='log/checkpoint/')
    paa('--model-path', type=str, help='model path')
    paa('--model-version', type=str, help='model version.')
    paa('--resume-from-checkpoint', help='if use checkpoint, use this argument.', action='store_true')
    paa('--resume-from-last-point', help='if use checkpoint, use this argument.', action='store_true')
    paa('--only-load-trainer-evaluator', help='', action='store_true')
    paa('--resume-checkpoint-path', help='if use checkpoint, use this argument.', type=str)
    paa('--console-level', help='log level on console', type=str, default='debug', choices=['info', 'debug'])
    paa('--device-name', help=DeviceName.ALL_INFO, type=str, default=DeviceName.CPU, choices=DeviceName.ALL_LIST)
    paa('--pre-train', help="Put on if you are doing pre-training", action='store_true')
    paa('--train-valid-test', help='', action='store_true')
    paa('--only-train', help='', action='store_true')
    paa('--use-for-challenge100', help='', action='store_true')
    paa('--use-for-challenge075', help='', action='store_true')
    paa('--use-title', help=' or '.join(ALL_TITLE_LIST), type=str, choices=ALL_TITLE_LIST)
    # optuna setting
    paa('--do-optuna', help="do optuna", action='store_true')
    paa('--optuna-file', help='optuna file', type=str)
    paa('--study-name', help='optuna study-name', type=str)
    paa('--n-trials', help='optuna n-trials', type=int)
    # special num count
    paa('--story-special-num', help='story special num', type=int, default=5)
    paa('--relation-special-num', help='relation special num', type=int, default=5)
    paa('--entity-special-num', help='entity special num', type=int, default=5)
    # e special
    paa('--padding-token-e', help='padding', type=int, default=0)
    paa('--cls-token-e', help='cls', type=int, default=1)
    paa('--mask-token-e', help='mask', type=int, default=2)
    paa('--sep-token-e', help='sep', type=int, default=3)
    paa('--bos-token-e', help='bos', type=int, default=4)
    # r special
    paa('--padding-token-r', help='padding', type=int, default=0)
    paa('--cls-token-r', help='cls', type=int, default=1)
    paa('--mask-token-r', help='mask', type=int, default=2)
    paa('--sep-token-r', help='sep', type=int, default=3)
    paa('--bos-token-r', help='bos', type=int, default=4)
    # story
    paa('--padding-token-s', help='padding', type=int, default=0)
    paa('--cls-token-s', help='cls', type=int, default=1)
    paa('--mask-token-s', help='mask', type=int, default=2)
    paa('--sep-token-s', help='sep', type=int, default=3)
    paa('--bos-token-s', help='bos', type=int, default=4)
    # model
    paa('--embedding-dim', type=int, default=128, help='The embedding dimension. Default: 128')
    paa('--entity-embedding-dim', type=int, default=128, help='The embedding dimension. Default: 128')
    paa('--relation-embedding-dim', type=int, default=128, help='The embedding dimension. Default: 128')
    paa('--separate-head-and-tail', action='store_true', default=False,
        help='If True, it head Embedding and tail Embedding are different.')
    paa('--batch-size', help='batch size', type=int, default=4)
    paa('--max-len', help='max length of 1 batch. default: 256', type=int, default=256)
    paa('--mask-percent', help='default: 0.15', type=float, default=0.15)
    paa('--mask-mask-percent', help='default: 0.80', type=float, default=0.80)
    paa('--mask-random-percent', help='default: 0.10', type=float, default=0.10)
    paa('--mask-nomask-percent', help='default: 0.10', type=float, default=0.10)
    paa('--no-use-pe', help='to check pe(position encoding) power, we have to make no pe model', action='store_true')
    paa('--epoch', help='max epoch', type=int, default=2)
    # optimizer
    paa('--lr', type=float, default=0.003, help='learning rate (default: 0.003)')
    paa('--lr-story', type=float, help='learning rate (default: same as --lr)')
    paa('--lr-relation', type=float, help='learning rate (default: same as --lr)')
    paa('--lr-entity', type=float, help='learning rate (default: same as --lr)')
    paa('--valid-interval', type=int, default=1, help='valid-interval', )
    paa('--loss-weight-story', type=float, default=1., help='loss-weight-story')
    paa('--loss-weight-relation', type=float, default=1., help='loss-weight-relation')
    paa('--loss-weight-entity', type=float, default=1., help='loss-weight-entity')
    # transformer
    paa('--nhead', type=int, default=4, help='nhead. Default: 4.')
    paa('--num-layers', type=int, default=4, help='num layers. Default: 4.')
    paa('--dim-feedforward', type=int, default=1028, help='dim of feedforward. Default: 1028.')
    paa('--transformer-drop', type=float, default=0.1, help='transformer-drop. Default: 0.1.')
    paa('--position-encoder-drop', type=float, default=0.1, help='position-encoder-drop. Default: 0.1.')

    args = parser.parse_args(args=args)
    return args


def pre_training(
        args: Namespace, data_helper: MyDataHelper, model: KgStoryTransformer01,
        lr, lr_story, lr_relation, lr_entity,
        loss_weight_story, loss_weight_relation, loss_weight_entity,
        summary_writer: SummaryWriter, *, logger: Logger, ):
    """

    Args:
        lr(float):
        lr_story(float):
        lr_relation(float):
        lr_entity(float):
        loss_weight_entity(float):
        loss_weight_relation(float):
        loss_weight_story(float):
        args(Namespace):
        data_helper(MyDataHelper):
        model(KgStoryTransformer01):
        summary_writer(SummaryWriter):
        logger(Logger):

    Returns:

    """
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
        {'params': _module.parameters(), 'lr': lr} for _name, _module in modules.items()
    ] + [
        {'params': model.head_maskdlm.parameters(), 'lr': lr_story},
        {'params': model.relation_maskdlm.parameters(), 'lr': lr_relation},
        {'params': model.tail_maskdlm.parameters(), 'lr': lr_entity},
    ]

    opt = torch.optim.Adam(optim_list)
    loss_fn_entity = torch.nn.CrossEntropyLoss(weight=torch.ones(entity_num).to(device))
    loss_fn_relation = torch.nn.CrossEntropyLoss(weight=torch.ones(relation_num).to(device))
    checkpoint_dir = args.checkpoint_dir
    # checkpoint_dir = CHECKPOINT_DIR.format(line_up_key_value(pid=args.pid, uid=uid))
    train = data_helper.train_dataloader
    valid = data_helper.valid_dataloader if args.train_valid_test else None
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
                summary_writer.add_scalar(f"pre_train/{_name}", _value, global_step=epoch)
        if summary_writer is not None and hasattr(model, 'weight_head'):
            summary_writer.add_scalar(f"pre_train/model_weight/story", model.weight_head.data, global_step=epoch)
            summary_writer.add_scalar(f"pre_train/model_weight/relation", model.weight_relation.data, global_step=epoch)
            summary_writer.add_scalar(f"pre_train/model_weight/entity", model.weight_tail.data, global_step=epoch)

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


def get_all_tokens(args):
    pad_token_e, pad_token_r = args.padding_token_e, args.padding_token_r
    cls_token_e, cls_token_r = args.cls_token_e, args.cls_token_r
    mask_token_e, mask_token_r = args.mask_token_e, args.mask_token_r
    sep_token_e, sep_token_r = args.sep_token_e, args.sep_token_r
    bos_token_e, bos_token_r = args.bos_token_e, args.bos_token_r
    return (
        (pad_token_e, pad_token_r), (cls_token_e, cls_token_r), (mask_token_e, mask_token_r),
        (sep_token_e, sep_token_r), (bos_token_e, bos_token_r)
    )


def make_data_helper(args, *, logger: Logger):
    """

    Args:
        args:
        logger:

    Returns:

    """
    entity_special_num, relation_special_num = args.entity_special_num, args.relation_special_num
    ((pad_token_e, pad_token_r), (cls_token_e, cls_token_r), (mask_token_e, mask_token_r),
     (sep_token_e, sep_token_r), (bos_token_e, bos_token_r)) = get_all_tokens(args)
    if args.use_for_challenge075:
        if not args.only_train: raise ValueError("If use for challenge, --only-train must True")
        if args.use_title is None: raise ValueError("--use-title must not None.")
        train_file = TITLE2FILE075[args.use_title]
    else:
        train_file = SRO_ALL_TRAIN_FILE
        pass
    data_helper = MyDataHelper(SRO_ALL_INFO_FILE, None, train_file, None, None, logger=logger,
                               entity_special_num=entity_special_num, relation_special_num=relation_special_num)
    data_helper.set_special_names(
        {pad_token_e: PAD_E, cls_token_e: CLS_E, mask_token_e: MASK_E, sep_token_e: SEP_E, bos_token_e: BOS_E},
        {pad_token_r: PAD_R, cls_token_r: CLS_R, mask_token_r: MASK_R, sep_token_r: SEP_R, bos_token_r: BOS_R},
    )
    return data_helper


def make_datasets(args, *, data_helper: MyDataHelper, logger: Logger):
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
    logger.debug("----- make_datasets start -----")
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


def make_model(args, *, data_helper: MyDataHelper, logger: Logger):
    """

    Args:
        args:
        data_helper:
        logger:

    Returns:

    """
    # get from args
    all_tokens = [_t for _tt in get_all_tokens(args) for _t in _tt]
    # get from data_helper
    num_entities, num_relations = len(data_helper.processed_entities), len(data_helper.processed_relations)
    logger.debug("----- make_model start -----")
    if 'model_version' not in args or args.model_version is None:
        raise ValueError("")
        pass
    elif args.model_version == '01':
        model = KgStoryTransformer01(args, num_entities, num_relations, special_tokens=SpecialTokens(*all_tokens))
        pass
    elif args.model_version == '02':
        model = KgStoryTransformer02(args, num_entities, num_relations, special_tokens=SpecialTokens(*all_tokens))
        pass
    elif args.model_version == '03':
        model = KgStoryTransformer03(args, num_entities, num_relations, special_tokens=SpecialTokens(*all_tokens))
        pass
    else:
        raise ValueError("aaa")
        pass

    model.assert_check()
    logger.info(model)

    return model


def make_set_dataloader(args, *, datasets: tuple[Dataset, Dataset, Dataset], data_helper: MyDataHelper, logger: Logger):
    logger.debug("----- make_set_dataloader -----")
    batch_size = args.batch_size
    dataset_train, dataset_valid, dataset_test = datasets
    dataloader_train = DataLoader(
        dataset_train, shuffle=True, batch_size=batch_size, num_workers=2, pin_memory=True)
    dataloader_valid = None if dataset_valid is None else DataLoader(
        dataset_valid, shuffle=False, batch_size=batch_size * 2, num_workers=2, pin_memory=True)
    dataloader_test = None if dataset_test is None else DataLoader(
        dataset_test, shuffle=False, batch_size=batch_size * 2, num_workers=2, pin_memory=True)
    data_helper.set_loaders(dataloader_train, None, dataloader_valid, dataloader_test)


def main_function(args: Namespace, *, logger: Logger):
    logger.info('----- make datahelper start. -----')
    data_helper = make_data_helper(args, logger=logger)
    logger.info('----- make datahelper complete. -----')
    datasets = make_datasets(args, data_helper=data_helper, logger=logger)

    logger.info('----- make and set dataloader start. -----')
    make_set_dataloader(args, datasets=datasets, data_helper=data_helper, logger=logger)
    logger.info('----- make and set dataloader complete. -----')

    logger.info('----- make model start -----')
    model = make_model(args, data_helper=data_helper, logger=logger)
    logger.info('----- make model complete. -----')

    summary_writer = SummaryWriter(log_dir=args.tensorboard_dir) if args.tensorboard_dir is not None else None
    train_items = {TRAINER: None, EVALUATOR: None, CHECKPOINTER_LAST: None, CHECKPOINTER_GOOD_LOSS: None}

    if args.pre_train:
        # setting hyper parameter
        lr = args.lr
        lr_story = args.lr_story or args.lr
        lr_relation = args.lr_relation or args.lr
        lr_entity = args.lr_entity or args.lr
        loss_weight_story = args.loss_weight_story
        loss_weight_relation = args.loss_weight_relation
        loss_weight_entity = args.loss_weight_story
        # setting path
        model_path = args.model_path
        assert model_path is not None

        model, info_dict = pre_training(
            args, data_helper=data_helper, model=model,
            lr=lr, lr_story=lr_story, lr_relation=lr_relation, lr_entity=lr_entity,
            loss_weight_story=loss_weight_story,
            loss_weight_relation=loss_weight_relation,
            loss_weight_entity=loss_weight_entity,
            summary_writer=summary_writer, logger=logger,
        )
        good_checkpoint: Checkpoint = info_dict[CHECKPOINTER_GOOD_LOSS]
        last_checkpoint: Checkpoint = info_dict[CHECKPOINTER_LAST]
        logger.info(f"goog model path: {good_checkpoint.last_checkpoint}")
        logger.info(f"last model path: {last_checkpoint.last_checkpoint}")
        if args.only_train:
            Checkpoint.load_objects(to_load={MODEL: model}, checkpoint=last_checkpoint.last_checkpoint)
        else:
            Checkpoint.load_objects(to_load={MODEL: model}, checkpoint=good_checkpoint.last_checkpoint)
        save_model(model, args.model_path, device=args.device)
        logger.info(f"save model path: {args.model_path}")
        train_items.update(info_dict)
    elif args.only_load_trainer_evaluator:
        model, info_dict = pre_training(
            args, data_helper=data_helper, model=model,
            lr=0., lr_story=0., lr_relation=0., lr_entity=0.,
            loss_weight_story=1.,
            loss_weight_relation=1.,
            loss_weight_entity=1.,
            summary_writer=summary_writer, logger=logger,
        )
        train_items.update(info_dict)

    return model, {DATA_HELPER: data_helper, DATASETS: datasets, 'train_items': train_items}


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
