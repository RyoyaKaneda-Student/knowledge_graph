#!/usr/bin/python
# -*- coding: utf-8 -*-
# noinspection PyUnresolvedReferences
from argparse import Namespace
# noinspection PyUnresolvedReferences
from collections import namedtuple
# ========== python ==========
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
from ignite.engine import Engine, Events
from ignite.handlers import Timer
from ignite.metrics import Average, Accuracy
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.KGModel.kg_story_transformer import KgStoryTransformer01, add_bos_eos
from models.datasets.data_helper import (
    MyDataHelper, )
from models.datasets.datasets import StoryTriple
from utils.setup import setup, save_param
from utils.str_process import line_up_key_value
from utils.torch import (
    load_model, save_model,
    torch_fix_seed,
    DeviceName
)
# Made by me
from utils.utils import force_gc_after_function, version_check, elapsed_time_str

PROJECT_DIR = Path(__file__).resolve().parents[1]

MODEL_TMP_PATH = 'saved_models/.tmp/check-point.{}.model'

CPU: Final = 'cpu'
TRAIN: Final = 'train'
TEST: Final = 'test'
MRR: Final = 'mrr'
HIT_: Final = 'hit_'
STUDY: Final = 'study'

ALL_TAIL = 'all_tail'
TRIPLE = 'triple'

LOSS: Final = 'loss'
STORY_RELATION_ENTITY = ('story', 'relation', 'entity')
STORY_LOSS: Final = 'story_loss'
RELATION_LOSS: Final = 'relation_loss'
ENTITY_LOSS: Final = 'entity_loss'
LOSS_NAME3 = (STORY_LOSS, RELATION_LOSS, ENTITY_LOSS)

STORY_PRED: Final = 'story_pred'
RELATION_PRED: Final = 'relation_pred'
ENTITY_PRED: Final = 'entity_pred'
PRED_NAME3 = (STORY_PRED, RELATION_PRED, ENTITY_PRED)

STORY_ANS: Final = 'story_ans'
RELATION_ANS: Final = 'relation_ans'
ENTITY_ANS: Final = 'entity_ans'
ANS_NAME3 = (STORY_ANS, RELATION_ANS, ENTITY_ANS)

STORY_ACCURACY: Final = 'story_accuracy'
RELATION_ACCURACY: Final = 'relation__accuracy'
ENTITY_ACCURACY: Final = 'entity_accuracy'
ACCURACY_NAME3 = (STORY_ACCURACY, RELATION_ACCURACY, ENTITY_ACCURACY)

METRIC_NAMES = [LOSS, *LOSS_NAME3, *ACCURACY_NAME3]


def setup_parser(args: Namespace = None) -> Namespace:
    """
    Args:
        args:

    Returns:

    """
    import argparse  # 1. argparseをインポート
    parser = argparse.ArgumentParser(description='データの初期化')
    paa = parser.add_argument
    paa('--notebook', help='if use notebook, use this argument.', action='store_true')
    paa('--logfile', help='the path of saving log', type=str, default='log/test.log')
    paa('--param-file', help='the path of saving param', type=str, default='log/param.pkl')
    paa('--tensorboard-dir', help='tensorboard direction', type=str, default='log/tensorboard/')
    paa('--console-level', help='log level on console', type=str, default='debug', choices=['info', 'debug'])
    paa('--device-name', help=DeviceName.ALL_INFO, type=str, default=DeviceName.CPU, choices=DeviceName.ALL_LIST)
    paa('--pre-train', help="", action='store_true')
    # optuna setting
    paa('--optuna-file', help='optuna file', type=str)
    paa('--study-name', help='optuna study-name', type=str)
    paa('--n-trials', help='optuna n-trials', type=int, default=20)
    #
    paa('--story-special-num', help='ストーリー', type=int, default=6)
    paa('--relation-special-num', help='リレーション', type=int, default=6)
    paa('--entity-special-num', help='エンティティ', type=int, default=6)
    # e special
    paa('--padding-token-e', help='padding', type=int, default=0)
    paa('--cls-token-e', help='cls', type=int, default=1)
    paa('--mask-token-e', help='mask', type=int, default=2)
    paa('--sep-token-e', help='sep', type=int, default=3)
    paa('--bos-token-e', help='bos', type=int, default=4)
    paa('--eos-token-e', help='eos', type=int, default=5)
    # r special
    paa('--padding-token-r', help='padding', type=int, default=0)
    paa('--cls-token-r', help='cls', type=int, default=1)
    paa('--mask-token-r', help='mask', type=int, default=2)
    paa('--sep-token-r', help='sep', type=int, default=3)
    paa('--bos-token-r', help='bos', type=int, default=4)
    paa('--eos-token-r', help='eos', type=int, default=5)
    # story
    paa('--padding-token-s', help='padding', type=int, default=0)
    paa('--cls-token-s', help='cls', type=int, default=1)
    paa('--mask-token-s', help='mask', type=int, default=2)
    paa('--sep-token-s', help='sep', type=int, default=3)
    paa('--bos-token-s', help='bos', type=int, default=4)
    paa('--eos-token-s', help='eos', type=int, default=5)
    # paa('--model', type=str, help=f"Choose from: {', '.join(name2model.keys())}")
    paa('--embedding-dim', type=int, default=128, help='The embedding dimension (1D). Default: 128')
    paa('--batch-size', help='batch size', type=int, default=4)
    paa('--max-len', help='max length of 1 batch. default: 256', type=int, default=256)
    paa('--mask-percent', help='default: 0.15', type=float, default=0.15)
    paa('--epoch', help='max epoch', type=int)

    paa('--model-path', type=str, help='model path')
    # optimizer
    paa('--lr', type=float, default=0.003, help='learning rate (default: 0.003)')
    paa('--label-smoothing', type=float, default=0.1, help='Label smoothing value to use. Default: 0.1')
    #
    paa('--transformer-drop', type=float, default=0.1, help='transformer-drop. Default: 0.1.')
    paa('--position-encoder-drop', type=float, default=0.1, help='position-encoder-drop. Default: 0.1.')
    paa('--nhead', type=int, default=4, help='nhead. Default: 4.')
    paa('--num-layers', type=int, default=8, help='num layers. Default: 8.')

    args = parser.parse_args(args=args)
    return args


def pre_training(
        args: Namespace, *, logger: Logger,
        data_helper: MyDataHelper,
        model: KgStoryTransformer01,
        lr: Union[torch.tensor, float],
        summary_writer: SummaryWriter,
        uid: Optional[str] = None
):
    device = args.device
    model.to(device)
    max_len = args.max_len

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    checkpoint_path = MODEL_TMP_PATH.format(line_up_key_value(pid=args.pid, uid=uid))
    train = data_helper.train_dataloader
    mask_percent = args.mask_percent
    max_epoch = args.epoch

    mask_token_e, mask_token_r = args.mask_token_e, args.mask_token_r

    @force_gc_after_function
    def train_step(_: Engine, batch) -> dict:
        triple = batch.to(device)
        # triple.shape = (batch, max_len, 3)
        opt.zero_grad()
        triple = triple.to(device)
        triple_ans = triple.detach()

        batch_size = triple.shape[0]
        mask_head = torch.where(torch.rand(batch_size, max_len) < mask_percent, True, False).to(device)
        mask_relation = torch.where(torch.rand(batch_size, max_len) < mask_percent, True, False).to(device)
        mask_tail = torch.where(torch.rand(batch_size, max_len) < mask_percent, True, False).to(device)
        triple[mask_head][:, 0] = mask_token_e
        triple[mask_relation][:, 1] = mask_token_r
        triple[mask_tail][:, 2] = mask_token_e
        _, story_pred, relation_pred, entity_pred = model.pre_train_forward(triple, mask_head, mask_relation, mask_tail)
        story_loss: torch.Tensor = loss_fn(story_pred, triple_ans[mask_head][:, 0])
        relation_loss: torch.Tensor = loss_fn(relation_pred, triple_ans[mask_relation][:, 1])
        entity_loss: torch.Tensor = loss_fn(entity_pred, triple_ans[mask_tail][:, 2])
        loss: torch.Tensor = story_loss + relation_loss + entity_loss
        loss.backward()
        opt.step()
        rev = {
            LOSS: loss,
            STORY_ANS: triple_ans[mask_head][:, 0],
            RELATION_ANS: triple_ans[mask_relation][:, 1],
            ENTITY_ANS: triple_ans[mask_tail][:, 2],
            STORY_PRED: story_pred, RELATION_PRED: relation_pred, ENTITY_PRED: entity_pred,
            STORY_LOSS: story_loss, RELATION_LOSS: relation_loss, ENTITY_LOSS: entity_loss,
        }
        return {key: value.detach() for key, value in rev.items()}

    trainer = Engine(train_step)

    # loss and average
    Average(output_transform=lambda _dict: _dict[LOSS]).attach(trainer, LOSS)
    Average(output_transform=lambda x: x[STORY_LOSS]).attach(trainer, STORY_LOSS)
    Average(output_transform=lambda x: x[RELATION_LOSS]).attach(trainer, RELATION_LOSS)
    Average(output_transform=lambda x: x[ENTITY_LOSS]).attach(trainer, ENTITY_LOSS)
    Accuracy(output_transform=lambda x: (x[STORY_PRED], x[STORY_ANS])).attach(trainer, STORY_ACCURACY)
    Accuracy(output_transform=lambda x: (x[RELATION_PRED], x[RELATION_ANS])).attach(trainer, RELATION_ACCURACY)
    Accuracy(output_transform=lambda x: (x[RELATION_PRED], x[RELATION_ANS])).attach(trainer, RELATION_ACCURACY)
    Accuracy(output_transform=lambda x: (x[ENTITY_PRED], x[ENTITY_ANS])).attach(trainer, ENTITY_ACCURACY)

    @trainer.on(Events.EPOCH_STARTED)
    def start_epoch_func(engine: Engine):
        epoch = engine.state.epoch
        logger.debug("----- epoch: {:>5} start -----".format(epoch))

    @trainer.on(Events.EPOCH_COMPLETED)
    def end_epoch_func(engine: Engine):
        epoch = engine.state.epoch
        save_model(model, checkpoint_path, device=device)
        metrics = engine.state.metrics
        for _name in METRIC_NAMES:
            _value = metrics[_name]
            logger.debug(f"-----metrics[{_name}]={_value}")
            if summary_writer is not None:
                summary_writer.add_scalar(f"pre_train/{_name}", _value, global_step=epoch)

    total_timer = Timer(average=False)

    @trainer.on(Events.STARTED)
    def start_train(engine: Engine):
        total_timer.reset()
        logger.info("training start. epoch length = {}".format(engine.state.max_epochs))

    @trainer.on(Events.COMPLETED)
    def complete_train(engine: Engine):
        epoch = engine.state.epoch
        time_str = elapsed_time_str(total_timer.value())
        logger.info("training complete. finish epoch: {:>5}, time: {:>7}".format(epoch, time_str))

    @trainer.on(Events.EPOCH_COMPLETED)
    def print_time_per_epoch(engine: Engine):
        epoch = engine.state.epoch
        time_str = elapsed_time_str(total_timer.value())
        print_text = "----- epoch: {:>5} complete. time: {:>8.2f}. total time: {:>7} -----".format(
            epoch, engine.state.times['EPOCH_COMPLETED'], time_str)

        logger.info(print_text)

    @trainer.on(Events.ITERATION_COMPLETED(every=100))
    def print_info_per_some_iter(engine: Engine):
        output = engine.state.output
        epoch = engine.state.epoch
        logger.debug("----- epoch: {:>5} iter {:>6} complete. total time: {:>7} -----".format(
            epoch, engine.state.iteration, elapsed_time_str(total_timer.value())))
        logger.debug(f"loss={output[LOSS].item()}")

    trainer.run(train, max_epochs=max_epoch)
    load_model(model, checkpoint_path, device=device, delete_file=True)
    return model, {'state': trainer.state}


def main_function(args: Namespace, *, logger: Logger):
    summary_writer = SummaryWriter(log_dir=args.tensorboard_dir)
    entity_special_num: int = args.entity_special_num
    relation_special_num: int = args.relation_special_num
    batch_size, lr = args.batch_size, args.lr
    max_len = args.max_len

    padding_token_e, padding_token_r = args.padding_token_e, args.padding_token_r
    sep_token_e, sep_token_r = args.sep_token_e, args.sep_token_r
    bos_token_e, bos_token_r = args.bos_token_e, args.bos_token_r
    eos_token_e, eos_token_r = args.eos_token_e, args.eos_token_r

    data_helper = MyDataHelper(
        "data/processed/KGCdata/All/SRO/info.hdf5",
        None,
        "data/processed/KGCdata/All/SRO/train.hdf5",
        None, None, logger=logger,
        entity_special_num=entity_special_num,
        relation_special_num=relation_special_num
    )
    data_helper.set_special_names(
        index2name_entity={0: '<pad_e>', 1: '<cls_e>', 2: '<mask_e>', 3: '<sep_e>', 4: '<bos_e>', 5: '<eos_e>'},
        index2name_relation={0: '<pad_r>', 1: '<cls_r>', 2: '<mask_r>', 3: '<sep_r>', 4: '<bos_r>', 5: '<eos_r>'},
    )

    entities = data_helper.processed_entities
    relations = data_helper.processed_relations

    num_entities, num_relations = len(data_helper.processed_entities), len(data_helper.processed_relations)
    triple = data_helper.processed_train_triple

    triple = add_bos_eos(triple,
                         bos_token_e, bos_token_r, bos_token_e,
                         eos_token_e, eos_token_r, eos_token_e,
                         is_shuffle_in_same_head=True,
                         )
    logger.debug("----- show example -----")
    for i in range(20):
        logger.debug(f"{entities[triple[i][0]]}, {relations[triple[i][1]]}, {entities[triple[i][2]]}")
    logger.debug("----- show example -----")

    bos_indices = np.where(triple[:, 0] == bos_token_e)[0]
    dataset = StoryTriple(triple, bos_indices, max_len,
                          padding_token_e, padding_token_r, padding_token_e,
                          sep_token_e, sep_token_r, sep_token_e)

    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)
    data_helper.set_loaders(dataloader, None, None, None)

    model = KgStoryTransformer01(args, num_entities, num_relations)

    if args.pre_train:
        assert args.model_path is not None
        model, _ = pre_training(
            args, logger=logger,
            data_helper=data_helper,
            model=model,
            lr=lr,
            summary_writer=summary_writer
        )
        save_model(model, args.model_path, device=args.device)

    return model, {'data_helper': data_helper, 'dataset': dataset}


def main(args=None):
    torch_fix_seed(seed=42)
    args, logger, device = setup(setup_parser, PROJECT_DIR, parser_args=args)
    version_check(torch, nn, np, pd, h5py, optuna, logger=logger)
    try:
        args.project_dir = PROJECT_DIR
        args.logger = logger
        args.device = device
        logger.debug(vars(args))
        logger.info(f"process id = {args.pid}")
        main_function(args, logger=logger)
    finally:
        save_param(args)
        pass


if __name__ == '__main__':
    main()
