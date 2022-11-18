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
from typing import List, Dict, Tuple, Optional, Union, Callable, Final, Literal, get_args

# Machine learning
import h5py
import numpy as np
import optuna
import pandas as pd
# torch
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.KGModel.kg_story_transformer import KgStoryTransformer01
from utils.numpy import negative_sampling

from models.utilMetrics.ranking_metric import RankingMetric

# Made by me
from utils.utils import force_gc, force_gc_after_function, version_check, elapsed_time_str, true_count, del_none, \
    get_true_position_items
from utils.str_process import line_up_key_value, info_str as _info_str
from utils.setup import setup, save_param, ChangeDisableNamespace
from utils.torch import (
    load_model, save_model,
    force_cuda_empty_cache_after_function, torch_fix_seed,
    DeviceName
)
from utils.textInOut import SQLITE_PREFIX
from utils.progress_manager import ProgressHelper
from utils.result_helper import ResultPerEpoch

from models.KGModel.model import (
    ConvE, DistMult, Complex, TransformerVer2E, TransformerVer2E_ERE,
    TransformerVer3E, TransformerVer3E_1, KGE_ERE, KGE_ERTails,
    MlpMixE,
)

from models.datasets.data_helper import (
    KGDATA_LITERAL, KGDATA_ALL,
    MyDataHelper, load_preprocess_data,
)
from models.datasets.datasets import MyDataset, MyDatasetWithFilter, MyTripleDataset

from ignite.engine import Engine, Events
from ignite.metrics import Average
from ignite.handlers import Timer

PROJECT_DIR = Path(__file__).resolve().parents[1]

MODEL_TMP_PATH = 'saved_models/.tmp/check-point.{}.model'

CPU: Final = 'cpu'
TRAIN: Final = 'train'
TEST: Final = 'test'
LOSS: Final = 'loss'
MRR: Final = 'mrr'
HIT_: Final = 'hit_'
STUDY: Final = 'study'

ALL_TAIL = 'all_tail'
TRIPLE = 'triple'

name2model = {
    'conve': ConvE,
    'distmult': DistMult,
    'complex': Complex,
    'transformere2': TransformerVer2E,
    'transformere2_ere': TransformerVer2E_ERE,
    'transformere3': TransformerVer3E,
    'transformere3_1': TransformerVer3E_1,
    'mlpmixere': MlpMixE,
    'kg-story-transformer01': KgStoryTransformer01
}


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
    paa('--logfile', help='the path of saving log', type=str)
    paa('--param-file', help='the path of saving param', type=str)
    paa('--tensorboard-dir', help='tensorboard direction', type=str)
    paa('--console-level', help='log level on console', type=str, default='info', choices=['info', 'debug'])
    paa('--no-show-bar', help='no show bar', action='store_true')
    paa('--device-name', help=DeviceName.ALL_INFO, type=str, default=DeviceName.CPU, choices=DeviceName.ALL_LIST)
    # select function
    paa('--function', help='function', type=str, choices=['do_1train', 'do_optuna', 'do_test'])
    # optuna setting
    paa('--optuna-file', help='optuna file', type=str)
    paa('--study-name', help='optuna study-name', type=str)
    paa('--n-trials', help='optuna n-trials', type=int, default=20)

    paa('--KGdata', help=' or '.join(KGDATA_ALL), type=str, choices=KGDATA_ALL)
    paa('--train-type', help='', type=str, choices=[ALL_TAIL, TRIPLE])
    paa('--do-negative-sampling', help='', action='store_true')
    paa('--negative-count', help='', type=int, default=None)
    paa('--eco-memory', help='メモリに優しく', action='store_true')
    paa('--entity-special-num', help='エンティティ', type=int, default=None)
    paa('--relation-special-num', help='リレーション', type=int, default=None)
    # e special
    paa('--padding-token-e', help='padding', type=int, default=None)
    paa('--cls-token-e', help='cls', type=int, default=None)
    paa('--mask-token-e', help='mask', type=int, default=None)
    # r special
    paa('--padding-token-r', help='padding', type=int, default=None)
    paa('--cls-token-r', help='cls', type=int, default=None)
    paa('--self-loop-token-r', help='self-loop', type=int, default=None)

    paa('--model', type=str, help=f"Choose from: {', '.join(name2model.keys())}")
    paa('--embedding-dim', type=int, default=200, help='The embedding dimension (1D). Default: 200')
    paa('--batch-size', help='batch size', type=int)
    paa('--epoch', help='max epoch', type=int)
    paa('--early-stopping-count', help='early-stopping-count', type=int, default=-1)

    paa('--model-path', type=str, help='model path')
    paa('--do-train', help='do-train', action='store_true')
    paa('--do-valid', help='do-valid', action='store_true')
    paa('--do-test', help='do-test', action='store_true')
    paa('--do-debug', help='do-debug', action='store_true')
    paa('--do-debug-data', help='do-debug about data', action='store_true')
    paa('--do-debug-model', help='do-debug about model', action='store_true')
    paa('--do-train-valid', help='do-train and valid', action='store_true')
    paa('--do-train-test', help='do-train and test', action='store_true')
    paa('--do-train-valid-test', help='do-train and valid and test', action='store_true')
    paa('--valid-interval', type=int, default=1, help='valid-interval', )
    paa('--embedding-shape1', type=int, default=20,
        help='The first dimension of the reshaped 2D embedding. The second dimension is inferred. Default: 20')
    # optimizer
    paa('--lr', type=float, default=0.003, help='learning rate (default: 0.003)')
    paa('--l2', type=float, default=0.0, help='Weight decay value to use in the optimizer. Default: 0.0')
    paa('--label-smoothing', type=float, default=0.1, help='Label smoothing value to use. Default: 0.1')
    # convE
    paa('--input-drop', type=float, default=0.2, help='Dropout for the input embeddings. Default: 0.2.')
    paa('--hidden-drop', type=float, default=0.3, help='Dropout for the hidden layer. Default: 0.3.')
    paa('--feat-drop', type=float, default=0.2,
        help='Dropout for the convolutional features. Default: 0.2.')
    paa('--use-bias', action='store_true', help='Use a bias in the convolutional layer. Default: True')
    paa('--lr-decay', type=float, default=0.995,
        help='Decay the learning rate by this factor every epoch. Default: 0.995')
    paa('--hidden-size', type=int, default=9728,
        help='The side of the hidden layer. The required size changes with the size of the embeddings. '
             'Default: 9728 (embedding size 200).')
    # transformere
    # paa('--input-drop', type=float, default=0.2, help='')
    # paa('--hidden-drop', type=float, default=0.3, help='')
    paa('--transformer-drop', type=float, default=0.1, help='transformer-drop. Default: 0.1.')
    paa('--position-encoder-drop', type=float, default=0.1, help='position-encoder-drop. Default: 0.1.')
    paa('--nhead', type=int, default=8, help='nhead. Default: 8.')
    paa('--dim-feedforward', type=int, default=8, help='dim-feedforward. Default: 256')
    paa('--num-layers', type=int, default=4, help='num layers. Default: 4.')

    args = parser.parse_args(args=args)
    if args.do_train_valid_test:
        del args.do_train_valid_test
        args.do_train, args.do_valid, args.do_test = True, True, True

    return args


def make_dataloader_all_tail_debug_model(data_helper: MyDataHelper, batch_size, *, logger):
    logger.debug(data_helper, batch_size)
    raise "todo"


def make_dataloader_all_tail(data_helper: MyDataHelper, batch_size, *, logger):
    logger.info("make_dataloader_all_tail")
    processed_ers = data_helper.processed_ers
    sparse_all_tail_data = data_helper.sparse_all_tail_data
    train_dataset = MyDataset(processed_ers, sparse_all_tail_data, 1, del_if_no_tail=True)
    train_valid_dataset = MyDatasetWithFilter(processed_ers, sparse_all_tail_data, 1, del_if_no_tail=True)
    valid_dataset = MyDatasetWithFilter(processed_ers, sparse_all_tail_data, 2, del_if_no_tail=True)
    test_dataset = MyDatasetWithFilter(processed_ers, sparse_all_tail_data, 3, del_if_no_tail=True)

    train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_valid = DataLoader(train_valid_dataset, batch_size=batch_size, shuffle=False)
    valid = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    data_helper.set_loaders(train, train_valid, valid, test)


def make_dataloader_triple(data_helper: MyDataHelper, batch_size):
    """
    todo
    Args:
        data_helper:
        batch_size:
        is_do_negative_sampling:
        negative_count:

    Returns:

    """
    train_triple, valid_triple, test_triple = (
        data_helper.processed_train_triple, data_helper.processed_valid_triple, data_helper.processed_test_triple)

    processed_r_is_reverse_list = data_helper.processed_r_is_reverse_list
    er2id = data_helper.processed_er2id

    train_dataset, valid_dataset, test_dataset = [
        MyTripleDataset.default_init(train_triple, processed_r_is_reverse_list, er_or_ee2id_dict=er2id
                                     ) for _triple in [train_triple, valid_triple, test_triple]
    ]

    train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_valid = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    valid = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    data_helper.set_loaders(train, train_valid, valid, test)


def get_model(args: Namespace, data_helper: MyDataHelper) -> Union[KGE_ERE, KGE_ERTails]:
    model_name = args.model

    assert model_name in name2model.keys(), f"Unknown model! :{model_name}"
    model = name2model[model_name](
        args, data_helper.processed_entity_length, data_helper.processed_relation_length, data_helper=data_helper)
    model.init()
    return model


def _use_values_in_train(args: Namespace, data_helper: MyDataHelper, do_valid: bool, uid: Optional[str]):
    device = args.device
    max_epoch = args.epoch
    early_stopping_count = args.early_stopping_count
    checkpoint_path = MODEL_TMP_PATH.format(line_up_key_value(pid=args.pid, uid=uid))
    valid_interval = args.valid_interval if do_valid else -1
    label_smoothing = args.label_smoothing
    # data
    train = data_helper.train_dataloader
    # if debug
    do_debug_model = args.do_debug_model
    l2 = args.l2
    return (
        device, max_epoch, early_stopping_count, checkpoint_path, valid_interval,
        label_smoothing, train, do_debug_model, l2
    )


@force_gc_after_function
def training_er_tails(
        args: Namespace, *, logger,
        model: KGE_ERTails, data_helper: MyDataHelper,
        lr: float,
        do_valid: bool,
        summary_writer: SummaryWriter = None,
        uid: Optional[str] = None, calculate_percent=False
):
    (device, max_epoch, early_stopping_count, checkpoint_path, valid_interval,
     label_smoothing, train, do_debug_model, l2) = _use_values_in_train(args, data_helper, do_valid, uid)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.l2)
    model.to(device)
    result = ResultPerEpoch(keywords=[LOSS, 'percent', 'negative_percent', MRR, HIT_, ])
    progress_helper: ProgressHelper = args.progress_helper
    loss_fn = model.loss

    @force_gc_after_function
    def train_step(_: Engine, batch) -> dict:
        er, e2s = batch
        opt.zero_grad()
        er, e2s = er.to(device), e2s.to(device)
        e2s = ((1.0 - label_smoothing) * e2s) + (1.0 / e2s.size(1))
        pred = model(er.split(1, 1))
        loss = loss_fn(pred, e2s)
        loss.backward()
        opt.step()
        return {LOSS: loss, 'pred': pred, 'er': er, 'e2s': batch[1].to(torch.bool)}

    def loss_transformer(output: dict):
        return output.get(LOSS).detach()

    def percent_transformer(output: dict):
        _pred, _er, _e2s = output['pred'].detach(), output['er'], output['e2s']
        _positive = _pred[_e2s].sum()
        _negative = _pred.sum() - _positive
        _positive = _positive / _pred.size(0)
        _negative = _negative / (_pred.size(0) * (_pred.size(1) - 1))
        # logger.info(f"{_positive=}, {_negative=}")
        _percent = torch.stack([_positive, _negative]).detach()
        output['percent'] = _percent
        return _percent

    trainer = Engine(train_step)
    loss_metric = Average(output_transform=loss_transformer)
    loss_metric.attach(trainer, LOSS)
    if calculate_percent:
        percent_metric = Average(output_transform=percent_transformer)
        percent_metric.attach(trainer, 'percent')

    @trainer.on(Events.EPOCH_STARTED)
    def start_epoch_func(engine: Engine):
        epoch = engine.state.epoch
        logger.debug("----- epoch: {:>5} start -----".format(epoch))
        result.start_epoch()

    @trainer.on(Events.EPOCH_COMPLETED)
    def end_epoch_func(engine: Engine):
        epoch = engine.state.epoch
        save_model(model, checkpoint_path, device=device)
        loss = engine.state.metrics[LOSS]
        result.write(LOSS, loss)
        logger.debug(f"loss = {loss}")
        if summary_writer is not None:
            logger.debug("write loss to tensorboard")
            summary_writer.add_scalar("train/loss", loss, global_step=epoch)

    @trainer.on(Events.EPOCH_COMPLETED)
    def end_epoch_func_for_percent(engine: Engine):
        epoch = engine.state.epoch
        if not calculate_percent: return
        percent, negative_percent = engine.state.metrics['percent']
        result.write('percent', percent)
        result.write('negative_percent', negative_percent)
        logger.debug(f"{percent=}, {negative_percent=}")
        if summary_writer is not None:
            logger.debug("write loss to tensorboard")
            summary_writer.add_scalar("train/percent", percent, global_step=epoch)

    @trainer.on(Events.EPOCH_COMPLETED(every=1000))
    def train_valid(engine: Engine):
        epoch = engine.state.epoch
        _, _result = testing_er_tails(
            args, model=model, data_helper=data_helper, is_train_valid=True,
            logger=logger)
        logger.debug("----- epoch: {:>5} train valid result -----".format(epoch))
        logger.debug("mrr: {}".format(_result['ranking_metric'][MRR]))
        logger.debug("hit_: {}".format([hit_i.item for hit_i in _result['ranking_metric'][HIT_]]))

    @trainer.on(Events.EPOCH_COMPLETED(every=valid_interval if valid_interval > 0 else max_epoch+1))
    def valid(engine: Engine):
        if valid_interval < 0: return
        epoch = engine.state.epoch
        _, _result = testing_er_tails(
            args, model=model, data_helper=data_helper, is_valid=True,
            logger=logger)
        result.write_all({key: _result['ranking_metric'][key] for key in (MRR, HIT_)})
        logger.debug("----- epoch: {:>5} valid result -----".format(epoch))
        logger.debug("mrr: {}".format(_result['ranking_metric'][MRR]))
        logger.debug("hit_: {}".format([hit_i.item() for hit_i in _result['ranking_metric'][HIT_]]))
        if summary_writer is not None:
            logger.debug("write  to tensorboard")
            summary_writer.add_scalar("valid/loss", _result[LOSS], global_step=epoch)
            summary_writer.add_scalar("valid/mrr", _result['ranking_metric'][MRR], global_step=epoch)
            summary_writer.add_scalars(
                "valid/hit", global_step=epoch,
                tag_scalar_dict={str(i + 1): hit_i.item() for i, hit_i in enumerate(_result['ranking_metric'][HIT_])},
            )

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
    def logging_time_per_epoch(engine: Engine):
        epoch = engine.state.epoch
        time_str = elapsed_time_str(total_timer.value())
        print_text = "----- epoch: {:>5} complete. time: {:>8.2f}. total time: {:>7} -----".format(
            epoch, engine.state.times['EPOCH_COMPLETED'], time_str)

        logger.info(print_text)

    @trainer.on(Events.ITERATION_COMPLETED(every=100))
    def print_info_per_some_iter(engine: Engine):
        output = engine.state.output
        epoch = engine.state.epoch
        print_text = ', '.join(del_none([
            f"loss={output[LOSS].item()}",
            f"positive={output['percent'][0].item()}" if 'percent' in output else None,
            f"negative={output['percent'][1].item()}" if 'percent' in output else None,
        ]))
        logger.debug("----- epoch: {:>5} iter {:>6} complete. total time: {:>7} -----".format(
            epoch, engine.state.iteration, elapsed_time_str(total_timer.value())))
        logger.debug(print_text)

    # about progress_helper
    trainer.add_event_handler(Events.STARTED, lambda x: progress_helper.add_key(TRAIN, total=max_epoch))
    trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda x: progress_helper.update_key(TRAIN))
    trainer.add_event_handler(Events.COMPLETED, lambda x: progress_helper.finish_key(TRAIN))
    try:
        trainer.run(train, max_epochs=max_epoch)
    except KeyboardInterrupt as e:
        load_model(model, checkpoint_path, device=device, delete_file=True)
        raise e

    del total_timer, trainer
    load_model(model, checkpoint_path, device=device, delete_file=True)
    return model, result


@force_gc_after_function
def training_ere(
        args: Namespace, *, logger,
        model: KGE_ERTails, data_helper: MyDataHelper,
        lr: float,
        do_valid: bool,
        summary_writer: SummaryWriter = None,
        uid: Optional[str] = None,
        calculate_percent=False
):
    (device, max_epoch, early_stopping_count, checkpoint_path, valid_interval,
     label_smoothing, train, do_debug_model, l2) = _use_values_in_train(args, data_helper, do_valid, uid)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.l2)
    model.to(device)
    result = ResultPerEpoch(keywords=[LOSS, MRR, HIT_, ])
    progress_helper: ProgressHelper = args.progress_helper
    count_per_items = data_helper.processed_id2count_entity
    loss_fn = model.loss

    @force_gc_after_function
    def train_step(_: Engine, batch) -> dict:
        head, relation, tail, _, _ = batch
        batch_size = len(head)
        opt.zero_grad()
        head, relation, tail = head.to(device), relation.to(device), tail.to(device)

        head, relation = head.view(batch_size, 1), relation.view(batch_size, 1)
        negative_tails = torch.from_numpy(
            negative_sampling(np.arange(len(count_per_items)), count_per_items, size=batch_size)
        ).to(device).reshape(batch_size, 1)

        pred_embedding = model((head, relation)).reshape(batch_size, -1)
        positive_embedding = model.emb_e(tail).reshape(batch_size, -1)
        negative_embedding = model.emb_e(negative_tails).reshape(batch_size, -1)

        # positive_score = torch.linalg.norm(pred_embedding - positive_embedding, dim=1)
        # negative_score = torch.linalg.norm(pred_embedding - negative_embedding, dim=1)

        positive_score = (pred_embedding * positive_embedding).sum(dim=1)
        negative_score = (pred_embedding * negative_embedding).sum(dim=1)

        loss = (+ loss_fn(positive_score, torch.ones_like(positive_score, device=device))
                + loss_fn(negative_score, torch.zeros_like(negative_score, device=device)))

        loss.backward()
        opt.step()
        return {LOSS: loss, 'pred': pred_embedding, 'head': head, 'relation': relation, tail: 'tail'}

    def loss_transformer(output: dict):
        return output.get(LOSS).detach()

    trainer = Engine(train_step)
    loss_metric = Average(output_transform=loss_transformer)
    loss_metric.attach(trainer, LOSS)

    @trainer.on(Events.EPOCH_STARTED)
    def start_epoch_func(engine: Engine):
        epoch = engine.state.epoch
        logger.debug("----- epoch: {:>5} start -----".format(epoch))
        result.start_epoch()

    @trainer.on(Events.EPOCH_COMPLETED)
    def end_epoch_func(engine: Engine):
        epoch = engine.state.epoch
        save_model(model, checkpoint_path, device=device)
        loss = engine.state.metrics[LOSS]
        result.write(LOSS, loss)
        logger.debug(f"loss = {loss}")
        if summary_writer is not None:
            logger.debug("write loss to tensorboard")
            summary_writer.add_scalar("train/loss", loss, global_step=epoch)

    @trainer.on(Events.EPOCH_COMPLETED(every=1000))
    def train_valid(engine: Engine):
        epoch = engine.state.epoch
        _, _result = testing_ere(
            args, model=model, data_helper=data_helper, is_train_valid=True,
            logger=logger)
        logger.debug("----- epoch: {:>5} train valid result -----".format(epoch))
        logger.debug("mrr: {}".format(_result['ranking_metric'][MRR]))
        logger.debug("hit_: {}".format([hit_i.item for hit_i in _result['ranking_metric'][HIT_]]))

    @trainer.on(Events.EPOCH_COMPLETED(every=valid_interval))
    def valid(engine: Engine):
        epoch = engine.state.epoch
        _, _result = testing_ere(
            args, model=model, data_helper=data_helper, is_valid=True,
            logger=logger)
        result.write_all({key: _result['ranking_metric'][key] for key in (MRR, HIT_)})
        logger.debug("----- epoch: {:>5} valid result -----".format(epoch))
        logger.debug("mrr: {}".format(_result['ranking_metric'][MRR]))
        logger.debug("hit_: {}".format([hit_i.item() for hit_i in _result['ranking_metric'][HIT_]]))
        if summary_writer is not None:
            logger.debug("write  to tensorboard")
            summary_writer.add_scalar("valid/loss", _result[LOSS], global_step=epoch)
            summary_writer.add_scalar("valid/mrr", _result['ranking_metric'][MRR], global_step=epoch)
            summary_writer.add_scalars(
                "valid/hit", global_step=epoch,
                tag_scalar_dict={str(i + 1): hit_i.item() for i, hit_i in enumerate(_result['ranking_metric'][HIT_])},
            )

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

    # about progress_helper
    trainer.add_event_handler(Events.STARTED, lambda x: progress_helper.add_key(TRAIN, total=max_epoch))
    trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda x: progress_helper.update_key(TRAIN))
    trainer.add_event_handler(Events.COMPLETED, lambda x: progress_helper.finish_key(TRAIN))
    try:
        trainer.run(train, max_epochs=max_epoch)
    except KeyboardInterrupt as e:
        load_model(model, checkpoint_path, device=device, delete_file=True)
        raise e

    del total_timer, trainer
    load_model(model, checkpoint_path, device=device, delete_file=True)
    return model, result


@force_gc_after_function
@torch.no_grad()
def testing_er_tails(
        args: Namespace, *, logger,
        model, data_helper: MyDataHelper,
        is_train_valid=False, is_valid=False, is_test=False
):
    model = model
    device = args.device
    test = data_helper.get_dataloader(False, is_train_valid, is_valid, is_test)

    def eval_step(_: Engine, batch) -> dict[str, torch.Tensor]:
        er, e2s, e2s_all = batch
        e, r = er.to(device).split(1, 1)
        e2s = e2s.to(device)
        e2s_all = e2s.to(device)
        pred: torch.Tensor = model.forward((e, r))
        loss = model.loss(pred, e2s)
        return {LOSS: loss, 'pred': pred, 'e2s': e2s, 'e2s_all': e2s_all != 0}

    def ranking_metric_transform(output: dict) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        pred, e2s, e2s_all = map(output.get, ['pred', 'e2s', 'e2s_all'])
        row, column = torch.where(e2s == 1)
        return pred, (row, column), e2s_all

    # test
    model.to(device)
    model.eval()
    evaluator = Engine(eval_step)
    ranking_metric = RankingMetric(output_transform=ranking_metric_transform)
    ranking_metric.attach(evaluator, 'ranking_metric')
    log_loss_metric = Average(output_transform=lambda x: x.get(LOSS).detach())
    log_loss_metric.attach(evaluator, LOSS)
    evaluator.run(test)
    result = {key: value for key, value in evaluator.state.metrics.items()}
    return model, result


@force_gc_after_function
@torch.no_grad()
def testing_ere(
        args: Namespace, *, logger,
        model, data_helper: MyDataHelper,
        is_train_valid=False, is_valid=False, is_test=False
):
    model: KGE_ERE = model
    device = args.device
    assert true_count(is_train_valid, is_valid, is_test) == 1
    test = data_helper.get_dataloader(False, is_train_valid, is_valid, is_test)
    all_tail = data_helper.sparse_all_tail_data
    loss_fn = model.loss

    def eval_step(engine: Engine, batch) -> dict[str, torch.Tensor]:
        head, relation, tail, _, er_id = batch
        e2s_all = torch.index_select(all_tail, 0, er_id).to_dense()
        batch_size = len(head)
        head, relation, tail = head.to(device), relation.to(device), tail.to(device)
        head, relation = head.view(batch_size, 1), relation.view(batch_size, 1)
        pred_embedding = model((head, relation)).reshape(batch_size, -1)

        all_e_embeddings = model.emb_e.weight
        pred_all_score = torch.mm(pred_embedding, all_e_embeddings.transpose(0, 1))

        # positive_score = torch.linalg.norm(pred_embedding - positive_embedding, dim=1)
        positive_score = pred_all_score[torch.arange(len(pred_all_score), device=device), tail]
        loss = loss_fn(positive_score, torch.ones_like(positive_score, device=device))

        return {LOSS: loss, 'pred': pred_all_score, 'tail': tail, 'e2s_all': e2s_all != 0}

    def ranking_metric_transform(output: dict) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        pred, tail, e2s_all = map(output.get, ['pred', 'tail', 'e2s_all'])
        row, column = torch.arange(len(tail), device=device), tail
        return pred, (row, column), e2s_all

    # test
    model.to(device)
    model.eval()
    evaluator = Engine(eval_step)
    ranking_metric = RankingMetric(output_transform=ranking_metric_transform)
    ranking_metric.attach(evaluator, 'ranking_metric')
    log_loss_metric = Average(output_transform=lambda x: x.get(LOSS).detach())
    log_loss_metric.attach(evaluator, LOSS)
    evaluator.run(test)
    result = {key: value for key, value in evaluator.state.metrics.items()}
    return model, result


def train_setup(args, *, logger: Logger):
    kg_data: KGDATA_LITERAL = args.KGdata
    entity_special_num: int = args.entity_special_num
    relation_special_num: int = args.relation_special_num
    batch_size: int = args.batch_size
    train_type: str = args.train_type

    assert batch_size is not None and train_type is not None

    # load data
    logger.info("{} start".format("load data"))
    data_helper = load_preprocess_data(kg_data, entity_special_num, relation_special_num, logger=logger)
    logger.info("{} end".format("load data"))

    # dataloader
    logger.info("{} start".format("make dataloader"))
    if train_type == ALL_TAIL:
        if args.do_debug_model:
            make_dataloader_all_tail_debug_model(data_helper, batch_size, logger=logger)
        else:
            make_dataloader_all_tail(data_helper, batch_size, logger=logger)
    elif train_type == TRIPLE:
        make_dataloader_triple(data_helper, batch_size)
        pass
    else:
        raise "Error."
        pass
    logger.info("{} end".format("make dataloader"))

    # model
    logger.info("{} start".format("make model"))
    model = get_model(args, data_helper)
    logger.info("{} end".format("make model"))

    return data_helper, model


def train_setup_for_kgc_data(args, *, logger: Logger):
    kg_data: KGDATA_LITERAL = args.KGdata
    entity_special_num: int = args.entity_special_num
    relation_special_num: int = args.relation_special_num
    batch_size: int = args.batch_size
    train_type: str = args.train_type

    assert batch_size is not None and train_type is not None

    # load data
    logger.info("{} start".format("load data"))
    data_helper = MyDataHelper(
        info_path, all_tail, train_path, None, None, logger=logger,
        entity_special_num=entity_special_num, relation_special_num=relation_special_num,
    )
    logger.info("{} end".format("load data"))

    # dataloader
    logger.info("{} start".format("make dataloader"))
    if train_type == ALL_TAIL:
        if args.do_debug_model:
            make_dataloader_all_tail_debug_model(data_helper, batch_size, logger=logger)
        else:
            make_dataloader_all_tail(data_helper, batch_size, logger=logger)
    elif train_type == TRIPLE:
        make_dataloader_triple(data_helper, batch_size)
        pass
    else:
        raise "Error."
        pass
    logger.info("{} end".format("make dataloader"))

    # model
    logger.info("{} start".format("make model"))
    model = get_model(args, data_helper)
    logger.info("{} end".format("make model"))

    return data_helper, model


def do_1train(args: Namespace, *, logger: Logger):
    """
    Args:
        args: Namespace
        logger: Logger

    Returns: None

    """
    assert type(args) is ChangeDisableNamespace
    is_do_train, is_do_valid, is_do_test = args.do_train, args.do_valid, args.do_test
    is_do_debug = args.do_debug
    train_type = args.train_type
    model_path = args.model_path
    device = args.device
    summary_writer = SummaryWriter(log_dir=args.tensorboard_dir)
    train_func, test_func = None if train_type not in [ALL_TAIL, TRIPLE] \
        else (training_er_tails, testing_er_tails) if train_type == ALL_TAIL \
        else (training_ere, testing_ere) if train_type == TRIPLE \
        else None

    logger.info(f"Function start".center(40, '='))

    if (not is_do_train) and (not is_do_valid) and (not is_do_test) and (not is_do_debug):
        logger.info(f"Function end".center(40, '='))
        return -1

    data_helper, model = train_setup(args, logger=logger)
    # training
    lr = args.lr

    if is_do_debug:
        logger.info(model)
        return data_helper, model

    if is_do_train:
        assert model_path is not None
        model, result = train_func(
            args, logger=logger,
            data_helper=data_helper,
            model=model,
            lr=lr,
            do_valid=is_do_valid,
            summary_writer=summary_writer
        )
        save_model(model, args.model_path, device=device)
        del result

    model = load_model(model, model_path, device=device)

    if is_do_valid:
        logger.info(_info_str(f"Test valid start."))
        model, result = test_func(
            args, logger=logger,
            data_helper=data_helper, model=model,
            is_valid=True,
        )
        # args.test_valid_result = result
        logger.info(f"===== Test valid result =====")
        logger.info(f"mrr: {result[MRR]}, ")
        logger.info(f"hit_: {[h.item() for h in result[HIT_]]}")
        logger.info(_info_str(f"Test valid complete."))

    if is_do_test:
        logger.info(_info_str(f"Test start."))
        model, result = test_func(
            args, logger=logger,
            data_helper=data_helper, model=model,
            is_test=True,
        )
        # args.test_result = result
        logger.info(f"===== Test result =====")
        logger.info(f"mrr: {result[MRR]}, ")
        logger.info(f"hit_: {[h.item() for h in result[HIT_]]}")
        logger.info(_info_str(f"Test complete."))

    logger.info(f"Function finish".center(40, '='))


def do_optuna(args, *, logger: Logger):
    assert type(args) is ChangeDisableNamespace
    is_do_train, is_do_valid, is_do_test = args.do_train, args.do_valid, args.do_test
    study_name = args.study_name
    optuna_file = args.optuna_file
    n_trials = args.n_trials
    progress_helper: ProgressHelper = args.progress_helper
    data_helper, _ = train_setup(args, logger=logger)

    @progress_helper.update_progress_after_function(STUDY)
    @force_cuda_empty_cache_after_function
    def objective(trial: optuna.Trial):
        nonlocal data_helper
        lr = trial.suggest_loguniform('lr', 1e-6, 5e-2)

        logger.info(_info_str(f"trial {trial.number} start."))
        model = get_model(args, data_helper)
        model.init()
        # train
        model, _ = training_er_tails(
            args, logger=logger,
            data_helper=data_helper,
            model=model,
            lr=lr, do_valid=is_do_valid
        )
        # valid
        _, result = testing_er_tails(
            args, logger=logger,
            data_helper=data_helper, model=model,
            is_valid=True,
        )
        logger.info(f"=====Valid result. mrr: {result[MRR]}, hit_: {result[HIT_]}")
        return result[MRR]

    study = optuna.create_study(
        direction='maximize', study_name=study_name, storage=SQLITE_PREFIX + optuna_file, load_if_exists=True
    )
    progress_helper.add_key(STUDY, total=n_trials)
    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)
    progress_helper.finish_key(STUDY)

    logger.info(_info_str("optuna study finish"))
    logger.info(f"==========best param = lr: {study.best_params['lr']}")
    logger.info(_info_str("update arge param"))

    logger.info(f"Optuna finish".center(40, '='))
    return study.best_params


def select_function(args, *, logger: Logger):
    fname = args.function
    if fname == 'do_test':
        args.do_train = False
        args.do_valid = True
        args.do_test = True

    if fname not in ('do_1train', 'do_optuna', 'do_test'):
        raise "you should select function"
    elif fname == 'do_1train':
        with ChangeDisableNamespace(args) as const_args:
            do_1train(const_args, logger=logger)

    elif fname == 'do_optuna':
        with ChangeDisableNamespace(args) as const_args:
            best_params = do_optuna(const_args, logger=logger)
        args.lr = best_params['lr']
        force_gc()
        with ChangeDisableNamespace(args) as const_args:
            do_1train(const_args, logger=logger)


def main(args=None):
    torch_fix_seed(seed=42)
    args, logger, device = setup(setup_parser, PROJECT_DIR, parser_args=args)
    version_check(torch, nn, np, pd, h5py, optuna, logger=logger)
    try:
        args.project_dir = PROJECT_DIR
        args.logger = logger
        args.device = device
        args.completed = {}
        args.progress_helper = ProgressHelper("log/progress.{pid}.txt", pid=args.pid)
        logger.debug(vars(args))
        logger.info(f"process id = {args.pid}")
        select_function(args, logger=logger)
        args.progress_helper.finish(delete=True)
    finally:
        del args.progress_helper
        save_param(args)
        pass


if __name__ == '__main__':
    main()
