#!/usr/bin/python
# -*- coding: utf-8 -*-
"""run script for Knowledge Graph Challenge

* This script is the script about Knowledge Graph Challenge.
* Define data, define model, define parameters, and run train.

Todo:
    * 色々

"""
# ========== python ==========
import warnings
from argparse import Namespace
from itertools import chain
from logging import Logger
from operator import itemgetter
from typing import Optional, Callable, Final, cast, Sequence

import h5py
# Machine learning
import numpy as np
import optuna
import pandas as pd
# torch
import torch
# torch ignite
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import Engine
from ignite.handlers import Checkpoint
from ignite.metrics import Average, Accuracy
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
# My Models
from models.KGModel.kg_sequence_transformer import (
    KgSequenceTransformer01, KgSequenceTransformer02, KgSequenceTransformer03, KgSequenceTransformer03preInit,
    KgSequenceTransformer00, KgSequenceTransformer)
from models.KGModel.translation import TransE
from models.datasets.data_helper import (
    MyDataHelperForStory, DefaultTokens, DefaultIds, SpecialTokens01 as SpecialTokens, MyDataLoaderHelper, )
from models.datasets.datasets_for_sequence import SimpleTriple, StoryTriple
from models.utilLoss.focal_loss import FocalLoss, GAMMA
from models.utilLoss.utils import LossFnName
# My utils
from utils.error import UnderDevelopmentError
from utils.torch import save_model, torch_fix_seed, DeviceName
from utils.typing import ConstMeta
from utils.utils import version_check
from utils.torch_ignite import (
    set_write_model_param_function, set_start_epoch_function, set_end_epoch_function,
    set_valid_function, training_with_ignite, set_early_stopping_function
)
# My const words about words used as tags
from models.KGModel.kg_sequence_transformer import (
    ALL_WEIGHT_LIST, HEAD_MASKED_LM, TAIL_MASKED_LM, RELATION_MASKED_LM, )
# My const value about torch ignite
from utils.torch_ignite import (TRAINER, EVALUATOR, GOOD_LOSS_CHECKPOINTE, LAST_CHECKPOINTE)
# My const words about words used as tags
from const.const_values import (CPU, MODEL, LOSS, PARAMS, LR,
                                DATA_HELPER, DATASETS, DATA_LOADERS, TRAIN_RETURNS,
                                HEAD_LOSS, RELATION_LOSS, TAIL_LOSS,
                                HEAD_ACCURACY, RELATION_ACCURACY, TAIL_ACCURACY,
                                HEAD_ANS, RELATION_ANS, TAIL_ANS,
                                HEAD_PRED, RELATION_PRED, TAIL_PRED,
                                PRE_TRAIN_SCALER_TAG_GETTER, PRE_VALID_SCALER_TAG_GETTER,
                                PRE_TRAIN_MODEL_WEIGHT_TAG_GETTER,
                                HEAD_METRIC_NAMES, RELATION_METRIC_NAMES, TAIL_METRIC_NAMES)
# My const words about file direction and title
from const.const_values import (
    PROJECT_DIR,
    ALL_TITLE_LIST, SRO_ALL_TRAIN_FILE, SRO_ALL_INFO_FILE, TITLE2SRO_FILE090, TITLE2SRO_FILE075, ABOUT_KILL_WORDS,
    LR_HEAD, LR_RELATION, LR_TAIL, LOSS_FUNCTION, STUDY,
)

from run_for_KGC import make_get_data_helper, make_get_datasets as make_get_kgc_datasets, get_all_tokens, \
    make_get_dataloader, do_train_test_ect


def fix_args(args: Namespace):
    """fix args
    Args:
        args(Namespace):

    Returns:
        Namespace
    """
    args.max_len = 1
    return args


def setup_parser(args: Optional[Sequence[str]] = None) -> Namespace:
    """make parser function

    * My first-setup function needs the function which make and return parser.

    Args:
        args(:obj:`Sequence[str]`, optional): args list or None. Default to None.

    Returns:
        Namespace: your args instance.

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
    paa('--SEED', type=int, default=42, help='seed. default 42 (It has no mean.) ')
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
    paa4('--entity-special-num', help='entity special num', type=int, default=5)
    paa4('--relation-special-num', help='relation special num', type=int, default=5)
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
    # model
    parser_group05 = parser.add_argument_group('model setting', 'There are the setting of model params.')
    paa5 = parser_group05.add_argument
    paa5('--entity-embedding-dim', help='The embedding dimension. Default: 128', type=int, default=128)
    paa5('--relation-embedding-dim', help='The embedding dimension. Default: 128', type=int, default=128)
    paa5('--batch-size', help='batch size', type=int, default=4)
    paa5('--init-embedding-using-bert', action='store_true',
         help='if it is set and the model is 03a, it will be pre_init by bert')
    # optimizer
    parser_group06 = parser.add_argument_group('model optimizer setting',
                                               'There are the setting of model optimizer params.')
    paa6 = parser_group06.add_argument
    paa6('--lr', type=float, default=0.003, help='learning rate (default: 0.003)')
    paa6('--loss-function', type=str, default=LossFnName.CROSS_ENTROPY_LOSS, choices=LossFnName.ALL_LIST(),
         help='loss function (default: CrossEntropyLoss)')
    paa6('--epoch', help='max epoch', type=int, default=2)
    paa6('--early-stopping', action='store_true', help='', )
    paa6('--early-stopping-count', type=int, default=10, )

    paa6('--valid-interval', type=int, default=1, help='valid-interval', )
    # if focal loss, this is need.
    parser_group061 = parser.add_argument_group('focal loss setting',
                                                'There are the setting of focal loss.')
    paa61 = parser_group061.add_argument
    paa61('--gamma', type=float, help='gamma')

    args = parser.parse_args(args=args)
    # old to new
    return fix_args(args)


def pre_training(args: Namespace, hyper_params, data_helper, data_loaders, model, *, logger, summary_writer) -> dict:
    """pre-training function.

    * main part of my train.

    Args:
        args(Namespace): args.
        hyper_params(tuple): hyper parameters.
        data_helper(MyDataHelper): MyDataHelper instance.
        data_loaders(MyDataLoaderHelper): MyDataLoaderHelper instance. It has .train_dataloader and .valid_dataloader.
        model(KgSequenceTransformer): model.
        summary_writer(SummaryWriter|None):
            tensorboard's SummaryWriter instance. if it is None, don't write to tensorboard.
        logger(Logger): logging.Logger.

    Returns:
        dict: keys=(MODEL, TRAINER, EVALUATOR, (CHECKPOINTER_GOOD_LOSS, CHECKPOINTER_LAST))

    """
    (lr, lr_head, lr_relation, lr_tail, loss_fn_name, other_params) = hyper_params
    do_weight_loss = False
    device: torch.device = args.device
    max_len = args.max_len
    max_epoch = args.epoch
    mask_token_e, mask_token_r = args.mask_token_e, args.mask_token_r
    bos_token_e, bos_token_r = args.bos_token_e, args.bos_token_r
    checkpoint_dir = args.checkpoint_dir
    resume_checkpoint_path = args.resume_checkpoint_path
    is_resume_from_checkpoint = args.resume_from_checkpoint
    is_resume_from_last_point = args.resume_from_last_point

    non_blocking = args.non_blocking

    # optional function
    def cpu_deep_copy_or_none(_tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """return deep copied tensor or None.

        * deep clone to cpu from gpu. However, if _tensor is None, return None.

        Args:
            _tensor(Optional[torch.Tensor]): Tensor or None item.

        Returns:
            Optional[torch.Tensor]: If input is None, return None. else return clone tensor(device=cpu)

        """
        return _tensor.to(CPU, non_blocking=non_blocking).detach().clone() if _tensor is not None else None


    def negative_sampling(_triple, weights):
        assert _triple.shape[1]==3
        _negative_triple = _triple.detach().clone()  # method e
        _negative_triple[:, 3] = torch.multinomial(weights, len(_triple), replacement=True)
        return _negative_triple


    entity_num, relation_num = data_helper.processed_entity_num, data_helper.processed_relation_num
    train = data_loaders.train_dataloader
    train_dataset = train.dataset
    valid = data_loaders.valid_dataloader if args.train_valid_test else None
    # count frequency list
    _, _, tail_index2count = train_dataset.get_index2count(device)

    opt = torch.optim.Adam([{PARAMS: model.parameters(), LR: lr}])
    # loss function setting
    gamma = other_params.get(GAMMA, None)
    if do_weight_loss:
        loss_fn_entity = None
        loss_fn_relation = None
        raise UnderDevelopmentError("todo")
    else:
        if loss_fn_name not in LossFnName.ALL_LIST():
            raise ValueError(f"The loss name {loss_fn_name} is not defined.")
        elif loss_fn_name == LossFnName.CROSS_ENTROPY_LOSS:
            loss_fn_entity = CrossEntropyLoss(weight=torch.ones(entity_num).to(device))
            loss_fn_relation = CrossEntropyLoss(weight=torch.ones(relation_num).to(device))
        elif loss_fn_name == LossFnName.FOCAL_LOSS:
            if gamma is None: raise ValueError("gamma must not None")
            loss_fn_entity = FocalLoss(weight=torch.ones(entity_num).to(device), gamma=gamma)
            loss_fn_relation = FocalLoss(weight=torch.ones(relation_num).to(device), gamma=gamma)

    # main train step
    # noinspection PyTypeChecker
    def train_step(_, batch) -> dict:
        """train step

        * changing study parameter by this function.

        """
        model.train()
        triple = batch
        batch_size = triple.shape[0]
        assert triple.shape == (batch_size, 3)
        # train start
        opt.zero_grad()

        triple: torch.Tensor = triple.to(device, non_blocking=non_blocking)
        negative_tail_triple =negative_sampling(triple, tail_index2count)
        assert len(triple) == len(negative_tail_triple)

        all_score = model(torch.cat([triple, negative_tail_triple]))

        loss: torch.Tensor = torch.tensor(0, dtype=torch.float).to(device)
        loss.backward()
        opt.step()

        # return values
        return_dict = {
            TAIL_ANS: cpu_deep_copy_or_none(mask_ans_tail),
            HEAD_PRED: cpu_deep_copy_or_none(head_pred),
            RELATION_PRED: cpu_deep_copy_or_none(relation_pred),
            TAIL_PRED: cpu_deep_copy_or_none(tail_pred),
            HEAD_LOSS: cpu_deep_copy_or_none(head_loss),
            RELATION_LOSS: cpu_deep_copy_or_none(relation_loss),
            TAIL_LOSS: cpu_deep_copy_or_none(object_loss),
            LOSS: cpu_deep_copy_or_none(loss),
        }
        return return_dict

    # main valid step
    @torch.no_grad()
    def valid_step(_, batch) -> dict:
        """valid step

        * valid model by valid data.

        """
        model.eval()
        triple: torch.Tensor = batch[0].to(device, non_blocking=non_blocking)
        valid_filter: torch.Tensor = batch[1].to(device, non_blocking=non_blocking)

        # triple.shape == (batch, max_len, 3) and valid_filter.shape == (batch, max_len)

        def get_valid_loss(_index, _loss_fn):
            """get valid loss

            """
            _triple_for_valid = triple.clone()
            _triple_for_valid[:, :, _index][valid_filter] = mask_token_e
            _model_input = [None, None, None]
            _model_input[_index] = valid_filter
            _, _pred_list = model(_triple_for_valid, *_model_input)
            _pred = _pred_list[_index]
            _valid_ans = triple[:, :, _index][valid_filter]
            _loss = _loss_fn(_pred, _valid_ans)
            return _pred, _valid_ans, _loss

        loss: torch.Tensor = torch.tensor(0, dtype=torch.float).to(device)
        head_valid_ans, relation_valid_ans, tail_valid_ans = None, None, None
        head_pred, relation_pred, tail_pred = None, None, None
        head_loss, relation_loss, tail_loss = None, None, None
        if len(valid_filter) > 0:
            if is_do_head_mask:
                head_pred, head_valid_ans, head_loss = get_valid_loss(0, loss_fn_entity)
                loss += head_loss  # * head_valid_ans
            if is_do_relation_mask:
                relation_pred, relation_valid_ans, relation_loss = get_valid_loss(1, loss_fn_relation)
                loss += relation_loss  # * relation_valid_ans
            if is_do_tail_mask:
                tail_pred, tail_valid_ans, tail_loss = get_valid_loss(2, loss_fn_entity)
                loss += tail_loss  # * object_valid_ans
                if tail_loss < 0: raise ValueError("error")

        # return dict
        return_dict = {
            HEAD_ANS: cpu_deep_copy_or_none(head_valid_ans),
            RELATION_ANS: cpu_deep_copy_or_none(relation_valid_ans),
            TAIL_ANS: cpu_deep_copy_or_none(tail_valid_ans),
            HEAD_PRED: cpu_deep_copy_or_none(head_pred),
            RELATION_PRED: cpu_deep_copy_or_none(relation_pred),
            TAIL_PRED: cpu_deep_copy_or_none(tail_pred),
            LOSS: cpu_deep_copy_or_none(loss),
            HEAD_LOSS: cpu_deep_copy_or_none(head_loss),
            RELATION_LOSS: cpu_deep_copy_or_none(relation_loss),
            TAIL_LOSS: cpu_deep_copy_or_none(tail_loss),
        }
        return return_dict

    metric_names = [LOSS]
    if is_do_head_mask: metric_names += HEAD_METRIC_NAMES
    if is_do_relation_mask: metric_names += RELATION_METRIC_NAMES
    if is_do_tail_mask: metric_names += TAIL_METRIC_NAMES
    logger.info(f"metric names: {metric_names}")

    def set_engine_metrics(step: Callable) -> tuple[Engine, dict]:
        """This is the function to set engine and metrics.

        Args:
            step(Callable): step function.

        Returns:
            tuple[Engine, dict]: Engine and matrix dict.

        """
        engine = Engine(step)
        ProgressBar().attach(engine)
        head_getter = itemgetter(HEAD_PRED, HEAD_ANS)
        relation_getter = itemgetter(RELATION_PRED, RELATION_ANS)
        tail_getter = itemgetter(TAIL_PRED, TAIL_ANS)
        getter_list = (head_getter, relation_getter, tail_getter)

        # loss and average of trainer
        metrics = {
            LOSS: Average(itemgetter(LOSS)),
            HEAD_LOSS: Average(itemgetter(HEAD_LOSS)),
            RELATION_LOSS: Average(itemgetter(RELATION_LOSS)),
            TAIL_LOSS: Average(itemgetter(TAIL_LOSS)),
            **{_key: Accuracy(_getter) for _key, _getter in zip(ACCURACY_NAME3, getter_list)},
            **{_key: TopKCategoricalAccuracy(1, _getter) for _key, _getter in zip(TOP1_NAME3, getter_list)},
            **{_key: TopKCategoricalAccuracy(3, _getter) for _key, _getter in zip(TOP3_NAME3, getter_list)},
            **{_key: TopKCategoricalAccuracy(10, _getter) for _key, _getter in zip(TOP10_NAME3, getter_list)},
        }

        [_value.attach(engine, _key) for _key, _value in metrics.items()
         if _key in metric_names and _key in metrics]

        return engine, metrics

    model.to(device)
    trainer, trainer_matrix = set_engine_metrics(train_step)

    _kwargs = dict(summary_writer=summary_writer, metric_names=metric_names, param_names=ALL_WEIGHT_LIST, logger=logger)
    set_start_epoch_function(trainer, optional_func=train_dataset.per1epoch, logger=logger)
    set_end_epoch_function(trainer, PRE_TRAIN_SCALER_TAG_GETTER, **_kwargs)
    set_write_model_param_function(
        trainer, model, PRE_TRAIN_MODEL_WEIGHT_TAG_GETTER, lambda _key: getattr(model, _key).data, **_kwargs)

    # valid function
    if args.train_valid_test:
        evaluator, evaluator_matrix = set_engine_metrics(valid_step)
        set_valid_function(trainer, evaluator, valid, args.valid_interval, PRE_VALID_SCALER_TAG_GETTER, **_kwargs)
        # early stopping
        if args.early_stopping:
            set_early_stopping_function(
                trainer, evaluator, args.early_stopping_count, lambda engine: -engine.state.metrics[LOSS])
    else:
        evaluator, evaluator_matrix = None, None
        pass

    # Interrupt
    if args.only_load_trainer_evaluator:
        logger.info("load trainer and evaluator. then end")
        return {MODEL: model, TRAINER: trainer, EVALUATOR: evaluator,
                GOOD_LOSS_CHECKPOINTE: None, LAST_CHECKPOINTE: None}
    else:
        logger.info("training with pytorch ignite")
        good_checkpoint, last_checkpoint, dict_ = training_with_ignite(
            model, opt, max_epoch, trainer, evaluator, checkpoint_dir,
            resume_checkpoint_path, is_resume_from_checkpoint, is_resume_from_last_point,
            train=train, device=device, non_blocking=non_blocking, logger=logger)
        return {MODEL: model, TRAINER: trainer, EVALUATOR: evaluator,
                GOOD_LOSS_CHECKPOINTE: good_checkpoint, LAST_CHECKPOINTE: last_checkpoint}


def make_get_model(args: Namespace, *, data_helper: MyDataHelperForStory, logger: Logger):
    entity_embedding_dim, relation_embedding_dim = args.entity_embedding_dim, args.relation_embedding_dim
    entity_num, relation_num = data_helper.processed_entity_num, data_helper.processed_relation_num
    model = TransE(entity_embedding_dim, relation_embedding_dim, entity_num, relation_num)
    return model


def make_get_datasets(args: Namespace, *, data_helper, logger: Logger):
    """make_get_datasets

    Returns:
        tuple[Dataset, Dataset, Dataset]: dataset_train, dataset_valid, dataset_test

    """
    dataset_train, dataset_valid, dataset_test = make_get_kgc_datasets(args, data_helper=data_helper, logger=logger)
    dataset_train = SimpleTriple(
        dataset_train.triple[:dataset_train.sequence_length].to('cpu').detach().numpy().copy(),
        data_helper.processed_entity_num, data_helper.processed_relation_num)
    return dataset_train, dataset_valid, dataset_test


def main_function(args: Namespace, *, logger: Logger):
    """main_function

    Args:
        args(Namespace):
        logger(Logger):

    """
    # load raw data and make datahelper. Support for special-tokens by datahelper.
    logger.info('----- make datahelper start. -----')
    data_helper = make_get_data_helper(args, logger=logger)
    logger.info('----- make datahelper complete. -----')
    # make dataset.
    logger.info('----- make datasets start. -----')
    datasets = make_get_datasets(args, data_helper=data_helper, logger=logger)
    logger.info('----- make dataloader start. -----')
    data_loaders = make_get_dataloader(args, datasets=datasets, logger=logger)
    logger.info('----- make dataloader complete. -----')
    # train test ect
    logger.info('----- do train start -----')
    train_returns = do_train_test_ect(
        args, make_get_model, pre_training, data_helper=data_helper, data_loaders=data_loaders, logger=logger)
    logger.info('----- do train complete -----')
    return {MODEL: train_returns[MODEL], DATA_HELPER: data_helper, DATASETS: datasets,
            DATA_LOADERS: data_loaders, TRAIN_RETURNS: train_returns}


def main(args=None):
    """main

    """
    from utils.setup import setup, save_param

    args, logger, device = setup(setup_parser, PROJECT_DIR, parser_args=args)
    version_check(torch, np, pd, h5py, optuna, logger=logger)
    torch_fix_seed(seed=args.SEED)
    del args.SEED
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
