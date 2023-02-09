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
from models.datasets.datasets_for_sequence import SimpleTriple
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

from run_for_KGC import make_get_data_helper, make_get_datasets, get_all_tokens, make_get_dataloader


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


def make_get_model(args: Namespace, *, data_helper: MyDataHelperForStory, logger: Logger):
    entity_embedding_dim, relation_embedding_dim = args.entity_embedding_dim, args.relation_embedding_dim
    entity_num, relation_num = data_helper.processed_entity_num, data_helper.processed_relation_num
    model = TransE(entity_embedding_dim, relation_embedding_dim, entity_num, relation_num)
    return model


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
    logger.info(datasets[0])
    data_loaders = make_get_dataloader(args, datasets=datasets, logger=logger)
    logger.info('----- make dataloader start. -----')
    data_loaders = make_get_dataloader(args, datasets=datasets, logger=logger)
    logger.info('----- make dataloader complete. -----')
    # train test ect
    logger.info('----- do train start -----')
    train_returns = do_train_test_ect(
        args, data_helper=data_helper, data_loaders=data_loaders, logger=logger)
    logger.info('----- do train complete -----')


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
