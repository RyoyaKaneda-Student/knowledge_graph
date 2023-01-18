#!/usr/bin/python
# -*- coding: utf-8 -*-
"""run script for Knowledge Graph Embedding.

* This script is the script about Knowledge Graph Challenge.
* Define data, define model, define parameters, and run train.

Todo:
    * 色々

"""
from argparse import Namespace
# ========== python ==========
from itertools import chain
from logging import Logger
from pathlib import Path
# noinspection PyUnresolvedReferences
from typing import List, Dict, Tuple, Optional, Union, Callable, Final, Literal, get_args, cast, Sequence, TypedDict

# Machine learning
import h5py
import numpy as np
import optuna
import pandas as pd
# torch
import torch
# torch ignite
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

# My Models
from models.KGModel.kg_story_transformer import (
    KgStoryTransformer, )
from models.datasets.data_helper import (
    MyDataHelper, DefaultTokens, SpecialTokens01 as SpecialTokens, MyDataLoaderHelper, )
# My utils
from utils.torch import torch_fix_seed, DeviceName
from utils.utils import version_check

PROJECT_DIR = Path(__file__).resolve().parents[1]

# About words used as tags
CPU: Final[str] = 'cpu'
TRAIN: Final[str] = 'train'
PRE_TRAIN: Final[str] = 'pre_train'
TEST: Final[str] = 'test'
MRR: Final[str] = 'mrr'
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
DATASETS: Final[str] = 'datasets'
TRAIN_RETURNS: Final[str] = 'train_returns'
# about loss tags
LOSS: Final[str] = 'loss'
STORY_LOSS: Final[str] = 'story_loss'
RELATION_LOSS: Final[str] = 'relation_loss'
OBJECT_LOSS: Final[str] = 'entity_loss'
LOSS_NAME3: Final[tuple[str, str, str]] = (STORY_LOSS, RELATION_LOSS, OBJECT_LOSS)
# about pred tags
STORY_PRED: Final[str] = 'story_pred'
RELATION_PRED: Final[str] = 'relation_pred'
ENTITY_PRED: Final[str] = 'entity_pred'
PRED_NAME3: Final[tuple[str, str, str]] = (STORY_PRED, RELATION_PRED, ENTITY_PRED)
# about answer tags
STORY_ANS: Final[str] = 'story_ans'
RELATION_ANS: Final[str] = 'relation_ans'
OBJECT_ANS: Final[str] = 'object_ans'
ANS_NAME3: Final[tuple[str, str, str]] = (STORY_ANS, RELATION_ANS, OBJECT_ANS)
# about accuracy tags
STORY_ACCURACY: Final[str] = 'story_accuracy'
RELATION_ACCURACY: Final[str] = 'relation_accuracy'
ENTITY_ACCURACY: Final[str] = 'entity_accuracy'
ACCURACY_NAME3: Final[tuple[str, str, str]] = (STORY_ACCURACY, RELATION_ACCURACY, ENTITY_ACCURACY)
# about metric tags
METRIC_NAMES: Final[tuple[str, str, str, str, str, str, str]] = (
    LOSS, STORY_LOSS, RELATION_LOSS, OBJECT_LOSS, STORY_ACCURACY, RELATION_ACCURACY, ENTITY_ACCURACY)

from const.const_values import (
    SVO_ALL_TRAIN_FILE, SVO_ALL_INFO_FILE
)


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

    args = parser.parse_args(args=args)
    return args


def get_all_tokens(args: Namespace):
    """get all_tokens

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
    """make and get data_helper

    Args:
        args(Namespace): args
        logger(Logger): logging.Logger

    Returns:
        MyDataHelper: MyDataHelper()

    """
    use_title = args.use_title
    ((pad_token_e, pad_token_r), (cls_token_e, cls_token_r), (mask_token_e, mask_token_r),
     (sep_token_e, sep_token_r), (bos_token_e, bos_token_r)) = get_all_tokens(args)
    is_090, is_075 = args.use_for_challenge090, args.use_for_challenge075
    if is_090 or is_075:
        if not args.only_train: raise ValueError("If use for challenge, --only-train must True")
        if args.use_title is None: raise ValueError("--use-title must not None.")

    train_file = SVO_ALL_TRAIN_FILE
    entity_special_dicts = {
        pad_token_e: DefaultTokens.PAD_E, cls_token_e: DefaultTokens.CLS_E, mask_token_e: DefaultTokens.MASK_E,
        sep_token_e: DefaultTokens.SEP_E, bos_token_e: DefaultTokens.BOS_E
    }
    relation_special_dicts = {
        pad_token_r: DefaultTokens.PAD_R, cls_token_r: DefaultTokens.CLS_R, mask_token_r: DefaultTokens.MASK_R,
        sep_token_r: DefaultTokens.SEP_R, bos_token_r: DefaultTokens.BOS_R
    }
    data_helper = MyDataHelper(SVO_ALL_INFO_FILE, train_file, None, None, logger=logger,
                               entity_special_dicts=entity_special_dicts, relation_special_dicts=relation_special_dicts)
    data_helper.show(logger)
    return data_helper


def make_get_datasets(args: Namespace, *, data_helper: MyDataHelper, logger: Logger):
    """make and get datasets
    
    Args:
        args(Namespace): args
        data_helper(MyDataHelper): data_helper
        logger(Logger): logger

    Returns:
        tuple[Dataset, Dataset, Dataset]: dataset_train, dataset_valid, dataset_test

    """
    # get from args
    ((pad_token_e, pad_token_r), _, _, (sep_token_e, sep_token_r), (bos_token_e, bos_token_r)) = get_all_tokens(args)
    entity_special_num = args.entity_special_num
    max_len = args.max_len
    train_valid_test, only_train = args.train_valid_test, args.only_train
    # get from data_helper
    entities, relations = data_helper.processed_entities, data_helper.processed_relations
    triple = data_helper.processed_train_triple


def make_get_model(args: Namespace, *, data_helper: MyDataHelper, logger: Logger):
    """make and get model

    Args:
        args(Namespace): args
        data_helper(MyDataHelper): data_helper
        logger(Logger): logger

    Returns:
        KgStoryTransformer: KgStoryTransformer

    """
    # get from args
    all_tokens = [_t for _t in chain.from_iterable(get_all_tokens(args))]
    # get from data_helper
    num_entities, num_relations = len(data_helper.processed_entities), len(data_helper.processed_relations)
    logger.debug("----- make_model start -----")

    Model_ = None
    model = Model_(args, num_entities, num_relations, special_tokens=SpecialTokens(*all_tokens))
    model.assert_check()
    model.init(args, data_helper=data_helper)
    logger.info(model)

    return model


def make_get_dataloader(args: Namespace, *, datasets: tuple[Dataset, Dataset, Dataset], logger: Logger):
    """make and get dataloader

    Args:
        args(Namespace): use "batch_size" in args.
        datasets(tuple[Dataset, Dataset, Dataset]): dataset_train, dataset_valid and dataset_test
        logger(Logger): logging.Logger

    Returns:
        MyDataLoaderHelper: the dataclasses which has dataloader.

    """
    batch_size = args.batch_size
    train_dataset, valid_dataset, test_dataset = datasets
    dataloader_train = None if train_dataset is None else DataLoader(
        train_dataset, shuffle=True, batch_size=batch_size, num_workers=2, pin_memory=True)
    dataloader_valid = None if valid_dataset is None else DataLoader(
        valid_dataset, shuffle=False, batch_size=batch_size * 2, num_workers=2, pin_memory=True)
    dataloader_test = None if test_dataset is None else DataLoader(
        test_dataset, shuffle=False, batch_size=batch_size * 2, num_workers=2, pin_memory=True)
    data_loaders = MyDataLoaderHelper(datasets, dataloader_train, None, dataloader_valid, dataloader_test)
    logger.debug(f"{dataloader_train=}, {dataloader_valid=}, {dataloader_test=}")
    return data_loaders


def do_train_test_ect(args: Namespace, *, data_helper, data_loaders, model, logger: Logger):
    """do train test ect

    Args:
        args(Namespace): args
        data_helper(MyDataHelper): data_helper
        data_loaders(MyDataLoaderHelper): data_loaders
        model(nn.Module): model
        logger(Logger): logger

    Returns:
        dict: Keys=(MODEL, TRAINER, EVALUATOR, CHECKPOINTER_LAST, CHECKPOINTER_GOOD_LOSS)

    """
    # Now we are ready to start except for the hyper parameters.
    summary_writer = SummaryWriter(log_dir=args.tensorboard_dir) if args.tensorboard_dir is not None else None
    train_returns = {MODEL: None, TRAINER: None, EVALUATOR: None, CHECKPOINTER_LAST: None, CHECKPOINTER_GOOD_LOSS: None}

    return train_returns


def main_function(args: Namespace, *, logger: Logger):
    """main function

    * First, load data and make datahelper
    * Second, make datasets and dataloader.
    * Third, make model by setting hyper parameters.
    * Forth, Do train. if only load trained model or data, do nothing.
    * Finally, return data and trained model.

    Args:
        args(Namespace): args
        logger(Logger): logging.Logger

    Returns:
        dict: keys=(MODEL, DATA_HELPER, DATASETS, DATA_LOADERS, TRAIN_RETURNS)

    """
    raise NotImplementedError("Todo")
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
    train_returns = do_train_test_ect(
        args, data_helper=data_helper, data_loaders=data_loaders, model=model, logger=logger)
    logger.info('----- do train complete -----')
    # return some value
    return {MODEL: model, DATA_HELPER: data_helper, DATASETS: datasets,
            DATA_LOADERS: data_loaders, TRAIN_RETURNS: train_returns}


SEED: Final = 42


def main(args: Optional[Sequence[str]] = None):
    """main

    * Set Seed, set parser, save parser parameter and do main_function.
    * This function itself do nothing. Only call main_function.
    * If some error in main_function, this function saves the parameters for the moment and exits.

    Args:
        args(:obj:`Sequence[str]`, optional): args list or None. Default to None.

    """
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
