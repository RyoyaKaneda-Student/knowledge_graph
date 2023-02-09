#!/usr/bin/python
# -*- coding: utf-8 -*-
"""run script for Knowledge Graph Challenge

* This script is the script about Knowledge Graph Challenge.
* Define data, define model, define parameters, and run train.

Todo:
    * 色々

"""
# ========== python ==========
from argparse import Namespace
from logging import Logger
from typing import Optional, Final, Sequence

import h5py
# Machine learning
import numpy as np
import optuna
import pandas as pd
# torch
import torch
# torch ignite
from torch.utils.data import Dataset

# My const words about words used as tags
# My const value about torch ignite
# My const words about words used as tags
from const.const_values import (MODEL, DATA_HELPER, DATASETS, DATA_LOADERS, TRAIN_RETURNS,
                                )
# My const words about file direction and title
from const.const_values import (
    PROJECT_DIR,
)
# My Models
from models.datasets.data_helper import (DefaultTokens, DefaultIds, )
from models.datasets.data_helper_for_wn18rr import MyDataHelperForWN18RR
from models.datasets.datasets_for_wn18rr import WN18RRDataset, WN18RRDatasetForValid
from models.utilLoss.utils import LossFnName
# My utils
from run_for_KGC import (
    ModelVersion, make_get_dataloader, do_train_test_ect, get_all_tokens, param_init_setting as param_init_setting_old)
from utils.torch import torch_fix_seed, DeviceName
from utils.utils import version_check

WN18RR_TRAIN_PATH = f"{PROJECT_DIR}/data/external/KGdata/WN18RR/text/train.txt"
WN18RR_VALID_PATH = f"{PROJECT_DIR}/data/external/KGdata/WN18RR/text/valid.txt"
WN18RR_TEST_PATH = f"{PROJECT_DIR}/data/external/KGdata/WN18RR/text/test.txt"


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
    paa51('--skip-head-mask', action='store_true', help='', )
    paa51('--skip-relation-mask', action='store_true', help='', )
    paa51('--skip-tail-mask', action='store_true', help='', )
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
    paa6('--lr-head', type=float, help='learning rate (default: same as --lr)')
    paa6('--lr-relation', type=float, help='learning rate (default: same as --lr)')
    paa6('--lr-tail', type=float, help='learning rate (default: same as --lr)')
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
    return args


def param_init_setting(args, *, logger):
    """param_init_setting

    """
    args = param_init_setting_old(args, logger=logger)
    return args


def make_get_data_helper(args: Namespace, *, logger: Logger):
    """make and get data_helper

    Args:
        args(Namespace): args
        logger(Logger): logging.Logger

    Returns:
        MyDataHelperForWN18RR: MyDataHelperForWN18RR()

    """
    ((pad_token_e, pad_token_r), (cls_token_e, cls_token_r),
     (mask_token_e, mask_token_r), (sep_token_e, sep_token_r), _) = get_all_tokens(args)

    entity_special_dicts = {
        pad_token_e: DefaultTokens.PAD_E, cls_token_e: DefaultTokens.CLS_E,
        mask_token_e: DefaultTokens.MASK_E, sep_token_e: DefaultTokens.SEP_E,
    }
    relation_special_dicts = {
        pad_token_r: DefaultTokens.PAD_R, cls_token_r: DefaultTokens.CLS_R,
        mask_token_r: DefaultTokens.MASK_R, sep_token_r: DefaultTokens.SEP_R,
    }
    max_len = args.max_len
    data_helper = MyDataHelperForWN18RR(
        WN18RR_TRAIN_PATH, WN18RR_VALID_PATH, WN18RR_TEST_PATH, max_len, logger=logger,
        entity_special_dicts=entity_special_dicts, relation_special_dicts=relation_special_dicts)
    return data_helper


def make_get_datasets(args: Namespace, *, data_helper: MyDataHelperForWN18RR, logger: Logger):
    """make and get datasets
    
    Args:
        args(Namespace): args
        data_helper(MyDataHelperForWN18RR): data_helper
        logger(Logger): logger

    Returns:
        tuple[WN18RRDataset, WN18RRDatasetForValid, WN18RRDatasetForValid]: dataset_train, dataset_valid, dataset_test

    """
    # get from args
    ((pad_token_e, pad_token_r), _, _, _, _) = get_all_tokens(args)
    entity_num, relation_num = data_helper.processed_entity_num, data_helper.processed_relation_num
    # make triple
    train_triple = data_helper.get_processed_triple(data_helper.data.train_df.values[:, :3].tolist())

    train_triple_sequence = data_helper.get_processed_train_sequence(pad_token_e, pad_token_r, 0)
    valid_triple_sequence = data_helper.get_processed_valid_sequence(pad_token_e, pad_token_r, 0)
    test_triple_sequence = data_helper.get_processed_test_sequence(pad_token_e, pad_token_r, 0)

    train_dataset = WN18RRDataset(train_triple, train_triple_sequence, entity_num, relation_num)
    valid_dataset = WN18RRDatasetForValid(None, valid_triple_sequence, entity_num, relation_num, 2)
    test_dataset = WN18RRDatasetForValid(None, test_triple_sequence, entity_num, relation_num, 3)
    logger.debug("----- max_len: {} -----".format(args.max_len))
    logger.debug("----- lens: train_dataset={}, valid_dataset={}, test_dataset={} ".format(
        *map(len, [train_dataset, valid_dataset, test_dataset]))
    )
    return train_dataset, valid_dataset, test_dataset


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
    # other args settings.
    args = param_init_setting(args, logger=logger)
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
    # train test ect
    logger.info('----- do train start -----')
    train_returns = do_train_test_ect(
        args, data_helper=data_helper, data_loaders=data_loaders, logger=logger)
    logger.info('----- do train complete -----')
    # return some value
    return {MODEL: train_returns[MODEL], DATA_HELPER: data_helper, DATASETS: datasets,
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
