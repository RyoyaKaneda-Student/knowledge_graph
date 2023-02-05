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
from models.KGModel.kg_story_transformer import (
    KgStoryTransformer01, KgStoryTransformer02, KgStoryTransformer03, KgStoryTransformer03preInit,
    KgStoryTransformer00, KgStoryTransformer)
from models.datasets.data_helper import (
    MyDataHelper, DefaultTokens, DefaultIds, SpecialTokens01 as SpecialTokens, MyDataLoaderHelper, )
from models.datasets.data_helper_for_wn18rr import MyDataHelperForWN18RR
from models.datasets.datasets_for_story import add_bos, StoryTriple, StoryTripleForValid
from models.utilLoss.focal_loss import FocalLoss, GAMMA
from models.utilLoss.utils import LossFnName
# My utils
from run_for_KGC import ModelVersion
from utils.error import UnderDevelopmentError
from utils.torch import save_model, torch_fix_seed, DeviceName
from utils.typing import ConstMeta
from utils.utils import version_check
from utils.torch_ignite import (
    set_write_model_param_function, set_start_epoch_function, set_end_epoch_function,
    set_valid_function, training_with_ignite, set_early_stopping_function
)
# My const words about words used as tags
from models.KGModel.kg_story_transformer import (
    ALL_WEIGHT_LIST, HEAD_MASKED_LM, TAIL_MASKED_LM, RELATION_MASKED_LM, )
# My const value about torch ignite
from utils.torch_ignite import (TRAINER, EVALUATOR, GOOD_LOSS_CHECKPOINTE, LAST_CHECKPOINTE)
# My const words about words used as tags
from const.const_values import (CPU, MODEL, LOSS, PARAMS, LR,
                                DATA_HELPER, DATASETS, DATA_LOADERS, TRAIN_RETURNS,
                                STORY_LOSS, RELATION_LOSS, OBJECT_LOSS,
                                STORY_ACCURACY, RELATION_ACCURACY, ENTITY_ACCURACY,
                                STORY_ANS, RELATION_ANS, OBJECT_ANS,
                                STORY_PRED, RELATION_PRED, ENTITY_PRED,
                                PRE_TRAIN_SCALER_TAG_GETTER, PRE_VALID_SCALER_TAG_GETTER,
                                PRE_TRAIN_MODEL_WEIGHT_TAG_GETTER,
                                METRIC_NAMES)
# My const words about file direction and title
from const.const_values import (
    PROJECT_DIR,
    ALL_TITLE_LIST, SRO_ALL_TRAIN_FILE, SRO_ALL_INFO_FILE, TITLE2SRO_FILE090, TITLE2SRO_FILE075, ABOUT_KILL_WORDS,
    LR_STORY, LR_RELATION, LR_ENTITY, LOSS_FUNCTION, STUDY,
)

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
    paa('--old-data', type=int, default=0,
        help='If you use old data, please enter the number. Basically, do not put anything in.')
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
    # r special
    parser_group042 = parser.add_argument_group(
        'special (tail) embedding token setting', 'There are the setting of special relation embedding token setting.')
    paa42 = parser_group042.add_argument
    paa42('--padding-token-r', help='padding', type=int, default=DefaultIds.PAD_R_DEFAULT_ID)
    paa42('--cls-token-r', help='cls', type=int, default=DefaultIds.CLS_R_DEFAULT_ID)
    paa42('--mask-token-r', help='mask', type=int, default=DefaultIds.MASK_R_DEFAULT_ID)
    paa42('--sep-token-r', help='sep', type=int, default=DefaultIds.SEP_R_DEFAULT_ID)
    # story
    parser_group043 = parser.add_argument_group(
        'special (tail) embedding token setting', 'There are the setting of special (head) embedding token setting.')
    paa43 = parser_group043.add_argument
    paa43('--padding-token-s', help='padding', type=int, default=DefaultIds.PAD_E_DEFAULT_ID)
    paa43('--cls-token-s', help='cls', type=int, default=DefaultIds.CLS_E_DEFAULT_ID)
    paa43('--mask-token-s', help='mask', type=int, default=DefaultIds.MASK_E_DEFAULT_ID)
    paa43('--sep-token-s', help='sep', type=int, default=DefaultIds.SEP_E_DEFAULT_ID)
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
    paa6('--loss-function', type=str, default=LossFnName.CROSS_ENTROPY_LOSS, choices=LossFnName.ALL_LIST(),
         help='loss function (default: CrossEntropyLoss)')
    paa6('--loss-weight-story', type=float, default=1., help='loss-weight-story')
    paa6('--loss-weight-relation', type=float, default=1., help='loss-weight-relation')
    paa6('--loss-weight-entity', type=float, default=1., help='loss-weight-entity')
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
    return (
        (pad_token_e, pad_token_r), (cls_token_e, cls_token_r),
        (mask_token_e, mask_token_r), (sep_token_e, sep_token_r),
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
    ((pad_token_e, pad_token_r), (cls_token_e, cls_token_r), (mask_token_e, mask_token_r), (sep_token_e, sep_token_r)
     ) = get_all_tokens(args)

    entity_special_dicts = {
        pad_token_e: DefaultTokens.PAD_E, cls_token_e: DefaultTokens.CLS_E,
        mask_token_e: DefaultTokens.MASK_E, sep_token_e: DefaultTokens.SEP_E,
    }
    relation_special_dicts = {
        pad_token_r: DefaultTokens.PAD_R, cls_token_r: DefaultTokens.CLS_R,
        mask_token_r: DefaultTokens.MASK_R, sep_token_r: DefaultTokens.SEP_R,
    }
    data_helper = MyDataHelperForWN18RR(
        WN18RR_TRAIN_PATH, WN18RR_VALID_PATH, WN18RR_TEST_PATH, 32, logger=logger,
        entity_special_dicts=entity_special_dicts, relation_special_dicts=relation_special_dicts)
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
    ((pad_token_e, pad_token_r), _, _, (sep_token_e, sep_token_r)) = get_all_tokens(args)
    entity_special_num = args.entity_special_num
    max_len = args.max_len
    train_valid_test, only_train = args.train_valid_test, args.only_train
    # get from data_helper
    entities, relations = data_helper.processed_entities, data_helper.processed_relations
    triple = data_helper.processed_train_triple
    # Check the number of data before including special tokens, as you will need them when creating the validation data.
    len_of_default_triple = len(triple)
    # add bos token in triples
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
    logger.info(f"\n{model}")

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
        model(KgStoryTransformer): model
        logger(Logger): logger

    Returns:
        dict: Keys=(MODEL, TRAINER, EVALUATOR, CHECKPOINTER_LAST, CHECKPOINTER_GOOD_LOSS)

    """

    # Now we are ready to start except for the hyper parameters.
    def func(_hyper_params, _summary_writer):
        """Training and save checkpoint.

        """
        # setting path
        _model_path = args.model_path
        if _model_path is None: raise ValueError("model path must not None")
        # training.
        _train_returns = pre_training(
            args, _hyper_params, data_helper, data_loaders, model, summary_writer=_summary_writer, logger=logger)
        # check the output of the training.
        _good_checkpoint, _last_checkpoint = map(_train_returns.get, (GOOD_LOSS_CHECKPOINTE, LAST_CHECKPOINTE))
        _checkpoint = _last_checkpoint.last_checkpoint if args.only_train else _good_checkpoint.last_checkpoint
        Checkpoint.load_objects(to_load={MODEL: model}, checkpoint=_checkpoint)
        # re-save as cpu model
        save_model(model, _model_path, device=args.device)
        return _train_returns, _good_checkpoint, _last_checkpoint, _checkpoint

    # default mode
    if args.pre_train and not args.do_optuna:
        summary_writer = SummaryWriter(log_dir=args.tensorboard_dir) if args.tensorboard_dir is not None else None
        # setting hyper parameter
        hyper_params = (
            args.lr, args.lr_story or args.lr, args.lr_relation or args.lr, args.lr_entity or args.lr,
            args.loss_function, args.loss_weight_story, args.loss_weight_relation, args.loss_weight_story,
            {GAMMA: args.gamma}
        )
        train_returns, good_checkpoint, last_checkpoint, checkpoint_ = func(hyper_params, summary_writer)
        logger.info(f"good model path: {good_checkpoint.last_checkpoint}")
        logger.info(f"last model path: {last_checkpoint.last_checkpoint}")
        logger.info(f"load checkpoint path: {checkpoint_}")
        logger.info(f"save model path: {args.model_path}")
    # optuna mode
    elif args.pre_train and args.do_optuna:
        def optimizer(trial: optuna.Trial):
            """optuna optimize function
            Args:
                trial:

            Returns:

            """
            _summary_writer = SummaryWriter(
                log_dir=f"{args.tensorboard_dir}/{trial.number}") if args.tensorboard_dir is not None else None
            lr = trial.suggest_float(LR, 1e-6, 1e-4, log=True)
            lr_story = trial.suggest_float(LR_STORY, 1e-5, 1e-3, log=True)
            lr_relation = trial.suggest_float(LR_RELATION, 1e-6, 1e-4, log=True)
            lr_entity = trial.suggest_float(LR_ENTITY, 1e-6, 1e-4, log=True)
            loss_function = trial.suggest_categorical(LOSS_FUNCTION, LossFnName.ALL_LIST())
            gamma = trial.suggest_float(GAMMA, 1e-6, 5.0) if loss_function == LossFnName.FOCAL_LOSS else None
            _hyper_params = (lr, lr_story, lr_relation, lr_entity, loss_function,
                             args.loss_weight_story, args.loss_weight_relation, args.loss_weight_entity,
                             {GAMMA: gamma})
            # check the output of the training.
            _summary_writer.add_text(
                'info', "lr={}, lr_story={}, lr_relation={}, lr_entity={}, loss_function={}, gamma={}".format(
                    lr, lr_story, lr_relation, lr_entity, loss_function, gamma), )
            _train_returns, _, _, _ = func(_hyper_params, _summary_writer)
            _evaluator = _train_returns[EVALUATOR]
            _evaluator.run(data_loaders.valid_dataloader)
            return _evaluator.state.metrics[OBJECT_LOSS]

        logger.info("---------- Optuna ----------")
        logger.info(f"---- name: {args.study_name}, trial num: {args.n_trials}, save file: {args.optuna_file} ----")
        study = optuna.create_study(
            study_name=args.study_name, storage=f'sqlite:///{PROJECT_DIR}/{args.optuna_file}',
            load_if_exists=True, direction='minimize'
        )
        study.optimize(optimizer, args.n_trials, gc_after_trial=True)
        train_returns = {STUDY: study, }
    # if checking the trained items, use this mode.
    elif args.only_load_trainer_evaluator:
        hyper_params = (0., 0., 0., 0., LossFnName.CROSS_ENTROPY_LOSS, 1., 1., 1., {})
        train_returns = pre_training(
            args, hyper_params, data_helper, data_loaders, model, summary_writer=None, logger=logger)
    else:
        train_returns = {}
        pass
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