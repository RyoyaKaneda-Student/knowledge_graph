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
from typing import Optional, Callable, Final, cast, Sequence, Type

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
from ignite.metrics import Average, Accuracy, TopKCategoricalAccuracy
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
# My Models
from models.KGModel.kg_sequence_transformer import (
    KgSequenceTransformer01, KgSequenceTransformer02, KgSequenceTransformer03, KgSequenceTransformer03preInit,
    KgSequenceTransformer, ENTITY_EMBEDDINGS, RELATION_EMBEDDINGS)
from models.datasets.data_helper import (
    MyDataHelperForStory, DefaultTokens, DefaultIds, SpecialTokens01 as SpecialTokens, MyDataLoaderHelper, )
from models.datasets.datasets_for_sequence import add_bos, StoryTriple, StoryTripleForValid, SimpleTriple
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
                                HEAD_ANS, RELATION_ANS, TAIL_ANS,
                                HEAD_PRED, RELATION_PRED, TAIL_PRED,
                                PRE_TRAIN_SCALER_TAG_GETTER, PRE_VALID_SCALER_TAG_GETTER,
                                PRE_TRAIN_MODEL_WEIGHT_TAG_GETTER,
                                HEAD_METRIC_NAMES, RELATION_METRIC_NAMES, TAIL_METRIC_NAMES,
                                ACCURACY_NAME3, TOP1_NAME3, TOP3_NAME3,
                                TOP10_NAME3)
# My const words about file direction and title
from const.const_values import (
    PROJECT_DIR,
    ALL_TITLE_LIST, SRO_ALL_TRAIN_FILE, SRO_ALL_INFO_FILE, TITLE2SRO_FILE090, TITLE2SRO_FILE075, ABOUT_KILL_WORDS,
    LR_HEAD, LR_RELATION, LR_TAIL, LOSS_FUNCTION, STUDY,
)


class ModelVersion(metaclass=ConstMeta):
    """Model Versions

    * This is only const value class.

    """
    V01: Final = '01'
    V02: Final = '02'
    V03: Final = '03'
    V03a: Final = '03a'

    @classmethod
    def ALL_LIST(cls) -> tuple[str, ...]:
        """All list of this const values.

        Returns:
            tuple[str, ...]: (01, 02, 03, 03a).

        """
        return cls.V01, cls.V02, cls.V03, cls.V03a


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
    paa5('--force-set-label', action='store_true',
         help='If true, however there has no label, the entity name set as label.')
    paa5('--no-grad-entity-embedding', action='store_true', help='If true, entity embedding not change')
    paa5('--no-grad-relation-embedding', action='store_true', help='If true,  entity embedding not change')
    # mask percent
    parser_group051 = parser.add_argument_group(
        'model setting of mask-percent', 'MUST mask-mask + mask-random + mask-nomask == 1.00.')
    paa51 = parser_group051.add_argument
    paa51('--mask-percent', help='default: 0.15', metavar='mask-rate', type=float, default=0.15)
    paa51('--mask-mask-percent', help='default: 0.80', metavar='mask-rate', type=float, default=0.80)
    paa51('--mask-random-percent', help='default: 0.10', metavar='random-rate', type=float, default=0.10)
    paa51('--mask-nomask-percent', help='default: 0.10', metavar='nomask-rate', type=float, default=0.10)
    paa51('--no-bos-mask', action='store_true', help='', )
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

    # deprecated
    parser_group_deprecated = parser.add_argument_group('deprecated', 'These params ware deprecated.')
    paa_deprecated = parser_group_deprecated.add_argument
    paa_deprecated('--lr-story', type=float, help='learning rate (default: same as --lr). This is old parameter.')
    paa_deprecated('--lr-entity', type=float, help='learning rate (default: same as --lr). This is old parameter.')

    args = parser.parse_args(args=args)
    # old to new
    return fix_args(args)


def fix_args(args: Namespace):
    """fix args
    Args:
        args(Namespace):

    Returns:
        Namespace
    """
    if args.lr_story is not None:
        if args.lr_head is not None: raise ValueError()
        args.lr_head = args.lr_story
        warnings.warn("The parameter --lr-story is deprecated. please use --lr-head")
    if args.lr_entity is not None:
        if args.lr_tail is not None: raise ValueError()
        args.lr_tail = args.lr_entity
        warnings.warn("The parameter --lr-entity is deprecated. please use --lr-tail")
        del args.lr_entity

    for key in ('lr_story', 'lr_entity'):
        if hasattr(args, key): delattr(args, key)
        pass

    if not hasattr(args, 'old_data'):
        args.old_data = 0
        pass

    for key in ('skip_head_mask', 'skip_relation_mask', 'skip_tail_mask'):
        if not hasattr(args, key): setattr(args, key, False)
        pass
    return args


def param_init_setting(args: Namespace, *, logger: Logger):
    """args setting

    """
    if args.train_anyway:
        logger.warning("This process do not have reproducible.")
        torch.backends.cudnn.benchmark = True
        args.non_blocking = True
    else:
        torch.backends.cudnn.benchmark = False
        args.non_blocking = False
    if not args.do_optuna:
        for key in ('optuna_file', 'study_name', 'n_trials'):
            if hasattr(args, key): delattr(args, key)
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

    is_do_head_mask, is_do_relation_mask, is_do_tail_mask = (
        not args.skip_head_mask, not args.skip_relation_mask, not args.skip_tail_mask)

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

    # mask percents
    mask_percent = args.mask_percent
    mask_mask_percent = mask_percent * args.mask_mask_percent
    mask_nomask_percent = mask_percent * args.mask_nomask_percent
    mask_random_percent = mask_percent * args.mask_random_percent
    logger.debug(f"{mask_percent=}, {mask_mask_percent=}, {mask_nomask_percent=}, {mask_random_percent=}")
    if not mask_mask_percent + mask_nomask_percent + mask_random_percent + (1 - mask_percent) == 1.:
        raise ValueError(
            "mask_mask_percent + mask_nomask_percent + mask_random_percent + (1 - mask_percent) must be 1.0")

    # optional function
    # noinspection PyTypeChecker
    def mask_function(_random_all, _value, _mask_token, _bos_token, weights) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Mask by mask_token

        Args:
            _random_all(torch.Tensor): random values. All parameters are within 0.0 ~ 1.0.
            _value(torch.Tensor): Correct value.
            _mask_token(int): mask token
            _bos_token(int): bos token
            weights(torch.Tensor): the weight of random values frequency.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]

        """
        if args.no_bos_mask:
            _random_all[_value == _bos_token] = 1.0
            pass
        _mask_filter = _random_all < mask_percent
        _mask_ans = _value[_mask_filter].detach().clone()
        _mask_value = _value[_mask_filter]

        _random = _random_all[_mask_filter]
        _mask_random_filter = _random < mask_random_percent
        _mask_mask_filter = _random >= (mask_nomask_percent + mask_random_percent)

        _mask_random_filter_count = torch.count_nonzero(_mask_random_filter).item()
        _mask_mask_filter_count = torch.count_nonzero(_mask_mask_filter).item()

        if _mask_random_filter_count > 0:
            _mask_value[_mask_random_filter] = torch.multinomial(weights, _mask_random_filter_count, replacement=True)
        _mask_value[_mask_mask_filter] = _mask_token
        return _mask_filter, _mask_ans, _mask_value

    entity_num, relation_num = data_helper.processed_entity_num, data_helper.processed_relation_num
    train = data_loaders.train_dataloader
    train_dataset = train.dataset
    valid = data_loaders.valid_dataloader if args.train_valid_test else None
    # count frequency list
    head_index2count, relation_index2count, tail_index2count = train_dataset.get_index2count(device)

    # optimizer setting
    modules = {_name: _module for _name, _module in model.named_children()
               if _name not in (HEAD_MASKED_LM, RELATION_MASKED_LM, TAIL_MASKED_LM)}
    if args.no_grad_entity_embedding:
        logger.info("entity embeddings is not change.")
        modules.pop(ENTITY_EMBEDDINGS).requires_grad = False
    if args.no_grad_relation_embedding:
        logger.info("relation embeddings is not change.")
        modules.pop(RELATION_EMBEDDINGS).requires_grad = False
        pass

    opt = torch.optim.Adam([{PARAMS: _module.parameters(), LR: lr} for _name, _module in modules.items()
                            ] + [
                               {PARAMS: model.head_maskdlm.parameters(), LR: lr_head},
                               {PARAMS: model.relation_maskdlm.parameters(), LR: lr_relation},
                               {PARAMS: model.tail_maskdlm.parameters(), LR: lr_tail},
                           ])
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
        assert triple.shape == (batch_size, max_len, 3)
        # train start
        opt.zero_grad()

        triple: torch.Tensor = triple.to(device, non_blocking=non_blocking)

        mask_filter_head, mask_filter_relation, mask_filter_tail = None, None, None
        mask_ans_head, mask_ans_relation, mask_ans_tail = None, None, None
        if is_do_head_mask:
            mask_filter_head, mask_ans_head, mask_value_head = mask_function(
                torch.rand((batch_size, max_len)), triple[:, :, 0], mask_token_e, bos_token_e, head_index2count)
            triple[:, :, 0][mask_filter_head] = mask_value_head
        if is_do_relation_mask:
            mask_filter_relation, mask_ans_relation, mask_value_relation = mask_function(
                torch.rand((batch_size, max_len)), triple[:, :, 1], mask_token_r, bos_token_r, relation_index2count)
            triple[:, :, 1][mask_filter_relation] = mask_value_relation
        if is_do_tail_mask:
            mask_filter_tail, mask_ans_tail, mask_value_tail = mask_function(
                torch.rand((batch_size, max_len)), triple[:, :, 2], mask_token_e, bos_token_e, tail_index2count)
            triple[:, :, 2][mask_filter_tail] = mask_value_tail

        _, (head_pred, relation_pred, tail_pred) = model(
            triple, mask_filter_head, mask_filter_relation, mask_filter_tail
        )

        loss: torch.Tensor = torch.tensor(0, dtype=torch.float).to(device)
        head_loss, relation_loss, object_loss = None, None, None

        if is_do_head_mask and len(mask_ans_head) > 0:
            head_loss = loss_fn_entity(head_pred, mask_ans_head)
            loss += head_loss
        if is_do_relation_mask and len(mask_ans_relation) > 0:
            relation_loss = loss_fn_relation(relation_pred, mask_ans_relation)
            loss += relation_loss
        if is_do_tail_mask and len(mask_ans_tail) > 0:
            object_loss = loss_fn_entity(tail_pred, mask_ans_tail)
            loss += object_loss

        loss.backward()
        opt.step()

        # return values
        return_dict = {
            HEAD_ANS: cpu_deep_copy_or_none(mask_ans_head),
            RELATION_ANS: cpu_deep_copy_or_none(mask_ans_relation),
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


def make_get_data_helper(args: Namespace, *, logger: Logger):
    """make and get data_helper

    Args:
        args(Namespace): args
        logger(Logger): logging.Logger

    Returns:
        MyDataHelperForStory: MyDataHelperForStory()

    """
    use_title = args.use_title
    ((pad_token_e, pad_token_r), (cls_token_e, cls_token_r), (mask_token_e, mask_token_r),
     (sep_token_e, sep_token_r), (bos_token_e, bos_token_r)) = get_all_tokens(args)
    is_090, is_075 = args.use_for_challenge090, args.use_for_challenge075
    if is_090 or is_075:
        if not args.only_train: raise ValueError("If use for challenge, --only-train must True")
        if args.use_title is None: raise ValueError("--use-title must not None.")
    if is_090:
        train_file = TITLE2SRO_FILE090[use_title]
    elif is_075:
        train_file = TITLE2SRO_FILE075[use_title]
    else:
        train_file = SRO_ALL_TRAIN_FILE
    info_file = SRO_ALL_INFO_FILE

    if getattr(args, 'old_data', None):
        if args.old_data == 1:
            logger.info("----- use old data (version 1) -----")
            train_file = train_file.replace('data', 'data.tmp1', 1)
            info_file = info_file.replace('data', 'data.tmp1', 1)
        if args.old_data == 2:
            logger.info("----- use old data (version 2) -----")
            train_file = train_file.replace('data', 'data.tmp2', 1)
            info_file = info_file.replace('data', 'data.tmp2', 1)
    entity_special_dicts = {
        pad_token_e: DefaultTokens.PAD_E, cls_token_e: DefaultTokens.CLS_E, mask_token_e: DefaultTokens.MASK_E,
        sep_token_e: DefaultTokens.SEP_E, bos_token_e: DefaultTokens.BOS_E
    }
    relation_special_dicts = {
        pad_token_r: DefaultTokens.PAD_R, cls_token_r: DefaultTokens.CLS_R, mask_token_r: DefaultTokens.MASK_R,
        sep_token_r: DefaultTokens.SEP_R, bos_token_r: DefaultTokens.BOS_R
    }
    data_helper = MyDataHelperForStory(
        info_file, train_file, None, None, logger=logger,
        entity_special_dicts=entity_special_dicts, relation_special_dicts=relation_special_dicts)
    data_helper.show(logger)
    return data_helper


def make_get_datasets(args: Namespace, *, data_helper, logger: Logger):
    """make and get datasets
    
    Args:
        args(Namespace): args
        data_helper(MyDataHelper): data_helper
        logger(Logger): logger

    Returns:
        tuple[Dataset, Dataset, Dataset]: [dataset_train, dataset_valid, dataset_test]

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

    # region set type
    triple_train: np.ndarray
    triple_valid: np.ndarray
    triple_test: np.ndarray
    valid_filter: np.ndarray
    test_filter: np.ndarray
    # endregion

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
            pass
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
    elif only_train:
        triple_train = triple
        triple_valid = triple
        triple_test = triple
        valid_filter = np.zeros(len(triple), dtype=bool)
        test_filter = np.zeros(len(triple), dtype=bool)
    else:
        raise ValueError("Either --train-valid-test or --only-train is required.")
        pass

    if max_len > 1:
        logger.info("----- Sequence triple data. -----")
        entity_num, relation_num = data_helper.processed_entity_num, data_helper.processed_relation_num
        dataset_train = StoryTriple(
            triple_train, np.where(triple_train[:, 0] == bos_token_e)[0], max_len,
            pad_token_e, pad_token_r, pad_token_e, sep_token_e, sep_token_r, sep_token_e,
            entity_num, relation_num
        )
        dataset_valid = StoryTripleForValid(
            triple_valid, np.where(triple_valid[:, 0] == bos_token_e)[0], valid_filter, max_len,
            pad_token_e, pad_token_r, pad_token_e, sep_token_e, sep_token_r, sep_token_e,
            entity_num, relation_num
        )
        dataset_test = StoryTripleForValid(
            triple_test, np.where(triple_test[:, 0] == bos_token_e)[0], test_filter, max_len,
            pad_token_e, pad_token_r, pad_token_e, sep_token_e, sep_token_r, sep_token_e,
            entity_num, relation_num
        )
    elif max_len == 1:
        logger.info("----- Simple triple data. -----")
        dataset_train = SimpleTriple(triple_train[triple_train[:, 0] != bos_token_e])
        dataset_valid = SimpleTriple(triple_valid[valid_filter])  # The valid data has no bos_token.
        dataset_test = SimpleTriple(triple_test[test_filter])  # The test data has no bos_token.
    else:
        raise ValueError()
        pass

    return dataset_train, dataset_valid, dataset_test


def make_get_model(args: Namespace, *, data_helper, logger: Logger):
    """make and get model

    Args:
        args(Namespace): args
        data_helper(MyDataHelper): data_helper
        logger(Logger): logger

    Returns:
        KgSequenceTransformer: KgStoryTransformer

    """
    # get from args
    model_dict = {
        ModelVersion.V01: KgSequenceTransformer01,
        ModelVersion.V02: KgSequenceTransformer02,
        ModelVersion.V03: KgSequenceTransformer03,
        ModelVersion.V03a: KgSequenceTransformer03preInit
    }

    all_tokens = [_t for _t in chain.from_iterable(get_all_tokens(args))]
    # get from data_helper
    entity_num, relation_num = data_helper.processed_entity_num, data_helper.processed_relation_num
    version_ = args.model_version
    logger.debug("----- make_model start -----")
    Model_: Type[KgSequenceTransformer] = model_dict[version_]

    if Model_ is None: raise f"model-version '{version_}' is not defined."

    model = Model_(args, entity_num, relation_num, special_tokens=SpecialTokens(*all_tokens))
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
    if args.train_valid_test:
        dataloader_valid = DataLoader(
            valid_dataset, shuffle=False, batch_size=batch_size * 2, num_workers=2, pin_memory=True)
        dataloader_test = DataLoader(
            test_dataset, shuffle=False, batch_size=batch_size * 2, num_workers=2, pin_memory=True)
    elif args.only_train:
        dataloader_valid, dataloader_test = None, None
        pass
    else:
        raise ValueError()
        pass

    data_loaders = MyDataLoaderHelper(datasets, dataloader_train, None, dataloader_valid, dataloader_test)
    logger.debug(f"{dataloader_train=}, {dataloader_valid=}, {dataloader_test=}")
    return data_loaders


def do_train_test_ect(args: Namespace, *, data_helper, data_loaders, logger: Logger):
    """do train test ect

    Args:
        args(Namespace): args
        data_helper(MyDataHelper): data_helper
        data_loaders(MyDataLoaderHelper): data_loaders
        logger(Logger): logger

    Returns:
        dict: Keys=(MODEL, TRAINER, EVALUATOR, CHECKPOINTER_LAST, CHECKPOINTER_GOOD_LOSS)

    """
    model = None

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
        model = make_get_model(args, data_helper=data_helper, logger=logger)
        summary_writer = SummaryWriter(log_dir=args.tensorboard_dir) if args.tensorboard_dir is not None else None
        # setting hyper parameter
        hyper_params = (
            args.lr, args.lr_head or args.lr, args.lr_relation or args.lr, args.lr_tail or args.lr,
            args.loss_function, {GAMMA: args.gamma}
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
            nonlocal model
            model = make_get_model(args, data_helper=data_helper, logger=logger)
            _summary_writer = SummaryWriter(
                log_dir=f"{args.tensorboard_dir}/{trial.number}") if args.tensorboard_dir is not None else None
            lr = trial.suggest_float(LR, 1e-6, 1e-4, log=True)
            lr_head = trial.suggest_float(LR_HEAD, 1e-5, 1e-3, log=True)
            lr_relation = trial.suggest_float(LR_RELATION, 1e-6, 1e-4, log=True)
            lr_tail = trial.suggest_float(LR_TAIL, 1e-6, 1e-4, log=True)
            loss_function = trial.suggest_categorical(LOSS_FUNCTION, LossFnName.ALL_LIST())
            gamma = trial.suggest_float(GAMMA, 1e-6, 5.0) if loss_function == LossFnName.FOCAL_LOSS else None
            _hyper_params = (lr, lr_head, lr_relation, lr_tail, loss_function, {GAMMA: gamma})
            # check the output of the training.
            _summary_writer.add_text(
                'info', "lr={}, lr_head={}, lr_relation={}, lr_tail={}, loss_function={}, gamma={}".format(
                    lr, lr_head, lr_relation, lr_tail, loss_function, gamma), )
            _train_returns, _, _, _ = func(_hyper_params, _summary_writer)

            _evaluator = _train_returns[EVALUATOR]
            _evaluator.run(data_loaders.valid_dataloader)
            return _evaluator.state.metrics[TAIL_LOSS]

        logger.info("---------- Optuna ----------")
        logger.info(f"---- name: {args.study_name}, trial num: {args.n_trials}, save file: {args.optuna_file} ----")
        study = optuna.create_study(
            study_name=args.study_name, storage=f'sqlite:///{PROJECT_DIR}/{args.optuna_file}',
            load_if_exists=True, direction='minimize'
        )
        study.optimize(optimizer, args.n_trials, gc_after_trial=True)
        train_returns = {STUDY: study, MODEL: None}
    # if checking the trained items, use this mode.
    elif args.only_load_trainer_evaluator:
        model = make_get_model(args, data_helper=data_helper, logger=logger)
        hyper_params = (0., 0., 0., 0., LossFnName.CROSS_ENTROPY_LOSS, {})
        train_returns = pre_training(
            args, hyper_params, data_helper, data_loaders, model, summary_writer=None, logger=logger)
    else:
        train_returns = {MODEL: None}
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


def main(args: Optional[Sequence[str]] = None):
    """main

    * Set Seed, set parser, save parser parameter and do main_function.
    * This function itself do nothing. Only call main_function.
    * If some error in main_function, this function saves the parameters for the moment and exits.

    Args:
        args(:obj:`Sequence[str]`, optional): args list or None. Default to None.

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
    pass
