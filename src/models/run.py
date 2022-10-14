# coding: UTF-8
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(os.path.join(PROJECT_DIR, 'src'))

# ========== python ==========
from logging import Logger
# noinspection PyUnresolvedReferences
from typing import List, Dict, Tuple, Optional, Union, Callable
# noinspection PyUnresolvedReferences
from tqdm import tqdm
from argparse import Namespace
# Machine learning
import h5py
import numpy as np
import pandas as pd
import optuna
# torch
import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader

"""
# torch ignite
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.contrib.handlers.tensorboard_logger import *
"""

# Made by me
from utils.utils import force_gc, force_gc_after_function, get_from_dict, version_check
from utils.str_process import line_up_key_value, blank_or_NOT, info_str as _info_str
from utils.setup import setup, save_param, ChangeDisableNamespace
from utils.torch import (
    cuda_empty_cache as _cec, load_model, save_model, decorate_loader, param_count_check, random_indices_choice,
    force_cuda_empty_cache_after_function, LossHelper, torch_fix_seed)
from utils.textInOut import SQLITE_PREFIX
from utils.progress_manager import ProgressHelper

from model import ConvE, DistMult, Complex, TransformerE, TransformerVer2E, TransformerVer2E_ERE, TransformerVer3E, \
    KGE_ERE, KGE_ERTails
from models.datasets.data_helper import MyDataHelper, load_preprocess_data
from models.datasets.datasets import MyTripleTrainDataset

PROCESSED_DATA_PATH = './data/processed/'
EXTERNAL_DATA_PATH = './data/external/'

MODEL_TMP_PATH = 'saved_models/.tmp/check-point.{}.model'

CPU: str = 'cpu'
TRAIN: str = 'train'
TEST: str = 'test'
MRR: str = 'mrr'
HIT_: str = 'hit_'

KGDATA_ALL = ['FB15k-237', 'WN18RR', 'YAGO3-10']
name2model = {
    'conve': ConvE,
    'distmult': DistMult,
    'complex': Complex,
    'transformere1': TransformerE,
    'transformere2': TransformerVer2E,
    'transformere2_ere': TransformerVer2E_ERE,
    'transformere3': TransformerVer3E,
}


def setup_parser(args=None) -> Namespace:
    import argparse  # 1. argparseをインポート
    parser = argparse.ArgumentParser(description='データの初期化')
    parser.add_argument('--logfile', help='ログファイルのパス', type=str)
    parser.add_argument('--param-file', help='パラメータを保存するファイルのパス', type=str)
    parser.add_argument('--console-level', help='コンソール出力のレベル', type=str, default='info')
    parser.add_argument('--no-show-bar', help='バーを表示しない', action='store_true')
    parser.add_argument('--device-name', help='cpu or cuda or mps', type=str, default='cpu',
                        choices=['cpu', 'cuda', 'mps'])
    # select function
    parser.add_argument('--function', help='function', type=str, choices=['do_1train', 'do_optuna', 'do_test'])
    # optuna setting
    parser.add_argument('--optuna-file', help='optuna file', type=str)
    parser.add_argument('--study-name', help='optuna study-name', type=str)
    parser.add_argument('--n-trials', help='optuna n-trials', type=int, default=20)

    parser.add_argument('--KGdata', help=' or '.join(KGDATA_ALL), type=str,
                        choices=KGDATA_ALL)
    parser.add_argument('--train-type', help='', type=str, choices=['all_tail', 'triples'])
    parser.add_argument('--do-negative-sampling', help='', action='store_true')
    parser.add_argument('--negative-count', help='', type=int, default=None)
    parser.add_argument('--eco-memory', help='メモリに優しく', action='store_true')
    parser.add_argument('--entity-special-num', help='エンティティ', type=int, default=None)
    parser.add_argument('--relation-special-num', help='リレーション', type=int, default=None)
    # e special
    parser.add_argument('--padding-token-e', help='padding', type=int)
    parser.add_argument('--cls-token-e', help='cls', type=int)
    # r special
    parser.add_argument('--padding-token-r', help='padding', type=int)
    parser.add_argument('--cls-token-r', help='cls', type=int)
    parser.add_argument('--self-loop-token-r', help='self-loop', type=int)

    parser.add_argument('--model', type=str, help=f"Choose from: {', '.join(name2model.keys())}")
    parser.add_argument('--embedding-dim', type=int, default=200,
                        help='The embedding dimension (1D). Default: 200')
    parser.add_argument('--batch-size', help='batch size', type=int)
    parser.add_argument('--epoch', help='max epoch', type=int)

    parser.add_argument('--model-path', type=str, help='model path')
    parser.add_argument('--do-train', help='do-train', action='store_true')
    parser.add_argument('--do-valid', help='do-valid', action='store_true')
    parser.add_argument('--do-test', help='do-test', action='store_true')
    parser.add_argument('--do-debug', help='do-debug', action='store_true')
    parser.add_argument('--do-train-valid', help='do-train and valid', action='store_true')
    parser.add_argument('--do-train-test', help='do-train and test', action='store_true')
    parser.add_argument('--do-train-valid-test', help='do-train and valid and test', action='store_true')
    parser.add_argument('--valid-interval', type=int, default=1, help='valid-interval', )

    parser.add_argument('--embedding-shape1', type=int, default=20,
                        help='The first dimension of the reshaped 2D embedding. '
                             'The second dimension is inferred. Default: 20')
    # optimizer
    parser.add_argument('--lr', type=float, default=0.003, help='learning rate (default: 0.003)')
    parser.add_argument('--l2', type=float, default=0.0,
                        help='Weight decay value to use in the optimizer. Default: 0.0')
    parser.add_argument('--label-smoothing', type=float, default=0.1, help='Label smoothing value to use. Default: 0.1')
    # convE
    parser.add_argument('--hidden-drop', type=float, default=0.3, help='Dropout for the hidden layer. Default: 0.3.')
    parser.add_argument('--input-drop', type=float, default=0.2, help='Dropout for the input embeddings. Default: 0.2.')
    parser.add_argument('--feat-drop', type=float, default=0.2,
                        help='Dropout for the convolutional features. Default: 0.2.')
    parser.add_argument('--use-bias', action='store_true', help='Use a bias in the convolutional layer. Default: True')
    parser.add_argument('--lr-decay', type=float, default=0.995,
                        help='Decay the learning rate by this factor every epoch. Default: 0.995')
    parser.add_argument('--hidden-size', type=int, default=9728,
                        help='The side of the hidden layer. The required size changes with the size of the embeddings. '
                             'Default: 9728 (embedding size 200).')
    # transformere
    parser.add_argument('--nhead', type=int, default=8, help='nhead. Default: 8.')
    parser.add_argument('--transformer-drop', type=float, default=0.1, help='transformer-drop. Default: 0.1.')

    # コマンドライン引数をパースして対応するハンドラ関数を実行
    _args = parser.parse_args(args=args)
    if _args.do_train_valid_test:
        del _args.do_train_valid_test
        _args.do_train = True
        _args.do_valid = True
        _args.do_test = True

    return _args


def _select_model(args, model_name, e_length, r_length, **kwargs) -> Union[KGE_ERTails, KGE_ERE]:
    if model_name in name2model.keys():
        model: Union[KGE_ERTails, KGE_ERE] = name2model[model_name](args, e_length, r_length, **kwargs)
    else:
        raise Exception(f"Unknown model! :{model_name}")
    return model


def make_dataloader_all_tail(data_helper: MyDataHelper, batch_size, eco_memory, *, logger):
    train_dataset = data_helper.get_train_dataset(eco_memory=False, more_eco_memory=eco_memory)
    valid_dataset = data_helper.get_valid_dataset(more_eco_memory=eco_memory)
    test_dataset = data_helper.get_test_dataset(more_eco_memory=eco_memory)

    """
    for i in range(100):
        output, counts = torch.unique(valid_dataset[i][1], return_counts=True)
        logger.debug(f"{output=}, {counts=}")
        output, counts = torch.unique(valid_dataset[i][2], return_counts=True)
        logger.debug(f"{output=}, {counts=}")
        # logger.debug(f"{test_dataset[0]=}")
    """

    train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    data_helper.set_loaders(train, valid, test)


def make_dataloader_triple(data_helper: MyDataHelper, batch_size, eco_memory, is_do_negative_sampling, negative_count):
    train_triple: MyTripleTrainDataset = data_helper.get_train_triple_dataset(train_mode=True)
    valid_dataset = data_helper.get_valid_dataset(more_eco_memory=eco_memory)
    test_dataset = data_helper.get_valid_dataset(more_eco_memory=eco_memory)

    if is_do_negative_sampling:
        assert negative_count is not None
        special_num, _ = data_helper.get_er_special_num()
        train_data = data_helper.get_train_dataset(more_eco_memory=True)
        train_triple.add_negative(negative_count, train_data, special_num)

    train = DataLoader(train_triple, batch_size=batch_size, shuffle=True)
    valid = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    data_helper.set_loaders(train, valid, test)


def get_model(args, data_helper):
    model = _select_model(
        args, args.model, data_helper.get_final_e_length(), data_helper.get_final_r_length(), data_helper=data_helper)
    model.init()
    return model


@force_gc_after_function
def training(
        args: Namespace, *, logger,
        progress_helper: ProgressHelper,
        model, data_helper,
        lr,
        do_valid,
        no_show_bar=False,
        uid=None
):
    device = args.device
    max_epoch = args.epoch
    pid = args.pid
    checkpoint_path = MODEL_TMP_PATH.format(line_up_key_value(pid=pid, uid=uid))
    valid_interval = args.valid_interval if do_valid else -1
    lr = lr
    label_smoothing = args.label_smoothing
    # data
    train = data_helper.train_dataloader
    train_type = args.train_type

    #
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.l2)

    model.to(device)
    result = {
        'train_loss': [],
        'mrr': [],
        'hit_': [],
        'completed_epoch': -1
    }

    early_total_count = 200
    loss_helper = LossHelper(progress_helper=progress_helper, early_total_count=early_total_count)

    def append_to_result(_name, _value, *, _epoch=None):
        if not _epoch:
            result[_name].append(_value)
        else:
            result[_name][_epoch] = _value

    for epoch in progress_helper.progress(range(max_epoch), 'epoch', total=max_epoch):
        # train
        logger.info(f"{'-' * 10}epoch {epoch + 1} start. {'-' * 10}")
        model.train()
        loss = torch.tensor(0., requires_grad=False, device=device)
        sum_train = 0

        for idx, x in decorate_loader(train, no_show_bar=no_show_bar):
            opt.zero_grad()
            _loss: torch.Tensor
            if train_type == 'all_tail':
                er, e2s = [_x.to(device) for _x in x]
                output, counts = torch.unique(e2s, return_counts=True)
                # logger.debug(f"{output=}, {counts=}")
                e2s = ((1.0 - label_smoothing) * e2s) + (1.0 / e2s.size(1))
                e, r = er.split(1, 1)
                pred: torch.Tensor = model.forward((e, r))
                _loss = model.loss(pred, e2s)
                del e, r, er, pred, e2s
            elif train_type == 'triples':
                er, tail, is_exists = [_x.to(device) for _x in x]
                e, r = er.split(1, 1)
                pred: torch.Tensor = model.forward((e, r, tail))
                _loss = model.loss(pred, is_exists)
                del e, r, er, tail, pred
            else:
                raise "this is not supported."

            _loss.backward()
            opt.step()
            # loss check
            _loss = _loss.detach().sum()
            loss += _loss
            _cec()

        logger.debug(f"sum_train: {sum_train}")
        loss /= len(train)
        loss = loss.to(CPU).item()

        loss_helper.update(loss)

        append_to_result('train_loss', loss)
        logger.info("-----train result (epoch={}): loss = {}".format(epoch + 1, loss))
        logger.debug(f"{'-' * 5}epoch {epoch + 1} train end. {'-' * 5}")
        del loss
        _cec()
        # valid
        if do_valid and (epoch + 1) % valid_interval == 0:
            logger.debug(f"{'-' * 10}epoch {epoch + 1} valid start. {'-' * 5}")
            _, _result = testing(args, logger=logger, model=model, data_helper=data_helper, is_valid=True,
                                 no_show_bar=no_show_bar)
            mrr, hit_ = get_from_dict(_result, ('mrr', 'hit_'))
            append_to_result('mrr', mrr)
            append_to_result('hit_', hit_)
            logger.info("-----valid result (epoch={}): mrr = {}".format(epoch + 1, mrr))
            logger.info("-----valid result (epoch={}): hit = {}".format(epoch + 1, hit_))
            logger.debug(f"{'-' * 5}epoch {epoch + 1} valid end. {'-' * 5}")

        logger.info(f"{'-' * 10}epoch {epoch + 1} end.{'-' * 10}")
        save_model(model, checkpoint_path, device=device)
        result['completed_epoch'] = epoch + 1
        # early stopping
        if loss_helper.not_update_count >= early_total_count:
            logger.info(f"early stopping")
            break

    load_model(model, checkpoint_path, device=device, delete_file=True)
    return model, result


@force_gc_after_function
@torch.no_grad()
def testing(
        args: Namespace, *, logger,
        model, data_helper,
        is_valid=False, is_test=False,
        no_show_bar=False,
):
    model = model
    device = args.device
    train_type = args.train_type

    if train_type == 'triples':
        entity_special_num = args.entity_special_num
        pass

    if is_valid:
        logger.debug(f"{'-' * 5}This is valid{'-' * 5}")
        test = data_helper.valid_dataloader
    elif is_test:
        logger.debug(f"{'-' * 5}This is test{'-' * 5}")
        test = data_helper.test_dataloader
    else:
        raise "Either valid or test must be specified."
        pass

    len_test = 0
    zero_tensor = torch.tensor(0., dtype=torch.float32, device=device)

    # test
    model.to(device)
    model.eval()

    mrr = torch.tensor(0., device=device, dtype=torch.float32, requires_grad=False)
    hit_ = torch.tensor([0.] * 10, device=device, dtype=torch.float32, requires_grad=False)

    for idx, (er, e2s, e2s_all) in decorate_loader(test, no_show_bar=no_show_bar):
        e, r = er.to(device).split(1, 1)
        """
        output, counts = torch.unique(e2s, return_counts=True)
        logger.debug(f"{output=}, {counts=}")
        output, counts = torch.unique(e2s_all, return_counts=True)
        logger.debug(f"{output=}, {counts=}")
        """
        if train_type == 'all_tail':
            # model: KGE_ERTails
            pred: torch.Tensor = model.forward((e, r))
        elif train_type == 'triples':
            # model: KGE_ERE
            pred = torch.zeros_like(e2s)
            for i in tqdm(range(entity_special_num, e2s.shape[1]), leave=False):
                _pred = model.forward((e, r, torch.full((len(e), 1), i).to(device)))
                pred[:, i] = _pred[:, 0, ]
        del e, r, er
        # make filter
        e2s = e2s.to(device)
        row, column = torch.where(e2s == 1)
        del e2s
        e2s_all_binary: torch.Tensor = e2s_all != 0
        del e2s_all
        #
        pred = pred[row]  # 複製
        e2s_all_binary = e2s_all_binary[row]  # 複製
        # row is change
        len_row = len(row)
        row = [i for i in range(len_row)]
        #
        e2s_all_binary[row, column] = False
        pred[e2s_all_binary] = zero_tensor
        del e2s_all_binary
        #
        ranking = torch.argsort(pred, dim=1, descending=True)  # これは0 始まり
        del pred
        _cec()
        ranks = torch.argsort(ranking, dim=1)[row, column]
        del ranking
        _cec()
        ranks += 1
        # mrr and hit
        mrr += (1. / ranks).sum()
        for i in range(10):
            hit_[i] += torch.count_nonzero(ranks <= (i + 1))
        del ranks, row, column
        # after
        _cec()
        len_test += len_row

    logger.debug(f"{mrr=}, {hit_=}, {len_test=}")
    mrr = (mrr / len_test)
    hit_ = (hit_ / len_test)
    result = {
        'mrr': mrr.item(), 'hit_': hit_.tolist()
    }
    del mrr, hit_
    _cec()
    logger.debug("=====Test result: mrr = {}".format(result['mrr']))
    logger.debug("=====Test result: hit = {}".format(result['hit_']))
    return model, result


def train_setup(args, *, logger: Logger):
    # padding token is 0
    # cls token is 1
    # so special num is 1
    kg_data = args.KGdata
    eco_memory = args.eco_memory
    entity_special_num = args.entity_special_num
    relation_special_num = args.relation_special_num
    # padding_token = 0
    batch_size = args.batch_size
    train_type = args.train_type
    is_do_negative_sampling = args.do_negative_sampling
    negative_count = args.negative_count

    assert batch_size is not None
    assert train_type is not None

    # load data
    logger.info(_info_str(f"load data start."))
    data_helper = load_preprocess_data(kg_data, eco_memory, entity_special_num, relation_special_num, logger=logger)
    logger.debug(f"=====this is {blank_or_NOT(eco_memory)} eco_memory mode=====")
    logger.info(_info_str(f"load data complete."))

    # dataloader
    logger.info(_info_str(f"make dataloader start."))
    if train_type == 'all_tail':
        make_dataloader_all_tail(data_helper, batch_size, eco_memory, logger=logger)
    elif train_type == 'triples':
        assert is_do_negative_sampling is not None
        make_dataloader_triple(data_helper, batch_size, eco_memory, is_do_negative_sampling, negative_count)
    else:
        pass
    logger.info(_info_str(f"make dataloader complete."))

    # model
    logger.info(_info_str(f"make model start."))
    model = get_model(args, data_helper)
    logger.info(_info_str(f"make model complete."))
    logger.info(f"grad param count: {param_count_check(model)}")
    return data_helper, model


def do_1train(args: Namespace, *, logger: Logger, progress_helper: ProgressHelper, ):
    """
    Args:
        args: Namespace
        logger: Logger
        progress_helper: ProgressHelper

    Returns: None

    """
    assert type(args) is ChangeDisableNamespace
    is_do_train, is_do_valid, is_do_test = args.do_train, args.do_valid, args.do_test
    is_do_debug = args.do_debug
    model_path = args.model_path
    device = args.device
    no_show_bar = args.no_show_bar

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
        logger.info(_info_str(f"Train start."))

        model, result = training(
            args, logger=logger, progress_helper=progress_helper,
            data_helper=data_helper,
            model=model,
            lr=lr,
            do_valid=is_do_valid,
            no_show_bar=no_show_bar
        )
        save_model(model, args.model_path, device=device)
        # args.train_result = result
        logger.info(_info_str(f"Train complete."))
        del result

    model = load_model(model, model_path, device=device)

    if is_do_valid:
        logger.info(_info_str(f"Test valid start."))
        model, result = testing(
            args, logger=logger,
            data_helper=data_helper, model=model,
            is_valid=True,
            no_show_bar=no_show_bar
        )
        # args.test_valid_result = result
        logger.info(f"=====Test valid result. mrr: {result['mrr']}, hit_: {result['hit_']}")
        logger.info(_info_str(f"Test valid complete."))

    if is_do_test:
        logger.info(_info_str(f"Test start."))
        model, result = testing(
            args, logger=logger,
            data_helper=data_helper, model=model,
            is_test=True,
            no_show_bar=no_show_bar,
        )
        # args.test_result = result
        logger.info(f"=====Test result. mrr: {result['mrr']}, hit_: {result['hit_']}")
        logger.info(_info_str(f"Test complete."))

    logger.info(f"Function finish".center(40, '='))


def do_optuna(args, *, logger: Logger, progress_helper: ProgressHelper, ):
    assert type(args) is ChangeDisableNamespace
    batch_size = args.batch_size
    no_show_bar = args.no_show_bar

    study_name = args.study_name
    optuna_file = args.optuna_file
    n_trials = args.n_trials

    logger.info(f"Optuna start".center(40, '='))

    data_helper, _ = train_setup(args, logger=logger)
    progress_helper.add_key('study', total=n_trials)

    @progress_helper.update_progress_after_function('study')
    @force_cuda_empty_cache_after_function
    def objective(trial: optuna.Trial):
        nonlocal data_helper, progress_helper
        lr = trial.suggest_loguniform('lr', 1e-6, 1e-2)

        logger.info(_info_str(f"trial {trial.number} start."))
        model = get_model(args, data_helper)
        model.init()
        # train
        model, _ = training(
            args, logger=logger, progress_helper=progress_helper,
            data_helper=data_helper,
            model=model,
            lr=lr,
            do_valid=False,
            no_show_bar=no_show_bar,
            # optuna additional setting
            uid=trial.number
        )
        # valid
        _, result = testing(
            args, logger=logger,
            data_helper=data_helper, model=model,
            is_valid=True,
            no_show_bar=no_show_bar
        )
        logger.info(f"=====Valid result. mrr: {result['mrr']}, hit_: {result['hit_']}")
        return result['mrr']

    study = optuna.create_study(
        direction='maximize', study_name=study_name, storage=SQLITE_PREFIX + optuna_file, load_if_exists=True
    )
    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)

    logger.info(_info_str("optuna study finish"))
    logger.info(f"==========best param = lr: {study.best_params['lr']}")
    logger.info(_info_str("update arge param"))

    logger.info(f"Optuna finish".center(40, '='))
    return study.best_params


def select_function(args, *, logger: Logger, progress_helper: ProgressHelper):
    fname = args.function
    if fname == 'do_test':
        args.do_train = False
        args.do_valid = True
        args.do_test = True

    if fname not in ('do_1train', 'do_optuna', 'do_test'):
        raise "you should select function"
    elif fname == 'do_1train':
        with ChangeDisableNamespace(args) as const_args:
            do_1train(const_args, logger=logger, progress_helper=progress_helper)

    elif fname == 'do_optuna':
        with ChangeDisableNamespace(args) as const_args:
            best_params = do_optuna(const_args, logger=logger, progress_helper=progress_helper)
        args.lr = best_params['lr']
        force_gc()
        with ChangeDisableNamespace(args) as const_args:
            do_1train(const_args, logger=logger, progress_helper=progress_helper)


def main():
    torch_fix_seed(seed=42)
    args, logger, device = setup(setup_parser, PROJECT_DIR)
    version_check(torch, nn, np, pd, h5py, optuna, logger=logger)
    try:
        args.project_dir = PROJECT_DIR
        args.logger = logger
        args.device = device
        args.completed = {}
        logger.debug(vars(args))
        logger.debug(f"process id = {args.pid}")
        progress_helper = ProgressHelper("log/progress.{pid}.txt", pid=args.pid)
        select_function(args, logger=logger, progress_helper=progress_helper)
        progress_helper.finish(delete=True)
    finally:
        save_param(args)
        pass


if __name__ == '__main__':
    main()
