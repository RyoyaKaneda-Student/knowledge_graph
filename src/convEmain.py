import json
import torch
import torch.nn as nn
import pickle
import numpy as np
import argparse
import sys
import os
import math
from logging import Logger

from os.path import join
import torch.backends.cudnn as cudnn

import optuna
from optuna import Trial

from evaluation import ranking_and_hits
from model import ConvE, DistMult, Complex, TransformerE

from spodernet.preprocessing.pipeline import Pipeline, DatasetStreamer
from spodernet.preprocessing.processors import JsonLoaderProcessors, Tokenizer, AddToVocab, SaveLengthsToState, \
    StreamToHDF5, SaveMaxLengthsToState, CustomTokenizer
from spodernet.preprocessing.processors import ConvertTokenToIdx, ApplyFunction, ToLower, DictKey2ListMapper, \
    ApplyFunction, StreamToBatch
from spodernet.utils.global_config import Config, Backends
from spodernet.preprocessing.batching import StreamBatcher
from spodernet.preprocessing.pipeline import Pipeline
from spodernet.preprocessing.processors import TargetIdx2MultiTarget
from spodernet.hooks import LossHook, ETAHook
from spodernet.utils.util import Timer
from spodernet.preprocessing.processors import TargetIdx2MultiTarget
import argparse

np.set_printoptions(precision=3)

cudnn.benchmark = False

''' Preprocess knowledge graph using spodernet. '''


def setup_logger(name, logfile, console_level=None) -> Logger:
    import logging
    name = name
    console_level = logging.DEBUG if console_level == 'debug' else logging.INFO

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # create file handler which logs even DEBUG messages
    fh = logging.FileHandler(logfile)
    fh.setLevel(logging.DEBUG)
    fh_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(filename)s - %(name)s - %(funcName)s - %(message)s')
    fh.setFormatter(fh_formatter)

    # create console handler with a INFO log level
    ch = logging.StreamHandler()
    ch.setLevel(console_level)
    ch_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', '%Y-%m-%d %H:%M:%S')
    ch.setFormatter(ch_formatter)

    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.propagate = False

    logger.info("logger set complete. ")
    return logger


def preprocess(dataset_name, delete_data=False):
    full_path = 'data/KGdata/{0}/e1rel_to_e2_full.json'.format(dataset_name)
    train_path = 'data/KGdata/{0}/e1rel_to_e2_train.json'.format(dataset_name)
    dev_ranking_path = 'data/KGdata/{0}/e1rel_to_e2_ranking_dev.json'.format(dataset_name)
    test_ranking_path = 'data/KGdata/{0}/e1rel_to_e2_ranking_test.json'.format(dataset_name)

    keys2keys = {}
    keys2keys['e1'] = 'e1'  # entities
    keys2keys['rel'] = 'rel'  # relations
    keys2keys['rel_eval'] = 'rel'  # relations
    keys2keys['e2'] = 'e1'  # entities
    keys2keys['e2_multi1'] = 'e1'  # entity
    keys2keys['e2_multi2'] = 'e1'  # entity
    input_keys = ['e1', 'rel', 'rel_eval', 'e2', 'e2_multi1', 'e2_multi2']
    d = DatasetStreamer(input_keys)
    d.add_stream_processor(JsonLoaderProcessors())
    d.add_stream_processor(DictKey2ListMapper(input_keys))

    # process full vocabulary and save it to disk
    d.set_path(full_path)
    p = Pipeline(dataset_name, delete_data, keys=input_keys, skip_transformation=True)
    p.add_sent_processor(ToLower())
    p.add_sent_processor(CustomTokenizer(lambda x: x.split(' ')), keys=['e2_multi1', 'e2_multi2'])
    p.add_token_processor(AddToVocab())
    p.execute(d)
    p.save_vocabs()

    # process train, dev and test sets and save them to hdf5
    p.skip_transformation = False
    for path, name in zip([train_path, dev_ranking_path, test_ranking_path], ['train', 'dev_ranking', 'test_ranking']):
        d.set_path(path)
        p.clear_processors()
        p.add_sent_processor(ToLower())
        p.add_sent_processor(CustomTokenizer(lambda x: x.split(' ')), keys=['e2_multi1', 'e2_multi2'])
        p.add_post_processor(ConvertTokenToIdx(keys2keys=keys2keys),
                             keys=['e1', 'rel', 'rel_eval', 'e2', 'e2_multi1', 'e2_multi2'])
        p.add_post_processor(StreamToHDF5(name, samples_per_file=1000, keys=input_keys))
        p.execute(d)


def select_model(args, vocab, *, logger, ) -> nn.Module:
    model_name = args.model
    if model_name is None:
        model = ConvE(args, vocab['e1'].num_token, vocab['rel'].num_token)
    elif model_name == 'conve':
        model = ConvE(args, vocab['e1'].num_token, vocab['rel'].num_token)
    elif model_name == 'distmult':
        model = DistMult(args, vocab['e1'].num_token, vocab['rel'].num_token)
    elif model_name == 'complex':
        model = Complex(args, vocab['e1'].num_token, vocab['rel'].num_token)
    elif model_name == 'transformere':
        model = TransformerE(args, vocab['e1'].num_token, vocab['rel'].num_token)
    else:
        raise Exception(f"Unknown model! :{model_name}")
        pass
    return model


def train(
        args, vocab, num_entities, input_keys, device, *, logger,
        model_path, is_optuna=False,
):
    train_batcher = StreamBatcher(args.KGdata, 'train', args.batch_size, randomize=False, keys=input_keys,
                                  loader_threads=args.loader_threads)
    dev_rank_batcher = StreamBatcher(args.KGdata, 'dev_ranking', args.test_batch_size, randomize=False,
                                     loader_threads=args.loader_threads, keys=input_keys)
    test_rank_batcher = StreamBatcher(args.KGdata, 'test_ranking', args.test_batch_size, randomize=False,
                                      loader_threads=args.loader_threads, keys=input_keys)

    model = select_model(args, vocab, logger=logger)
    model_path_tmp = f"{model_path}.tmp"
    train_batcher.at_batch_prepared_observers.insert(
        1, TargetIdx2MultiTarget(num_entities, 'e2_multi1', 'e2_multi1_binary')
    )

    eta = ETAHook('train', print_every_x_batches=args.log_interval, log=logger)
    train_batcher.subscribe_to_events(eta)
    train_batcher.subscribe_to_start_of_epoch_event(eta)
    train_batcher.subscribe_to_events(LossHook('train', print_every_x_batches=args.log_interval, log=logger))

    model.to(device)

    logger.info("training ready")

    def save_model_func(_model_path):
        torch.save(model.state_dict(), _model_path)

    def load_model_func(_model_path):
        model_params = torch.load(_model_path)
        # print(model)
        # total_param_size = []
        # params = [(key, value.size(), value.numel()) for key, value in model_params.items()]
        # for key, size, count in params:
        #     total_param_size.append(count)
        #     print(key, size, count)
        # print(np.array(total_param_size).sum())
        model.load_state_dict(model_params)
        return model

    def valid_func():
        return ranking_and_hits(model, dev_rank_batcher, vocab, 'dev_evaluation', log=logger if not is_optuna else None)

    def test_func():
        return ranking_and_hits(model, test_rank_batcher, vocab, 'test_evaluation',
                                log=logger) if not is_optuna else None

    if args.resume:
        load_model_func(model_path)
        model.eval()
        if not is_optuna: test_func()
        valid_func()
    else:
        model.init()
        pass

    total_param_size = []
    params = [value.numel() for value in model.parameters()]

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    max_mrr = 0

    min_loss = float('inf')
    early_stopping_counter = 0
    tmp_loss = []

    for epoch in range(args.epochs):
        model.train()
        sum_train = 0
        for i, str2var in enumerate(train_batcher):
            opt.zero_grad()
            e1 = str2var['e1']
            rel = str2var['rel']
            e2_multi = str2var['e2_multi1_binary'].float()
            # label smoothing
            # e2_multi = ((1.0 - args.label_smoothing) * e2_multi) + (1.0 / e2_multi.size(1))

            sum_train += (e2_multi[e2_multi != 0]).sum()

            pred = model.forward(e1, rel)
            loss = model.loss(pred, e2_multi)
            tmp_loss.append(loss.item())
            loss.backward()
            opt.step()

            train_batcher.state.loss = loss.cpu()

        logger.debug(f"sum_train: {sum_train}")
        model.eval()
        with torch.no_grad():
            if (epoch + 1) % 5 == 0:
                _, _, _, _, _, mean_reciprocal_rank = valid_func()
                test_func()
                if max_mrr < mean_reciprocal_rank:
                    save_model_func(model_path_tmp)

        loss = (sum(tmp_loss) / len(tmp_loss))
        logger.info("loss {0}".format(loss))
        if min_loss > loss:
            early_stopping_counter = 0
            min_loss = loss
        else:
            early_stopping_counter += 1
        if early_stopping_counter == 10:
            break

    load_model_func(model_path_tmp)
    if is_optuna:
        os.remove(model_path_tmp)
        model.eval()
        with torch.no_grad():
            _, _, _, _, _, mean_reciprocal_rank = valid_func()
            return mean_reciprocal_rank.item()
    else:
        os.rename(model_path_tmp, model_path)
        return None


def test(
        args, vocab, num_entities, input_keys, device, *, logger,
        model_path, is_optuna=False,
):
    # train_batcher = StreamBatcher(args.KGdata, 'train', args.batch_size, randomize=True, keys=input_keys,
    #                               loader_threads=args.loader_threads)
    dev_rank_batcher = StreamBatcher(args.KGdata, 'dev_ranking', args.test_batch_size, randomize=False,
                                     loader_threads=args.loader_threads, keys=input_keys)
    test_rank_batcher = StreamBatcher(args.KGdata, 'test_ranking', args.test_batch_size, randomize=False,
                                      loader_threads=args.loader_threads, keys=input_keys)

    model = select_model(args, vocab, logger=logger)
    model.init()
    model.to(device)

    logger.info("test ready")

    def load_model_func(_model_path):
        model_params = torch.load(_model_path)
        # print(model)
        # total_param_size = []
        # params = [(key, value.size(), value.numel()) for key, value in model_params.items()]
        # for key, size, count in params:
        #     total_param_size.append(count)
        #     print(key, size, count)
        # print(np.array(total_param_size).sum())
        model.load_state_dict(model_params)
        return model

    def valid_func():
        return ranking_and_hits(model, dev_rank_batcher, vocab, 'dev_evaluation', log=logger if not is_optuna else None)

    def test_func():
        return ranking_and_hits(model, test_rank_batcher, vocab, 'test_evaluation',
                                log=logger) if not is_optuna else None

    model.eval()

    load_model_func(model_path)
    valid_func()
    test_func()


def conve1train(args, model_path, *, logger):
    device = torch.device("cpu")
    if Config.cuda:
        device = torch.device("cuda")
    elif args.device == 'mpu':
        device = torch.device("mps")

    if args.preprocess: preprocess(args.KGdata, delete_data=True)
    input_keys = ['e1', 'rel', 'rel_eval', 'e2', 'e2_multi1', 'e2_multi2']
    p = Pipeline(args.KGdata, keys=input_keys)
    p.load_vocabs()
    vocab = p.state['vocab']

    num_entities = vocab['e1'].num_token

    if args.test:
        test(
            args,
            vocab, num_entities, input_keys, device,
            is_optuna=False, model_path=model_path,
            logger=logger
        )
    else:
        train(
            args,
            vocab, num_entities, input_keys, device,
            is_optuna=False, model_path=model_path,
            logger=logger
        )


def conve_optuna(args, model_sub_dir, model_name, *, logger):
    device = torch.device("cpu")
    if Config.cuda:
        logger.info("use CUDA")
        device = torch.device("cuda")
    elif args.device == 'mpu':
        device = torch.device("mps")

    if args.preprocess: preprocess(args.KGdata, delete_data=True)
    input_keys = ['e1', 'rel', 'rel_eval', 'e2', 'e2_multi1', 'e2_multi2']
    p = Pipeline(args.KGdata, keys=input_keys)
    p.load_vocabs()
    vocab = p.state['vocab']

    num_entities = vocab['e1'].num_token

    def objective(trial: Trial):
        args.model = "conve"
        args.lr = trial.suggest_loguniform('lr', 1e-6, 1e-2)
        args.l2 = trial.suggest_loguniform('l2', 1e-6, 1e-2)
        model_dir = os.path.join(model_sub_dir, "op{}".format(trial.number))
        os.makedirs(model_dir, exist_ok=False)
        model_path = os.path.join(model_dir, model_name)
        return train(
            args,
            vocab, num_entities, input_keys, device,
            is_optuna=True, model_path=model_path,
            logger=logger
        )

    study = optuna.create_study(
        study_name='conve', storage=f'sqlite:///./{model_sub_dir}/optuna_study.db', load_if_exists=True,
        direction='maximize'
    )
    study.optimize(objective, n_trials=20)


def main():
    parser = argparse.ArgumentParser(description='Link prediction for knowledge graphs')
    parser.add_argument('--log-file', type=str, default='./convE_main.log', help='log file')
    parser.add_argument('--batch-size', type=int, default=128, help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=128,
                        help='input batch size for testing/validation (default: 128)')
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs to train (default: 1000)')
    parser.add_argument('--lr', type=float, default=0.003, help='learning rate (default: 0.003)')
    parser.add_argument('--seed', type=int, default=17, metavar='S', help='random seed (default: 17)')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--KGdata', type=str, default='FB15k-237',
                        help='Dataset to use: {FB15k-237, YAGO3-10, WN18RR, umls, nations, kinship}, default: FB15k-237')
    parser.add_argument('--l2', type=float, default=0.0,
                        help='Weight decay value to use in the optimizer. Default: 0.0')
    parser.add_argument('--model', type=str, default='conve',
                        help='Choose from: {conve, distmult, complex, transformere}')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Choose from: {cpu, cuda, mpu}', choices=['cpu', 'cuda', 'mpu'])
    parser.add_argument('--embedding-dim', type=int, default=200,
                        help='The embedding dimension (1D). Default: 200')
    parser.add_argument('--embedding-shape1', type=int, default=20,
                        help='The first dimension of the reshaped 2D embedding. '
                             'The second dimension is infered. Default: 20')
    parser.add_argument('--nhead', type=int, default=8,
                        help='number of head in transformer. Default: 8')  #
    parser.add_argument('--transformer-drop', type=float, default=0.1,
                        help='Dropout for the transformer. Default: 0.1.')  #
    parser.add_argument('--hidden-drop', type=float, default=0.3, help='Dropout for the hidden layer. Default: 0.3.')
    parser.add_argument('--input-drop', type=float, default=0.2, help='Dropout for the input embeddings. Default: 0.2.')
    parser.add_argument('--feat-drop', type=float, default=0.2,
                        help='Dropout for the convolutional features. Default: 0.2.')
    parser.add_argument('--lr-decay', type=float, default=0.995,
                        help='Decay the learning rate by this factor every epoch. Default: 0.995')
    parser.add_argument('--loader-threads', type=int, default=4,
                        help='How many loader threads to use for the batch loaders. Default: 4')
    parser.add_argument('--preprocess', action='store_true',
                        help='Preprocess the dataset. Needs to be executed only once. Default: 4')
    parser.add_argument('--resume', action='store_true', help='Resume a model.')
    parser.add_argument('--use-bias', action='store_true', help='Use a bias in the convolutional layer. Default: True')
    parser.add_argument('--label-smoothing', type=float, default=0.1, help='Label smoothing value to use. Default: 0.1')
    parser.add_argument('--hidden-size', type=int, default=9728,
                        help='The side of the hidden layer. The required size changes with the size of the embeddings. Default: 9728 (embedding size 200).')
    parser.add_argument('--optuna', action='store_true', help='optuna', default=False)
    parser.add_argument('--test', action='store_true', help='optuna', default=False)

    args = parser.parse_args()

    # parse console parameters and set global variables
    Config.backend = 'pytorch'
    Config.cuda = True if args.device == 'cuda' else False
    Config.embedding_dim = args.embedding_dim
    logger = setup_logger(__name__, logfile=args.log_file, console_level='debug')

    model_name = '{2}_{0}_{1}'.format(args.input_drop, args.hidden_drop, args.model)
    torch.manual_seed(args.seed)
    if not args.optuna:
        model_path = 'saved_models/{0}_{1}.model'.format(args.KGdata, model_name)
        conve1train(args, model_path, logger=logger)
    else:
        model_sub_dir = 'saved_models/optuna/'
        conve_optuna(args, model_sub_dir, model_name, logger=logger)


if __name__ == '__main__':
    main()
