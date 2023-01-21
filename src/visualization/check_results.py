#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Check results.

"""
# ========== python ==========
from logging import Logger
from typing import Final

import matplotlib.pyplot as plt
# Machine learning
import pandas as pd
from IPython.display import display
# jupyter
import seaborn as sns
# torch
import torch

from const.const_values import DATASETS, DATA_HELPER
from const.const_values import MODEL
from const.const_values import PROJECT_DIR
from models.KGModel.kg_model import HEAD, RELATION, TAIL
# My items
from models.datasets.data_helper import MyDataHelper, DefaultTokens
from models.datasets.datasets_for_story import StoryTriple
# main function
from run_for_KGC import main_function
# My utils
from utils.setup import load_param
from utils.setup import setup_logger, get_device
from utils.torch import load_model, torch_fix_seed

from const.const_values import SpeckledBand

MASK_E = DefaultTokens.MASK_E
KILL = 'word.predicate:kill'
SEED: Final[int] = 42


def get_args_from_path(args_path: str, *, logger: Logger, device: torch.device):
    """get_args_from_path

    """
    args = load_param(args_path)

    # args.pre_train = True
    args.logger = logger
    args.device = device
    args.batch_size = 1
    args.pre_train = False
    args.init_embedding_using_bert = False
    del args.optuna_file, args.device_name, args.pid, args.study_name, args.n_trials
    logger.info(args)
    return args


def get_from_return_dict(args, return_dict):
    """get some items from return_dict.

    Args:
        args:
        return_dict:

    Returns:

    """
    model = return_dict[MODEL]
    model.eval()

    dataset_train: StoryTriple = return_dict[DATASETS][0]
    triple: torch.Tensor = dataset_train.triple
    data_helper: MyDataHelper = return_dict[DATA_HELPER]
    # evaluator: Checkpoint = return_dict[TRAIN_RETURNS][EVALUATOR]

    load_model(model, args.model_path, args.device)

    index2entity, index2relation = data_helper.processed_entities, data_helper.processed_relations
    entity2index = {e: i for i, e in enumerate(index2entity)}
    relation2index = {r: i for i, r in enumerate(index2relation)}

    triple_df = pd.DataFrame([(index2entity[_t[0]], index2relation[_t[1]], index2entity[_t[2]]) for _t in triple],
                             columns=[HEAD, RELATION, TAIL])
    story_entities = triple_df[HEAD].tolist()
    return model, triple, entity2index, relation2index, index2entity, index2relation, triple_df, story_entities


# noinspection PyTypeChecker
def extract(model, target, inputs):
    """to get attention function.

    """
    features: torch.Tensor = None

    def forward_hook(_module, _inputs, _):
        nonlocal features
        x, _, _ = _inputs
        outputs = _module.forward(x, x, x, need_weights=True)[1]
        features = outputs.detach().clone()

    handle = target.register_forward_hook(forward_hook)

    model.eval()
    model(inputs, torch.LongTensor([[]]), torch.LongTensor([[]]), torch.LongTensor([[]]))

    handle.remove()

    return features


def get_attention(model, input_, entities, relations):
    """get attention. It is the output of Transformer's last layer.

    """
    assert len(input_) == 1
    features = extract(model, model.transformer.layers[-1].self_attn, input_)[0]
    df_attention = pd.DataFrame(
        [[entities[h], relations[r], entities[t]] + [features[j, i].item() for j in range(len(features))] for
         i, (h, r, t) in enumerate(input_[0])])
    df_attention.columns = [HEAD, RELATION, TAIL] + [f'atten_from{i}' for i in range(len(df_attention.columns) - 3)]
    return df_attention


def show_attension_heatmap(df_attention):
    sns.heatmap(df_attention.iloc[:, 3:])
    plt.show()


def make_ranking(args, from_story_name, to_story_name, predicate_, whom_, subject_, why_, what_, where_,
                 *, return_dict):
    (model, triple,
     entity2index, relation2index, index2entity, index2relation,
     triple_df, story_entities) = get_from_return_dict(args, return_dict)

    bos_triple = [
        entity2index[DefaultTokens.BOS_E], relation2index[DefaultTokens.BOS_R], entity2index[DefaultTokens.BOS_E]]
    mask_e_id = entity2index[DefaultTokens.MASK_E]
    Holmes_id = entity2index['AllTitle:Holmes']

    if not (from_story_name is None and to_story_name is None):
        _start_index = story_entities.index(from_story_name) - 1
        _end_index = len(story_entities) - story_entities[::-1].index(to_story_name)
    else:
        _start_index = 0
        _end_index = 0
    question_ = torch.tensor(
        [
            bos_triple,
            [mask_e_id, relation2index['kgc:infoSource'], Holmes_id],
            [mask_e_id, relation2index['kgc:hasPredicate'], entity2index[predicate_]],
            [mask_e_id, relation2index['kgc:whom'], entity2index[whom_]],
            [mask_e_id, relation2index['kgc:subject'], entity2index[subject_]],
            [mask_e_id, relation2index['kgc:why'], entity2index[why_]],
            [mask_e_id, relation2index['kgc:what'], entity2index[what_]],
            [mask_e_id, relation2index['kgc:where'], entity2index[where_]],
        ]
    )
    mask_ = torch.zeros_like(question_, dtype=torch.bool)  # not mask all position
    mask_[1:, 0] = True  # where head position without bos token
    mask_[1:, 2] = True  # where tail position without bos token

    last_triples = triple[_start_index: _end_index]

    questions = torch.cat([last_triples, question_], dim=0).unsqueeze(0)
    masks = torch.cat([torch.zeros_like(last_triples), mask_], dim=0).to(torch.bool).transpose(1, 0).unsqueeze(0)

    data_list = []
    with torch.no_grad():
        _, (story_pred, relation_pred, entity_pred) = model(questions, masks[:, 0], masks[:, 1], masks[:, 2])
        sorted_ = torch.argsort(entity_pred, dim=1, descending=True)
        for i in range(sorted_.shape[1]):
            ans_ = sorted_[:, i]
            info_source_, predicate_pred, whom_pred, subject_pred, why_pred, what_pred, where_pred = ans_
            data_list.append([
                index2entity[predicate_pred], index2entity[whom_pred], index2entity[subject_pred],
                index2entity[why_pred], index2entity[what_pred], index2entity[where_pred]
            ])
    df_ranking = pd.DataFrame(data_list, columns=['predicate', 'whom', 'subject', 'why', 'what', 'where'])
    df_attention = get_attention(model, questions, index2entity, index2relation)

    return df_ranking, df_attention


def main_func01(_title, _victim_name, criminal, predicate, _last_index, _story_len, *, args, logger, return_dict):
    from_ = f'{_title}:{_last_index - _story_len + 1}'
    to_ = f'{_title}:{_last_index}'
    predicate = predicate
    victim = f'{_title}:{_victim_name}'
    criminal = f'{_title}:{criminal}'
    df_ranking, df_attention = make_ranking(
        args, from_, to_, predicate, victim, MASK_E, MASK_E, MASK_E, MASK_E, return_dict=return_dict)

    pred_rank = df_ranking.index[df_ranking['subject'] == criminal].tolist()[0]
    logger.info(f"The pred ranking about {criminal} is {pred_rank}")
    display(df_ranking.iloc[:max(20, pred_rank)])
    len_ = len(df_attention)
    for i in range(len_ - 10, len_):
        logger.info(f"index{i}: {df_attention.iloc[i, :3].tolist()}")
        display(df_attention.sort_values(f'atten_from{i}', ascending=False).iloc[:20, [0, 1, 2, 3 + i]])
        logger.info("----------")
    return df_ranking, df_attention


def check_killer(_title, _victim_name, _killer_name, _last_index, _story_len, *, args, logger, return_dict):
    return main_func01(
        _title, _victim_name, _killer_name, KILL, _last_index, _story_len,
        args=args, logger=logger, return_dict=return_dict
    )


def do_madara_pred(args, logger, return_dict, last_index=401, story_len=80):
    title = SpeckledBand
    victim_name = 'Julia'
    killer_name = 'Roylott'

    df_ranking_SpeckledBand, df_attention_SpeckledBand = check_killer(
        title, victim_name, killer_name, last_index, story_len, args=args, logger=logger, return_dict=return_dict)


def main():
    """main

    """
    args_path = f'{PROJECT_DIR}/models/230114/01/param.pkl'
    logger: Logger = setup_logger(__name__, f'{PROJECT_DIR}/log/jupyter_run.log', 'info')
    device = get_device(device_name='cpu', logger=logger)
    args = get_args_from_path(args_path, logger=logger, device=device)
    torch_fix_seed(seed=SEED)
    return_dict = main_function(args, logger=logger)

    do_madara_pred(args, logger, return_dict)


if __name__ == '__main__':
    main()
