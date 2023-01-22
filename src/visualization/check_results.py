#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Check results.

"""
# ========== python ==========
import argparse
from argparse import Namespace
from logging import Logger
from typing import Final, Optional, Sequence

import matplotlib.pyplot as plt
# Machine learning
import pandas as pd
from IPython.display import display
# jupyter
import seaborn as sns
# torch
import torch

# My items
from models.datasets.data_helper import MyDataHelper, DefaultTokens
from models.datasets.datasets_for_story import StoryTriple
# main function
from run_for_KGC import main_function
# My utils
from utils.setup import setup, load_param
from utils.torch import load_model, torch_fix_seed, DeviceName
# const
from models.KGModel.kg_model import HEAD, RELATION, TAIL
from const.const_values import DATASETS, DATA_HELPER, AbbeyGrange, SilverBlaze, ResidentPatient, DevilsFoot, SpeckledBand
from const.const_values import MODEL, PROJECT_DIR, EN_TITLE2LEN_INFO

MASK_E = DefaultTokens.MASK_E
KILL = 'word.predicate:kill'
TAKE = 'word.predicate:take'
BRING = 'word.predicate:bring'
DIE = 'word.predicate:die'
HIDE = 'word.predicate:hide'
SEED: Final[int] = 42


def setup_parser(args: Optional[Sequence[str]] = None) -> Namespace:
    """make parser function

    * My first-setup function needs the function which make and return parser.

    Args:
        args(:obj:`Sequence[str]`, optional): args list or None. Default to None.

    Returns:
        Namespace: your args instance.

    """
    parser = argparse.ArgumentParser(description='This is make and training source code for KGC.')
    paa = parser.add_argument
    paa('args_path', type=str)
    paa('--notebook', help='if use notebook, use this argument.', action='store_true')
    paa('--console-level', help='log level on console', type=str, default='debug', choices=['info', 'debug'])
    paa('--logfile', help='the path of saving log', type=str, default='log/test.log')
    paa('--param-file', help='the path of saving param', type=str, default='log/param.pkl')
    paa('--device-name', help=DeviceName.ALL_INFO, type=str, default=DeviceName.CPU, choices=DeviceName.ALL_LIST)

    paa('--AbbeyGrange-100', help='AbbeyGrange 100', action='store_true')
    paa('--AbbeyGrange-090', help='AbbeyGrange 090', action='store_true')
    paa('--AbbeyGrange_075', help='AbbeyGrange 075', action='store_true')

    paa('--DevilsFoot-100', help='DevilsFoot 100', action='store_true')
    paa('--DevilsFoot-090', help='DevilsFoot 090', action='store_true')
    paa('--DevilsFoot-075', help='DevilsFoot 075', action='store_true')

    paa('--ResidentPatient-100', help='ResidentPatient 100', action='store_true')
    paa('--ResidentPatient-090', help='ResidentPatient 090', action='store_true')
    paa('--ResidentPatient-075', help='ResidentPatient 075', action='store_true')

    paa('--SilverBlaze-100', help='SilverBlaze 100', action='store_true')
    paa('--SilverBlaze-090', help='SilverBlaze 090', action='store_true')
    paa('--SilverBlaze-075', help='SilverBlaze 075', action='store_true')

    paa('--SpeckledBand-100', help='SpeckledBand 100', action='store_true')
    paa('--SpeckledBand-090', help='SpeckledBand 090', action='store_true')
    paa('--SpeckledBand-075', help='SpeckledBand 075', action='store_true')

    args = parser.parse_args(args=args)
    return args


def get_args_from_path(args_path: str, *, logger: Logger, device: torch.device):
    """get_args_from_path

    """
    args = load_param(args_path)

    # args.pre_train = True
    args.logger = logger
    args.device = device
    args.batch_size = 1
    args.pre_train = False

    del args.optuna_file, args.device_name, args.pid, args.study_name, args.n_trials
    logger.info(args)
    return args


def get_from_return_dict(args, return_dict):
    """get some items from return_dict.

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
     entity2index, relation2index,
     index2entity, index2relation,
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


def main_func01(args, _title, _victim_name, criminal, predicate, _last_index, _story_len, *, logger, return_dict):
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


def check_killer(args, _title, _victim_name, _killer_name, _last_index, _story_len, *, logger, return_dict):
    return main_func01(
        args, _title, _victim_name, _killer_name, KILL, _last_index, _story_len,
        logger=logger, return_dict=return_dict
    )


def AbbeyGrange_pred(args, logger, return_dict, last_index, story_len):
    title = AbbeyGrange
    victim_name = 'Sir_Eustace_Brackenstall'
    killer_name = 'Jack_Croker'

    df_ranking, df_attention = check_killer(
        args, title, victim_name, killer_name, last_index, story_len, logger=logger, return_dict=return_dict)
    return df_ranking, df_attention


def DevilsFoot1_pred(args, logger, return_dict, last_index, story_len):
    title = DevilsFoot
    victim_name = 'Brenda'
    killer_name = 'Mortimer'

    df_ranking, df_attention = check_killer(
        args, title, victim_name, killer_name, last_index, story_len, logger=logger, return_dict=return_dict)
    return df_ranking, df_attention


def DevilsFoot2_pred(args, logger, return_dict, last_index, story_len):
    title = DevilsFoot
    victim_name = 'Mortimer'
    killer_name = 'Sterndale'

    df_ranking, df_attention = check_killer(
        args, title, victim_name, killer_name, last_index, story_len, logger=logger, return_dict=return_dict)
    return df_ranking, df_attention


def ResidentPatient_pred(args, logger, return_dict, last_index, story_len):
    title = ResidentPatient
    victim_name = 'Blessington'
    killer_name = ''

    df_ranking, df_attention = check_killer(
        args, title, victim_name, killer_name, last_index, story_len, logger=logger, return_dict=return_dict)
    return df_ranking, df_attention


def SpeckledBand_pred(args, logger, return_dict, last_index, story_len):
    title = SpeckledBand
    victim_name = 'Julia'
    killer_name = 'Roylott'

    df_ranking, df_attention = check_killer(
        args, title, victim_name, killer_name, last_index, story_len, logger=logger, return_dict=return_dict)
    return df_ranking, df_attention


def SilverBlaze_pred(args, logger, return_dict, last_index, story_len):
    title = SilverBlaze
    victim_name = f'{title}:Silver_Blaze'

    df_ranking, df_attention = make_ranking(args, f'SilverBlaze:{last_index-story_len+1}', f'SilverBlaze:{last_index}',
                                            BRING, MASK_E, MASK_E, MASK_E, victim_name, MASK_E, return_dict=return_dict)

    pred_rank = df_ranking.index[df_ranking['what'] == victim_name].tolist()[0]
    # logger.info(f"The pred ranking about {criminal} is {pred_rank}")
    display(df_ranking.iloc[:max(20, pred_rank)])
    len_ = len(df_attention)
    for i in range(len_ - 10, len_):
        logger.info(f"index{i}: {df_attention.iloc[i, :3].tolist()}")
        display(df_attention.sort_values(f'atten_from{i}', ascending=False).iloc[:20, [0, 1, 2, 3 + i]])
        logger.info("----------")

    return df_ranking, df_attention


def main():
    """main

    """
    torch_fix_seed(seed=SEED)
    args, logger, device = setup(setup_parser, PROJECT_DIR)
    trained_args = get_args_from_path(args.args_path, logger=logger, device=device)
    return_dict = main_function(trained_args, logger=logger)

    title2functions = {
        AbbeyGrange: [AbbeyGrange_pred],
        DevilsFoot: [DevilsFoot1_pred, DevilsFoot2_pred],
        ResidentPatient: [ResidentPatient_pred],
        SilverBlaze: [SilverBlaze_pred],
        SpeckledBand: [SpeckledBand_pred]
    }
    for i, percent in enumerate(['100', '090', '075']):
        for title in [AbbeyGrange, DevilsFoot, ResidentPatient, SilverBlaze, SpeckledBand]:
            if getattr(args, f"{title}_{percent}"):
                last_index = EN_TITLE2LEN_INFO[title][i]
                [func(trained_args, logger, return_dict, last_index, 80) for func in title2functions[title]]


if __name__ == '__main__':
    main()
