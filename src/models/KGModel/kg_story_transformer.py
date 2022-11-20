#!/usr/bin/python
# -*- coding: utf-8 -*-
import itertools
import os
import sys
from pathlib import Path

# noinspection PyUnresolvedReferences
from typing import List, Dict, Tuple, Optional, Union, Callable

import abc

import numpy as np
import torch
from torch.nn import functional as F, Parameter

from torch.nn.init import xavier_normal_
# from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import TransformerEncoderLayer, TransformerEncoder, Softmax

from utils.torch import MM

from models.utilModules.tranformer import PositionalEncoding
from models.utilModules.mlp_mixer import MlpMixer, MlpMixerLayer

PROJECT_DIR = Path(__file__).resolve().parents[2]


def add_bos_eos(triple: np.ndarray,
                bos_token_h, bos_token_r, bos_token_t,
                eos_token_h, eos_token_r, eos_token_t,
                is_shuffle_in_same_head=False
                ):
    array_bos = np.array([[bos_token_h, bos_token_r, bos_token_t]])
    array_eos = np.array([[eos_token_h, eos_token_r, eos_token_t]])
    old_s = triple[0][0]
    before_i = 0
    tmp_list = []
    for i, (s, r, e) in enumerate(triple):
        if old_s != s:
            tmp = triple[before_i: i]
            if is_shuffle_in_same_head:
                np.random.shuffle(tmp)
            tmp_list.append(tmp)
            old_s, before_i = s, i

    new_triple = np.concatenate(
        list(itertools.chain(*[(array_bos, _tmp, array_eos) for _tmp in tmp_list]))
    )
    return new_triple


class KgStoryTransformer01(torch.nn.Module):
    def __init__(self, args, num_entities, num_relations, **kwargs):
        super().__init__()
        embedding_dim = args.embedding_dim

        nhead, num_layers, dim_feedforward = 4, 4, 1028  # 適当
        position_encoding_drop = 0.1
        transformer_drop = 0.1

        encoder_layer = TransformerEncoderLayer(
            d_model=embedding_dim, nhead=nhead, dropout=transformer_drop,
            batch_first=True, dim_feedforward=dim_feedforward,
            activation=F.gelu, norm_first=True
        )

        self.embedding_dim = embedding_dim
        self.max_len = args.max_len
        self.padding_token_r, self.padding_token_e = args.padding_token_e, args.padding_token_r
        self.cls_token_r, self.cls_token_e = args.cls_token_r, args.cls_token_e
        self.sep_token_r, self.sep_token_e = args.sep_token_r, args.sep_token_e
        self.mask_token_r, self.mask_token_e = args.mask_token_r, args.mask_token_e
        self.bos_token_r, self.bos_token_e = args.bos_token_r, args.bos_token_e
        self.eos_token_r, self.eos_token_e = args.eos_token_r, args.eos_token_e

        self.emb_entity = torch.nn.Embedding(
            num_entities, embedding_dim, padding_idx=self.padding_token_e)
        self.emb_relation = torch.nn.Embedding(
            num_relations, embedding_dim, padding_idx=self.padding_token_r)

        self.weight_head = torch.nn.Parameter(torch.tensor(1.0))
        self.weight_relation = torch.nn.Parameter(torch.tensor(1.0))
        self.weight_tail = torch.nn.Parameter(torch.tensor(1.0))

        self.head_dense = torch.nn.Linear(embedding_dim, embedding_dim)

        self.pe = PositionalEncoding(embedding_dim, dropout=position_encoding_drop, max_len=self.max_len)
        self.transformer = TransformerEncoder(encoder_layer, num_layers)
        self.norm_after_transformer = torch.nn.LayerNorm([embedding_dim])
        #
        self.head_maskdlm_layer = torch.nn.Linear(embedding_dim, embedding_dim)
        self.head_maskdlm_norm = torch.nn.LayerNorm([embedding_dim])
        #
        self.relation_maskdlm_layer = torch.nn.Linear(embedding_dim, embedding_dim)
        self.relation_maskdlm_norm = torch.nn.LayerNorm([embedding_dim])
        #
        self.tail_maskdlm_layer = torch.nn.Linear(embedding_dim, embedding_dim)
        self.tail_maskdlm_norm = torch.nn.LayerNorm([embedding_dim])
        #
        self.no_use_pe = 'no_use_pe' in kwargs and kwargs['no_use_pe']

    def get_head_pred(self, x: torch.Tensor):
        # x.shape = [semi_batch, embedding_dim]
        x = self.head_maskdlm_layer(x)
        x = F.gelu(x)
        x = self.head_maskdlm_norm(x)
        x = torch.mm(x, self.emb_entity.weight.transpose(1, 0))
        return x  # F.softmax(x, dim=)  # x.shape = [semi_batch, (num_story + some special entity num)]

    def get_relation_pred(self, x: torch.Tensor):
        # x.shape = [semi_batch, embedding_dim]
        x = self.relation_maskdlm_layer(x)
        x = F.gelu(x)
        x = self.relation_maskdlm_norm(x)
        x = torch.mm(x, self.emb_relation.weight.transpose(1, 0))
        return x  # x.shape = [semi_batch, (num_relation + some special entity num)]

    def get_tail_pred(self, x: torch.Tensor):
        # x.shape = [semi_batch, embedding_dim]
        x = self.tail_maskdlm_layer(x)
        x = F.gelu(x)
        x = self.tail_maskdlm_norm(x)
        x = torch.mm(x, self.emb_entity.weight.transpose(1, 0))
        return x  # F.softmax(x)  # x.shape = [semi_batch, (num_entities + some special entity num)]

    def forward(self, triple: torch.Tensor):
        # x.shape = [batch, triple_len, 3]
        emb_head = self.emb_entity(triple[:, :, 0])
        emb_head = F.gelu(self.head_dense(emb_head))
        emb_rel = self.emb_relation(triple[:, :, 1])
        emb_tail = self.emb_entity(triple[:, :, 2])
        #
        x = emb_head * self.weight_head + emb_rel * self.weight_relation + emb_tail * self.weight_tail
        x = self.pe(x) if not self.no_use_pe else x
        x = self.transformer.forward(x)
        x = self.norm_after_transformer(x)
        # x.shape = [batch, triple_len, embedding len]
        return x

    def pre_train_forward(self, triple: torch.Tensor,
                          mask_head: torch.Tensor, mask_relation: torch.Tensor, mask_tail: torch.Tensor):
        #
        x = self.forward(triple)
        # entity mask
        x = x.reshape(-1, self.embedding_dim)
        head_pred = self.get_head_pred(x[mask_head.reshape(-1)])
        relation_pred = self.get_relation_pred(x[mask_relation.reshape(-1)])
        tail_pred = self.get_tail_pred(x[mask_tail.reshape(-1)])
        return x, head_pred, relation_pred, tail_pred


if __name__ == '__main__':
    pass
