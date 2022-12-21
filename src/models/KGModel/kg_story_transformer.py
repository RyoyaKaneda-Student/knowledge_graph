#!/usr/bin/python
# -*- coding: utf-8 -*-
import dataclasses
import itertools
from collections import OrderedDict
# noinspection PyUnresolvedReferences
from typing import List, Dict, Tuple, Optional, Union, Callable

import numpy as np
import torch
from torch import flatten, mm
# from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import TransformerEncoderLayer, TransformerEncoder

from models.utilModules.tranformer import PositionalEncoding
from utils.torch import all_same_shape


def add_bos(triple: np.ndarray,
            bos_token_h, bos_token_r, bos_token_t,):
    array_bos = np.array([[bos_token_h, bos_token_r, bos_token_t]])
    old_s = triple[0][0]
    before_i = 0
    tmp_list = []
    for i, (s, r, e) in enumerate(triple):
        if old_s != s:
            tmp = triple[before_i: i]
            tmp_list.append(tmp)
            old_s, before_i = s, i

    new_triple = np.concatenate(
        list(itertools.chain(*[(array_bos, _tmp) for _tmp in tmp_list]))
    )
    return new_triple


class SpecialTokens:
    padding_token_e: int
    padding_token_r: int


@dataclasses.dataclass
class FakeSpecialTokens(SpecialTokens):
    default: int = 0

    def __getattr__(self):
        return self.default


@dataclasses.dataclass
class SpecialTokens01(SpecialTokens):
    padding_token_e: int
    padding_token_r: int
    cls_token_e: int
    cls_token_r: int
    mask_token_e: int
    mask_token_r: int
    sep_token_e: int
    sep_token_r: int
    bos_token_e: int
    bos_token_r: int


class KgStoryTransformer00(torch.nn.Module):
    def __init__(self, args, num_entity, num_relations, special_tokens, **kwargs):
        """

        Args:
            args:
            num_entity(int):
            num_relations(int):
            special_tokens(SpecialTokens):
            **kwargs:
        """
        super().__init__()
        # get from args
        max_len = args.max_len
        embedding_dim = args.embedding_dim
        entity_embedding_dim = args.entity_embedding_dim
        relation_embedding_dim = args.relation_embedding_dim
        nhead, num_layers, dim_feedforward = args.nhead, args.num_layers, args.dim_feedforward
        position_encoder_drop, transformer_drop = args.position_encoder_drop, args.transformer_drop
        padding_token_e, padding_token_r = special_tokens.padding_token_e, special_tokens.padding_token_r
        # set default value
        transformer_activation, transformer_norm_first = torch.nn.GELU(), True
        no_use_pe = args.no_use_pe
        is_separate_head_and_tail = args.separate_head_and_tail
        del args

        # define some model params
        self.embedding_dim = embedding_dim
        self.entity_embedding_dim = entity_embedding_dim
        self.relation_embedding_dim = relation_embedding_dim
        self.max_len = max_len
        self.special_tokens = special_tokens
        self.is_separate_head_and_tail = is_separate_head_and_tail
        self.no_use_pe = no_use_pe

        self.emb_entity = torch.nn.Embedding(num_entity, entity_embedding_dim, padding_idx=padding_token_e)
        self.emb_relation = torch.nn.Embedding(num_relations, relation_embedding_dim, padding_idx=padding_token_r)

        self.weight_head = torch.nn.Parameter(torch.tensor(1.0))
        self.weight_relation = torch.nn.Parameter(torch.tensor(1.0))
        self.weight_tail = torch.nn.Parameter(torch.tensor(1.0))

        self.pe = PositionalEncoding(embedding_dim, dropout=position_encoder_drop, max_len=self.max_len)
        self.transformer = TransformerEncoder(TransformerEncoderLayer(
            d_model=embedding_dim, nhead=nhead, dropout=transformer_drop,
            batch_first=True, dim_feedforward=dim_feedforward,
            activation=transformer_activation, norm_first=transformer_norm_first
        ), num_layers=num_layers)
        self.norm_after_transformer = torch.nn.LayerNorm([embedding_dim])  # it is because of norm first
        # maskdlm for head, relation, tail
        self.head_maskdlm = torch.nn.Sequential(OrderedDict(
            [('linear', torch.nn.Linear(embedding_dim, embedding_dim)), ('activation', torch.nn.Tanh())]))
        self.relation_maskdlm = torch.nn.Sequential(OrderedDict(
            [('linear', torch.nn.Linear(embedding_dim, embedding_dim)), ('activation', torch.nn.Tanh())]))
        self.tail_maskdlm = torch.nn.Sequential(OrderedDict(
            [('linear', torch.nn.Linear(embedding_dim, embedding_dim)), ('activation', torch.nn.Tanh())]))
        # LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)

    def assert_check(self):
        pass

    def get_head_pred(self, x: torch.Tensor):
        entity_embeddings = self.emb_entity.weight.transpose(1, 0)
        x = mm(self.head_maskdlm(x), entity_embeddings)
        return x

    def get_relation_pred(self, x: torch.Tensor):
        relation_embeddings = self.emb_relation.weight.transpose(1, 0)
        x = mm(self.relation_maskdlm(x), relation_embeddings)
        return x  # x.shape = [semi_batch, (num_relation + some special entity num)]

    def get_tail_pred(self, x: torch.Tensor):
        entity_embeddings = self.emb_entity.weight.transpose(1, 0)
        x = mm(self.tail_maskdlm(x), entity_embeddings)
        return x  # F.softmax(x)  # x.shape = [semi_batch, (num_entities + some special entity num)]

    def get_embedding(self, head, relation, tail) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert all_same_shape(head, relation, tail)
        return self.emb_entity(head), self.emb_relation(relation), self.emb_entity(tail)

    def get_combined_head_relation_tail(self, emb_head, emb_rel, emb_tail):
        assert emb_head.shape == emb_rel.shape and emb_rel.shape == emb_tail.shape
        x = emb_head * self.weight_head + emb_rel * self.weight_relation + emb_tail * self.weight_tail
        return self.pe(x) if not self.no_use_pe else x

    def _forward(self, triple: torch.Tensor):
        # batch, triple_len = triple.shape[0:2]  # for debugging
        # assert triple.shape == (batch, triple_len, 3)
        emb_head, emb_rel, emb_tail = self.get_embedding(triple[:, :, 0], triple[:, :, 1], triple[:, :, 2])
        # assert all_same_shape(emb_head) and emb_head.shape == (batch, triple_len, self.embedding_dim)
        x = self.get_combined_head_relation_tail(emb_head, emb_rel, emb_tail)
        # assert x.shape == (batch, triple_len, self.embedding_dim)
        x = self.norm_after_transformer(self.transformer(x))
        # assert x.shape == (batch, triple_len, self.embedding_dim)
        return x

    def forward(self, triple, mask_head_filter, mask_relation_filter, mask_tail_filter):
        """

        Args:
            triple(torch.Tensor):
            mask_head_filter(torch.Tensor):
            mask_relation_filter(torch.Tensor):
            mask_tail_filter(torch.Tensor):

        Returns:
            tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        # get item
        x = self._forward(triple)
        # entity mask
        x = x.reshape(-1, self.embedding_dim)
        head_pred, relation_pred, tail_pred = None, None, None
        if mask_head_filter is not None:
            head_pred = self.get_head_pred(x[flatten(mask_head_filter)])
            pass
        if mask_relation_filter is not None:
            relation_pred = self.get_relation_pred(x[flatten(mask_relation_filter)])
            pass
        if mask_tail_filter is not None:
            tail_pred = self.get_tail_pred(x[flatten(mask_tail_filter)])
            pass
        return x, (head_pred, relation_pred, tail_pred)


class KgStoryTransformer01(KgStoryTransformer00):
    def __init__(self, args, num_entity, num_relations, special_tokens, **kwargs):
        super(KgStoryTransformer01, self).__init__(args, num_entity, num_relations, special_tokens, **kwargs)
        entity_embedding_dim = args.entity_embedding_dim
        del args
        self.head_dense = torch.nn.Sequential(OrderedDict([
            ('dense1', torch.nn.Linear(entity_embedding_dim, entity_embedding_dim)),
            ('activation', torch.nn.GELU()),
            ('dense2', torch.nn.Linear(entity_embedding_dim, entity_embedding_dim)), ]))

    def assert_check(self):
        assert self.embedding_dim == self.entity_embedding_dim and \
               self.embedding_dim == self.entity_embedding_dim
        pass

    def get_combined_head_relation_tail(self, emb_head, emb_rel, emb_tail):
        assert emb_head.shape == emb_rel.shape and emb_rel.shape == emb_tail.shape
        x = emb_head * self.weight_head + emb_rel * self.weight_relation + emb_tail * self.weight_tail
        return self.pe(x) if not self.no_use_pe else x


class KgStoryTransformer02(KgStoryTransformer00):
    """
    推定をベクトル距離ではない方式にしたパターン.
    """
    def __init__(self, args, num_entity, num_relations, special_tokens, **kwargs):
        super(KgStoryTransformer02, self).__init__(args, num_entity, num_relations, special_tokens, **kwargs)
        embedding_dim = args.embedding_dim
        entity_embedding_dim = args.entity_embedding_dim
        del args
        del self.head_maskdlm, self.relation_maskdlm, self.tail_maskdlm
        self.head_dense = torch.nn.Sequential(OrderedDict([
            ('dense1', torch.nn.Linear(entity_embedding_dim, entity_embedding_dim)),
            ('layer_norm', torch.nn.LayerNorm([entity_embedding_dim])),
            ('activation', torch.nn.GELU()),
            ('dense2', torch.nn.Linear(entity_embedding_dim, entity_embedding_dim)), ]))
        self.head_maskdlm = torch.nn.Sequential(OrderedDict([
            ('linear1', torch.nn.Linear(embedding_dim, embedding_dim)),
            ('layer_norm', torch.nn.LayerNorm([embedding_dim])),
            ('activation', torch.nn.GELU()),
            ('linear2', torch.nn.Linear(embedding_dim, num_entity)), ]))
        self.relation_maskdlm = torch.nn.Sequential(OrderedDict([
            ('linear1', torch.nn.Linear(embedding_dim, embedding_dim)),
            ('layer_norm', torch.nn.LayerNorm([embedding_dim])),
            ('activation', torch.nn.GELU()),
            ('linear2', torch.nn.Linear(embedding_dim, num_relations)), ]))
        self.tail_maskdlm = torch.nn.Sequential(OrderedDict([
            ('linear1', torch.nn.Linear(embedding_dim, embedding_dim)),
            ('layer_norm', torch.nn.LayerNorm([embedding_dim])),
            ('activation', torch.nn.GELU()),
            ('linear2', torch.nn.Linear(embedding_dim, num_entity)), ]))

    def assert_check(self):
        assert self.embedding_dim == self.entity_embedding_dim and \
               self.embedding_dim == self.entity_embedding_dim
        pass

    def get_head_pred(self, x: torch.Tensor):
        x = self.head_maskdlm(x)
        return x

    def get_relation_pred(self, x: torch.Tensor):
        x = self.relation_maskdlm(x)
        return x

    def get_tail_pred(self, x: torch.Tensor):
        x = self.tail_maskdlm(x)
        return x

    def get_combined_head_relation_tail(self, emb_head, emb_rel, emb_tail):
        assert emb_head.shape == emb_rel.shape and emb_rel.shape == emb_tail.shape
        x = emb_head * self.weight_head + emb_rel * self.weight_relation + emb_tail * self.weight_tail
        return self.pe(x) if not self.no_use_pe else x


class KgStoryTransformer03(KgStoryTransformer02):
    """
    triple の全てを MLP にいれるタイプ.
    その他は KgStoryTransformer02 と同じ.
    """
    def __init__(self, args, num_entity, num_relations, special_tokens, **kwargs):
        super(KgStoryTransformer03, self).__init__(args, num_entity, num_relations, special_tokens, **kwargs)
        embedding_dim = args.embedding_dim
        entity_embedding_dim = args.entity_embedding_dim
        relation_embedding_dim = args.relation_embedding_dim
        del args
        del self.head_dense, self.weight_head, self.weight_relation, self.weight_tail
        self.input_dense = torch.nn.Sequential(OrderedDict([
            ('dense1', torch.nn.Linear(2*entity_embedding_dim+relation_embedding_dim, embedding_dim)),
            ('layer_norm', torch.nn.LayerNorm([embedding_dim])),
            ('activation', torch.nn.GELU()),
            ('dense2', torch.nn.Linear(embedding_dim, embedding_dim)), ]))

    def get_combined_head_relation_tail(self, emb_head, emb_rel, emb_tail):
        assert emb_head.shape == emb_tail.shape
        x = torch.cat([emb_head, emb_rel, emb_tail], dim=2)
        x = self.input_dense(x)
        return self.pe(x) if not self.no_use_pe else x


if __name__ == '__main__':
    raise ValueError()
    pass
