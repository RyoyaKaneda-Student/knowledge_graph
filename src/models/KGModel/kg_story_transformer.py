#!/usr/bin/python
# -*- coding: utf-8 -*-
import abc
import dataclasses
import itertools
from abc import ABC
from collections import OrderedDict
# noinspection PyUnresolvedReferences
from typing import List, Dict, Tuple, Optional, Union, Callable, overload, Final

import numpy as np
import torch
from torch import flatten, nn, mm
from torch.nn import Embedding, Linear, Sequential
# from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import TransformerEncoderLayer, TransformerEncoder

from models.utilModules.tranformer import PositionalEncoding
from utils.torch import all_same_shape

LINEAR: Final = 'linear'
ACTIVATION: Final = 'activation'


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


class Feedforward(torch.nn.Module):
    def __init__(self, d_model_in, d_model_out, dim_feedforward=None, activation=torch.nn.GELU()):
        super().__init__()
        dim_feedforward = dim_feedforward or d_model_out
        self.linear1 = Linear(d_model_in, dim_feedforward, bias=False)
        self.norm = torch.nn.LayerNorm([dim_feedforward])
        self.activation = activation
        self.linear2 = Linear(dim_feedforward, d_model_out)

    def forward(self, x: torch.Tensor):
        x = self.linear1(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x


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


class KgStoryTransformer(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, args, num_entity, num_relations, special_tokens):
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
        padding_token_e, padding_token_r = special_tokens.padding_token_e, special_tokens.padding_token_r
        # set default value
        del args

        # define some model params
        self.embedding_dim = embedding_dim
        self.entity_embedding_dim = entity_embedding_dim
        self.relation_embedding_dim = relation_embedding_dim
        self.max_len = max_len
        self.special_tokens = special_tokens

        self.entity_embeddings = Embedding(num_entity, entity_embedding_dim, padding_idx=padding_token_e)
        self.relation_embeddings = Embedding(num_relations, relation_embedding_dim, padding_idx=padding_token_r)

    @abc.abstractmethod
    def get_head_pred(self, x: torch.Tensor):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_relation_pred(self, x: torch.Tensor):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_tail_pred(self, x: torch.Tensor):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_triple_embedding(self, emb_head, emb_rel, emb_tail):
        raise NotImplementedError()

    def _forward(self, triple: torch.Tensor):
        x = self.get_triple_embedding(triple[:, :, 0], triple[:, :, 1], triple[:, :, 2])
        x = self.norm_after_transformer(self.transformer(x))
        return x

    def assert_check(self):
        pass


class KgStoryTransformer00(KgStoryTransformer, ABC):
    def __init__(self, args, num_entity, num_relations, special_tokens, **kwargs):
        """

        Args:
            args:
            num_entity(int):
            num_relations(int):
            special_tokens(SpecialTokens):
            **kwargs:
        """
        super(KgStoryTransformer00, self).__init__(args, num_entity, num_relations, special_tokens)
        # set default value
        transformer_activation, transformer_norm_first = torch.nn.GELU(), True,
        nhead = args.nhead
        num_layers = args.num_layers
        dim_feedforward = args.dim_feedforward
        position_encoder_drop, transformer_drop = args.position_encoder_drop, args.transformer_drop
        embedding_dim = args.embedding_dim
        # get embedding_dim and position_encoder_drop

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
        maskdlm_maker = lambda: OrderedDict(
            [(LINEAR, Linear(embedding_dim, embedding_dim)), (ACTIVATION, torch.nn.Tanh())])
        # maskdlm for head, relation, tail
        self.head_maskdlm = Sequential(maskdlm_maker())
        self.relation_maskdlm = Sequential(maskdlm_maker())
        self.tail_maskdlm = Sequential(maskdlm_maker())

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

    def get_emb_head(self, x: torch.Tensor):
        return self.entity_embeddings(x)

    def get_emb_relation(self, x: torch.Tensor):
        return self.relation_embeddings(x)

    def get_emb_tail(self, x: torch.Tensor):
        return self.entity_embeddings(x)

    def get_triple_embedding(self, head, relation, tail):
        emb_head, emb_rel, emb_tail = self.get_emb_head(head), self.get_emb_relation(relation), self.get_emb_tail(tail)
        x = emb_head * self.weight_head + emb_rel * self.weight_relation + emb_tail * self.weight_tail
        return self.pe(x)

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
        embedding_dim = args.embedding_dim
        self.head_activate = Feedforward(embedding_dim, embedding_dim)

    def get_emb_head(self, x: torch.Tensor):
        return self.head_activate(self.entity_embeddings(x))


class KgStoryTransformer02(KgStoryTransformer00):
    """
    推定をベクトル距離ではない方式にしたパターン.
    """
    def __init__(self, args, num_entity, num_relations, special_tokens, **kwargs):
        super(KgStoryTransformer02, self).__init__(args, num_entity, num_relations, special_tokens, **kwargs)
        embedding_dim = args.embedding_dim
        del self.head_maskdlm, self.relation_maskdlm, self.tail_maskdlm
        self.head_maskdlm = Feedforward(embedding_dim, num_entity, dim_feedforward=embedding_dim)
        self.relation_maskdlm = Feedforward(embedding_dim, num_relations, dim_feedforward=embedding_dim)
        self.tail_maskdlm = Feedforward(embedding_dim, num_entity, dim_feedforward=embedding_dim)

    def get_head_pred(self, x: torch.Tensor):
        return self.head_maskdlm(x)

    def get_relation_pred(self, x: torch.Tensor):
        return self.relation_maskdlm(x)

    def get_tail_pred(self, x: torch.Tensor):
        return self.tail_maskdlm(x)


class KgStoryTransformer0102(KgStoryTransformer01, KgStoryTransformer02):
    """
    """
    def __init__(self, args, num_entity, num_relations, special_tokens, **kwargs):
        super(KgStoryTransformer0102, self).__init__(args, num_entity, num_relations, special_tokens, **kwargs)


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
        del args, self.head_maskdlm, self.relation_maskdlm, self.tail_maskdlm
        del self.weight_head, self.weight_relation, self.weight_tail
        self.input_activate = Feedforward(entity_embedding_dim+relation_embedding_dim, embedding_dim)
        self.head_maskdlm = Feedforward(embedding_dim, num_entity)
        self.relation_maskdlm = Feedforward(embedding_dim, num_relations)
        self.tail_maskdlm = Feedforward(embedding_dim, num_entity)

    def get_triple_embedding(self, head, relation, tail):
        emb_head, emb_rel, emb_tail = self.get_emb_head(head), self.get_emb_relation(relation), self.get_emb_tail(tail)
        x = self.input_activate(torch.cat([emb_head, emb_rel, emb_tail], dim=2))
        return self.pe(x)


if __name__ == '__main__':
    raise ValueError()
    pass
