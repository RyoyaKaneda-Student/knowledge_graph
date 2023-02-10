#!/usr/bin/python
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from argparse import Namespace
from collections import OrderedDict
# noinspection PyUnresolvedReferences
from typing import List, Dict, Tuple, Optional, Union, Callable, overload, Final

import numpy as np
import pandas as pd
import torch
from torch import flatten, nn, mm
from torch.nn import Embedding, Linear, Sequential
# from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import TransformerEncoderLayer, TransformerEncoder
from transformers import BertTokenizer, BertModel

# my modules
from models.utilModules.tranformer import PositionalEncoding, Feedforward
# my utils
from utils.utils import tqdm, version_check

LINEAR: Final = 'linear'
ACTIVATION: Final = 'activation'

ENTITY_EMBEDDINGS: Final = 'entity_embeddings'
RELATION_EMBEDDINGS: Final = 'relation_embeddings'

MASKED_LM: Final = 'maskdlm'
HEAD_MASKED_LM: Final = 'head_maskdlm'
RELATION_MASKED_LM: Final = 'relation_maskdlm'
TAIL_MASKED_LM: Final = 'tail_maskdlm'

WEIGHT_HEAD: Final = 'weight_head'
WEIGHT_RELATION: Final = 'weight_relation'
WEIGHT_TAIL: Final = 'weight_tail'
ALL_WEIGHT_LIST = (WEIGHT_HEAD, WEIGHT_RELATION, WEIGHT_TAIL)


class KgSequenceTransformer(nn.Module, ABC):
    """KnowledgeGraph Sequence Transformer

    """

    def __init__(self, args, num_entity, num_relations, special_tokens,
                 do_head_pred=True, do_relation_pred=True, do_tail_pred=True,
                 *args_, **kwargs):
        """

        Args:
            args(Namespace):
            num_entity(int):
            num_relations(int):
            special_tokens(SpecialPaddingTokens):
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
        self.do_head_pred = do_head_pred
        self.do_relation_pred = do_relation_pred
        self.do_tail_pred = do_tail_pred

        self.entity_embeddings = Embedding(num_entity, entity_embedding_dim, padding_idx=padding_token_e)
        self.relation_embeddings = Embedding(num_relations, relation_embedding_dim, padding_idx=padding_token_r)

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
        # get entity id, relation id
        head, relation, tail = triple[:, :, 0], triple[:, :, 1], triple[:, :, 2]
        # get (head, relation, tail) embeddings
        head, relation, tail = self.get_emb_head(head), self.get_emb_relation(relation), self.get_emb_tail(tail)
        # get triples embeddings
        triples = self.get_triple_embedding(head, relation, tail)
        # get encoding embeddings
        triples = self.encoder(triples)
        # get pred items
        return triples, self.get_pred(triples, mask_head_filter, mask_relation_filter, mask_tail_filter)

    def assert_check(self, *args_, **kwargs):
        """check assertion

        """
        pass


class _Init(KgSequenceTransformer, ABC):
    @abstractmethod
    def init(self, args, *args_, **kwargs):
        """init model function

        """
        pass


class _GetEmb(KgSequenceTransformer, ABC):
    @abstractmethod
    def get_emb_head(self, head: torch.Tensor):
        """get head embedding function.

        """
        raise NotImplementedError()

    @abstractmethod
    def get_emb_relation(self, relation: torch.Tensor):
        """get relation embedding function.

        """
        raise NotImplementedError()

    @abstractmethod
    def get_emb_tail(self, tail: torch.Tensor):
        """get tail embedding function.

        """
        raise NotImplementedError()


class _GetTripleEmb(KgSequenceTransformer, ABC):
    @abstractmethod
    def get_triple_embedding(self, head_emb, relation_emb, tail_emb):
        raise NotImplementedError()


class _Encoder(KgSequenceTransformer, ABC):
    @abstractmethod
    def encoder(self, x):
        """Encoder(example transformer)

        Args:
            x:

        Returns:

        """
        raise NotImplementedError()


class _GetPred(KgSequenceTransformer, ABC):
    @abstractmethod
    def get_pred(self, x, mask_head_filter, mask_relation_filter, mask_tail_filter):
        raise NotImplementedError()

    @abstractmethod
    def get_head_pred(self, x: torch.Tensor):
        raise NotImplementedError()

    @abstractmethod
    def get_relation_pred(self, x: torch.Tensor):
        raise NotImplementedError()

    @abstractmethod
    def get_tail_pred(self, x: torch.Tensor):
        raise NotImplementedError()


# init functions


class _DefaultInit(_Init, ABC):
    def init(self, args, *args_, **kwargs):
        super(_DefaultInit, self).init(args, *args_, **kwargs)


class _LabelInit(_Init, ABC):
    """This is the option method for inheritance.

    * Entity embedding is pre-set according to the natural language of the label by init function.

    """

    def __init__(self, args, *args_, **kwargs):
        super(_LabelInit, self).__init__(args, *args_, **kwargs)

    def init(self, args, *args_, **kwargs):
        super(_LabelInit, self).init(args, *args_, **kwargs)
        from models.datasets.data_helper import MyDataHelperForStory

        if not args.init_embedding_using_bert:
            return
        device = args.device
        force_set_label: bool = getattr(args, 'force_set_label', False)
        data_helper: MyDataHelperForStory = kwargs['data_helper']

        embedding_dim, num_embeddings = self.entity_embeddings.embedding_dim, self.entity_embeddings.num_embeddings
        assert embedding_dim == 768, f"The entity_embedding_dim must 768 but entity_embedding_dim=={embedding_dim}"

        bert_model = BertModel.from_pretrained('bert-base-uncased')
        bert_model.to(device)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        processed_entities_label = [
            f"[CLS] {label}" if label != '' or label is None else
            f"[CLS] {entity_name}" if force_set_label else
            ''
            for label, entity_name in zip(data_helper.processed_entities_label, data_helper.processed_entities)
        ]
        result = tokenizer.batch_encode_plus(processed_entities_label, add_special_tokens=False)
        input_ids_list = result['input_ids']

        with torch.no_grad():
            # get pre_embeddings
            bert_model.eval()
            entity_embeddings_list = [
                bert_model(torch.tensor([input_ids]).to(device))[0][0, 0].to('cpu') if len(input_ids) > 1 else None
                for input_ids in tqdm(input_ids_list)]
            entity_embeddings_filter = torch.tensor([True if x is not None else False for x in entity_embeddings_list])
            entity_embeddings_list = [x for x in entity_embeddings_list if x is not None]
            entity_embeddings_tensor = torch.stack(entity_embeddings_list)
            pre_embeddings = torch.stack(entity_embeddings_list)
            # get setting parameters using pre_embeddings.
            pre_embedding_mean = torch.mean(entity_embeddings_tensor).item()
            pre_embedding_mean_var = torch.var(entity_embeddings_tensor).item()
            num_pre_emb = pre_embeddings.shape[0]
            # set parameters
            self.entity_embeddings.weight[entity_embeddings_filter] = pre_embeddings
            self.entity_embeddings.weight[~entity_embeddings_filter] = torch.normal(
                pre_embedding_mean, pre_embedding_mean_var, size=(num_embeddings - num_pre_emb, embedding_dim))


# Get (head, relation, tail) embeddings


class _GetEmbSimple(_GetEmb):
    def __init__(self, args, *args_, **kwargs):
        super(_GetEmbSimple, self).__init__(args, *args_, **kwargs)

    def get_emb_head(self, x: torch.Tensor):
        return self.entity_embeddings(x)

    def get_emb_relation(self, x: torch.Tensor):
        return self.relation_embeddings(x)

    def get_emb_tail(self, x: torch.Tensor):
        return self.entity_embeddings(x)


class _GetEmbByActivateHead(_GetEmbSimple):
    def __init__(self, args, *args_, **kwargs):
        super(_GetEmbByActivateHead, self).__init__(args, *args_, **kwargs)
        embedding_dim = args.embedding_dim
        self.head_activate = Feedforward(embedding_dim, embedding_dim)

    def get_emb_head(self, x: torch.Tensor):
        return self.head_activate(self.entity_embeddings(x))


# Get triples embeddings


class _GetTripleEmbAsSum(_GetTripleEmb):
    def __init__(self, args, *args_, **kwargs):
        super(_GetTripleEmbAsSum, self).__init__(args, *args_, **kwargs)
        self.weight_head = torch.nn.Parameter(torch.tensor(1.0))
        self.weight_relation = torch.nn.Parameter(torch.tensor(1.0))
        self.weight_tail = torch.nn.Parameter(torch.tensor(1.0))

    def get_triple_embedding(self, emb_head, emb_rel, emb_tail):
        x = emb_head * self.weight_head + emb_rel * self.weight_relation + emb_tail * self.weight_tail
        return x


class _GetTripleEmbAsCatActivate(_GetTripleEmb):
    def __init__(self, args, *args_, **kwargs):
        super(_GetTripleEmbAsCatActivate, self).__init__(args, *args_, **kwargs)
        # get from args
        embedding_dim = args.embedding_dim
        entity_embedding_dim = args.entity_embedding_dim
        relation_embedding_dim = args.relation_embedding_dim
        # set new module
        self.input_activate = Feedforward(2 * entity_embedding_dim + relation_embedding_dim, embedding_dim)

    def get_triple_embedding(self, emb_head, emb_rel, emb_tail):
        x = self.input_activate(torch.cat([emb_head, emb_rel, emb_tail], dim=2))
        return x


# Get encoding embeddings


class _TransformerEncoder(KgSequenceTransformer):
    def __init__(self, args, num_entity, num_relations, special_tokens, *args_, **kwargs):
        """

        Args:
            args:
            num_entity(int):
            num_relations(int):
            special_tokens(SpecialTokens):
            **kwargs:
        """
        super(_TransformerEncoder, self).__init__(args, num_entity, num_relations, special_tokens, *args_, **kwargs)
        # set default value
        transformer_activation, transformer_norm_first = torch.nn.GELU(), True,
        nhead = args.nhead
        num_layers = args.num_layers
        dim_feedforward = args.dim_feedforward
        position_encoder_drop, transformer_drop = args.position_encoder_drop, args.transformer_drop
        embedding_dim = args.embedding_dim
        max_len = args.max_len
        # get embedding_dim and position_encoder_drop
        self.is_use_position_encoding = not args.no_use_pe
        if self.is_use_position_encoding:
            self.pe = PositionalEncoding(embedding_dim, dropout=position_encoder_drop, max_len=max_len)
            pass
        self.transformer = TransformerEncoder(TransformerEncoderLayer(
            d_model=embedding_dim, nhead=nhead, dropout=transformer_drop,
            batch_first=True, dim_feedforward=dim_feedforward,
            activation=transformer_activation, norm_first=transformer_norm_first
        ), num_layers=num_layers)
        self.norm_after_transformer = torch.nn.LayerNorm([embedding_dim])  # it is because of norm first

    def encoder(self, x):
        x = self.pe(x) if self.is_use_position_encoding else x
        return self.norm_after_transformer(self.transformer(x))


# Get pred items

class _GetPredSimple(_GetPred, ABC):
    def __init__(self, args, *args_, **kwargs):
        super(_GetPredSimple, self).__init__(args, *args_, **kwargs)

    def get_pred(self, x, mask_head_filter, mask_relation_filter, mask_tail_filter):
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
        return head_pred, relation_pred, tail_pred


class _GetPredByNearValue(_GetPredSimple):
    def __init__(self, args, *args_, **kwargs):
        super(_GetPredByNearValue, self).__init__(args, *args_, **kwargs)
        embedding_dim = args.embedding_dim
        entity_embedding_dim = args.entity_embedding_dim
        relation_embedding_dim = args.entity_embedding_dim
        # maskdlm for head, relation, tail
        self.head_maskdlm = Feedforward(
            embedding_dim, entity_embedding_dim, dim_feedforward=embedding_dim, add_norm=False)
        self.relation_maskdlm = Feedforward(
            embedding_dim, relation_embedding_dim, dim_feedforward=embedding_dim, add_norm=False)
        self.tail_maskdlm = Feedforward(
            embedding_dim, entity_embedding_dim, dim_feedforward=embedding_dim, add_norm=False)

    def get_head_embeddings(self, x: torch.Tensor):
        return self.head_maskdlm(x)

    def get_relation_embeddings(self, x: torch.Tensor):
        return self.relation_maskdlm(x)

    def get_tail_embeddings(self, x: torch.Tensor):
        return self.tail_maskdlm(x)

    def get_head_pred(self, x: torch.Tensor):
        entity_embeddings = self.entity_embeddings.weight.transpose(1, 0)
        x = mm(self.get_head_embeddings(x), entity_embeddings)
        return x

    def get_relation_pred(self, x: torch.Tensor):
        relation_embeddings = self.relation_embeddings.weight.transpose(1, 0)
        x = mm(self.get_relation_embeddings(x), relation_embeddings)
        return x  # x.shape = [semi_batch, (num_relation + some special entity num)]

    def get_tail_pred(self, x: torch.Tensor):
        entity_embeddings = self.entity_embeddings.weight.transpose(1, 0)
        x = mm(self.get_tail_embeddings(x), entity_embeddings)
        return x  # F.softmax(x)  # x.shape = [semi_batch, (num_entities + some special entity num)]


class _GetPredByOneHot(_GetPredSimple):
    def __init__(self, args, num_entity, num_relations, *args_, **kwargs):
        super(_GetPredByOneHot, self).__init__(args, num_entity, num_relations, *args_, **kwargs)
        embedding_dim = args.embedding_dim

        self.head_maskdlm = Feedforward(
            embedding_dim, num_entity, dim_feedforward=embedding_dim, add_norm=False)
        self.relation_maskdlm = Feedforward(
            embedding_dim, num_relations, dim_feedforward=embedding_dim, add_norm=False)
        self.tail_maskdlm = Feedforward(
            embedding_dim, num_entity, dim_feedforward=embedding_dim, add_norm=False)

    def get_head_pred(self, x: torch.Tensor):
        return self.head_maskdlm(x)

    def get_relation_pred(self, x: torch.Tensor):
        return self.relation_maskdlm(x)

    def get_tail_pred(self, x: torch.Tensor):
        return self.tail_maskdlm(x)


# main models


class KgSequenceTransformer00(
    _DefaultInit, _GetEmbSimple, _GetTripleEmbAsSum, _TransformerEncoder, _GetPredByNearValue
):
    """KgSequenceTransformer00

    * get head, relation, tail embedding.
    * weighted sum 3 embedding, and it is triple embedding.
    * get activated embedding by using Transformer.
    * after embedding activation for head, relation, tail.
    * pred head, relation, tail item by these embeddings.

    """

    def __init__(self, args, num_entity, num_relations, special_tokens, *args_, **kwargs):
        """

        Args:
            args:
            num_entity(int):
            num_relations(int):
            special_tokens(SpecialTokens):
            **kwargs:
        """
        super(KgSequenceTransformer00, self).__init__(
            args, num_entity, num_relations, special_tokens, *args_, **kwargs)


class KgSequenceTransformer01(
    _DefaultInit, _GetEmbByActivateHead, _GetTripleEmbAsSum, _TransformerEncoder, _GetPredByNearValue,
):
    """01

    """
    pass


class KgSequenceTransformer02(
    _DefaultInit, _GetEmbSimple, _GetTripleEmbAsSum, _TransformerEncoder, _GetPredByOneHot,
):
    """KgSequenceTransformer02

    * Changed from 00.
    * --- ---
    * Original estimation method: vector distance
    * This model: onehot
    * --- ---

    """
    pass


class KgSequenceTransformer0102(
    _DefaultInit, _GetEmbByActivateHead, _GetTripleEmbAsSum, _TransformerEncoder, _GetPredByOneHot,
):
    """KgSequenceTransformer0102

    * Changed from 00.
    * --- ---
    * Original get_emb_head function: return entity embedding.
    * This model:  return ACTIVATE entity embedding by using feedfoward.
    * ---
    * Original estimation method, vector distance
    * This model, onehot
    * --- ---

    """

    def __init__(self, args, num_entity, num_relations, special_tokens, **kwargs):
        super(KgSequenceTransformer0102, self).__init__(args, num_entity, num_relations, special_tokens, **kwargs)


class KgSequenceTransformer03(
    _DefaultInit, _GetEmbSimple, _GetTripleEmbAsCatActivate, _TransformerEncoder, _GetPredByOneHot,
):
    """KgSequenceTransformer03

    * Changed from 00.
    * --- ---
    * Original get_triple_embedding function: return sum of embeddings.
    * This model:  return cat and activate embeddings.
    * ---
    * Original estimation method: vector distance.
    * This model: onehot.
    * --- ---

    """


class KgSequenceTransformer03preInit(
    _LabelInit, _GetEmbSimple, _GetTripleEmbAsCatActivate, _TransformerEncoder, _GetPredByOneHot,
):
    """KgSequenceTransformer03 and pre init by bert model.

    """
    pass


if __name__ == '__main__':
    version_check(np, pd, torch)
    pass
