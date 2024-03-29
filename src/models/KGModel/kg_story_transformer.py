#!/usr/bin/python
# -*- coding: utf-8 -*-
import itertools
from abc import ABC, ABCMeta, abstractmethod
from collections import OrderedDict
from argparse import Namespace
# noinspection PyUnresolvedReferences
from typing import List, Dict, Tuple, Optional, Union, Callable, overload, Final

import numpy as np
import torch
from torch import flatten, nn, mm
from torch.nn import Embedding, Linear, Sequential
# from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import TransformerEncoderLayer, TransformerEncoder
from transformers import BertTokenizer, BertModel
# my modules
from models.utilModules.tranformer import PositionalEncoding, Feedforward
# my utils
from utils.utils import tqdm

LINEAR: Final = 'linear'
ACTIVATION: Final = 'activation'

MASKED_LM: Final = 'maskdlm'
HEAD_MASKED_LM: Final = 'head_maskdlm'
RELATION_MASKED_LM: Final = 'relation_maskdlm'
TAIL_MASKED_LM: Final = 'tail_maskdlm'

WEIGHT_HEAD: Final = 'weight_head'
WEIGHT_RELATION: Final = 'weight_relation'
WEIGHT_TAIL: Final = 'weight_tail'
ALL_WEIGHT_LIST = (WEIGHT_HEAD, WEIGHT_RELATION, WEIGHT_TAIL)


class KgStoryTransformer(nn.Module, metaclass=ABCMeta):
    """KgStoryTransformer

    * get head, relation, tail embedding.
    * get triple embedding by using head, relation, tail embedding.
    * get activated embedding by using Transformer.
    * pred head, relation, tail item by using triple embedding.

    """
    def __init__(self, args, num_entity, num_relations, special_tokens, **kwargs):
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

        self.entity_embeddings = Embedding(num_entity, entity_embedding_dim, padding_idx=padding_token_e)
        self.relation_embeddings = Embedding(num_relations, relation_embedding_dim, padding_idx=padding_token_r)

    @abstractmethod
    def get_head_pred(self, x: torch.Tensor):
        raise NotImplementedError()

    @abstractmethod
    def get_relation_pred(self, x: torch.Tensor):
        raise NotImplementedError()

    @abstractmethod
    def get_tail_pred(self, x: torch.Tensor):
        raise NotImplementedError()

    @abstractmethod
    def get_triple_embedding(self, emb_head, emb_rel, emb_tail):
        raise NotImplementedError()

    def _forward(self, triple: torch.Tensor):
        x = self.get_triple_embedding(triple[:, :, 0], triple[:, :, 1], triple[:, :, 2])
        x = self.norm_after_transformer(self.transformer(x))
        return x

    def assert_check(self):
        pass

    def init(self, args, **kwargs):
        pass


class KgStoryTransformerLabelInit(KgStoryTransformer, ABC):
    """This is the option method for inheritance.

    * Entity embedding is pre-set according to the natural language of the label by init function.

    """

    def init(self, args, **kwargs):
        from models.datasets.data_helper import MyDataHelper

        if not args.init_embedding_using_bert:
            return
        device = args.device
        embedding_dim, num_embeddings = self.entity_embeddings.embedding_dim, self.entity_embeddings.num_embeddings
        assert embedding_dim == 768, f"The entity_embedding_dim must 768 but entity_embedding_dim=={embedding_dim}"

        data_helper: MyDataHelper = kwargs['data_helper']
        bert_model = BertModel.from_pretrained('bert-base-uncased')
        bert_model.to(device)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        processed_entities_label = [
            "[CLS] {}".format(x) if x != '' else '' for x in data_helper.processed_entities_label]
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
                pre_embedding_mean, pre_embedding_mean_var, size=(num_embeddings-num_pre_emb, embedding_dim))


class KgStoryTransformer00(KgStoryTransformer):
    """KgStoryTransformer00

    * get head, relation, tail embedding.
    * weighted sum 3 embedding, and it is triple embedding.
    * get activated embedding by using Transformer.
    * after embedding activation for head, relation, tail.
    * pred head, relation, tail item by these embeddings.

    """
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
    """KgStoryTransformer01

    * Changed from 00.
    * --- ---
    * Original get_emb_head function: return entity embedding.
    * This model:  return ACTIVATE entity embedding by using feedfoward.
    * --- ---

    """

    def __init__(self, args, num_entity, num_relations, special_tokens, **kwargs):
        super(KgStoryTransformer01, self).__init__(args, num_entity, num_relations, special_tokens, **kwargs)
        embedding_dim = args.embedding_dim
        self.head_activate = Feedforward(embedding_dim, embedding_dim)

    def get_emb_head(self, x: torch.Tensor):
        return self.head_activate(self.entity_embeddings(x))


class KgStoryTransformer02(KgStoryTransformer00):
    """KgStoryTransformer02

    * Changed from 00.
    * --- ---
    * Original estimation method: vector distance
    * This model: onehot
    * --- ---

    """

    def __init__(self, args, num_entity, num_relations, special_tokens, **kwargs):
        super(KgStoryTransformer02, self).__init__(args, num_entity, num_relations, special_tokens, **kwargs)
        del self.head_maskdlm, self.relation_maskdlm, self.tail_maskdlm

        embedding_dim = args.embedding_dim

        self.head_maskdlm = Feedforward(embedding_dim, num_entity, dim_feedforward=embedding_dim, add_norm=False)
        self.relation_maskdlm = Feedforward(embedding_dim, num_relations, dim_feedforward=embedding_dim, add_norm=False)
        self.tail_maskdlm = Feedforward(embedding_dim, num_entity, dim_feedforward=embedding_dim, add_norm=False)

    def get_head_pred(self, x: torch.Tensor):
        return self.head_maskdlm(x)

    def get_relation_pred(self, x: torch.Tensor):
        return self.relation_maskdlm(x)

    def get_tail_pred(self, x: torch.Tensor):
        return self.tail_maskdlm(x)


class KgStoryTransformer0102(KgStoryTransformer01, KgStoryTransformer02):
    """KgStoryTransformer0102

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
        super(KgStoryTransformer0102, self).__init__(args, num_entity, num_relations, special_tokens, **kwargs)


class KgStoryTransformer03(KgStoryTransformer02):
    """KgStoryTransformer0102

    * Changed from 00.
    * --- ---
    * Original get_triple_embedding function: return sum of embeddings.
    * This model:  return cat and activate embeddings.
    * ---
    * Original estimation method: vector distance.
    * This model: onehot.
    * --- ---

    """

    def __init__(self, args, num_entity, num_relations, special_tokens, **kwargs):
        super(KgStoryTransformer03, self).__init__(args, num_entity, num_relations, special_tokens, **kwargs)
        del self.weight_head, self.weight_relation, self.weight_tail
        # get from args
        embedding_dim = args.embedding_dim
        entity_embedding_dim = args.entity_embedding_dim
        relation_embedding_dim = args.relation_embedding_dim
        # set new module
        self.input_activate = Feedforward(2 * entity_embedding_dim + relation_embedding_dim, embedding_dim)

    def get_triple_embedding(self, head, relation, tail):
        emb_head, emb_rel, emb_tail = self.get_emb_head(head), self.get_emb_relation(relation), self.get_emb_tail(tail)
        x = self.input_activate(torch.cat([emb_head, emb_rel, emb_tail], dim=2))
        return self.pe(x)


class KgStoryTransformer03preInit(KgStoryTransformer03, KgStoryTransformerLabelInit):
    """KgStoryTransformer03 and pre init by bert model.

    """
    
    def init(self, args, **kwargs):
        """init

        change entity embeddings by labels.

        """
        super(KgStoryTransformer03preInit, self).init(args, **kwargs)


if __name__ == '__main__':
    raise NotImplementedError()
    pass
