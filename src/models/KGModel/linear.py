#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Distmult

* linear models
"""
# noinspection PyUnresolvedReferences
from typing import List, Dict, Tuple, Optional, Union, Callable, Literal, Final, get_args
# pytorch
import torch
from torch import nn
from torch.nn import functional as F
# My abstract module
from models.KGModel.kg_model import KGE_ERE


class DistMult(KGE_ERE):
    def __init__(self, entity_embedding_dim, relation_embedding_dim, entity_num, relation_num):
        """init

        """
        super(DistMult, self).__init__(entity_embedding_dim, relation_embedding_dim, entity_num, relation_num, None)
        if entity_embedding_dim != relation_embedding_dim:
            raise ValueError("DistMult will not allow to separate entity_embedding_dim and relation_embedding_dim.")

    def init(self):
        """init

        """
        pass

    def forward(self, triple: torch.Tensor) -> torch.Tensor:
        tail_len = triple.shape[1] - 2
        head, relation, tail = torch.split(triple, [1, 1, tail_len], dim=1)

        head_emb = self.entity_embeddings(head)
        relation_emb = self.relation_embeddings(relation)
        tail_emb = self.entity_embeddings(tail)

        return (relation_emb * head_emb) * tail_emb


class Rescal(KGE_ERE):
    def __init__(self, entity_embedding_dim, relation_embedding_dim, entity_num, relation_num):
        """init

        """
        super(Rescal, self).__init__(entity_embedding_dim, relation_embedding_dim, entity_num, relation_num, None)
        if entity_embedding_dim != relation_embedding_dim:
            raise ValueError("DistMult will not allow to separate entity_embedding_dim and relation_embedding_dim.")
        del self.relation_embeddings
        self.relation_embeddings = torch.normal(
            0, 1, size=(relation_num, relation_embedding_dim, relation_embedding_dim), requires_grad=True)

    def init(self):
        """init

        """
        pass

    def forward(self, triple: torch.Tensor) -> torch.Tensor:
        """forward

        """
        tail_len = triple.shape[1] - 2
        head, relation, tail = torch.split(triple, [1, 1, tail_len], dim=1)

        head_emb = self.entity_embeddings(head)
        relation_emb = self.relation_embeddings[torch.flatten(relation)]
        tail_emb = self.entity_embeddings(tail)

        return head_emb * torch.matmul(relation_emb, tail_emb)
