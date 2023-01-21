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

        return torch.sum((relation_emb * head_emb) * tail_emb, dim=2)


class Rescal(KGE_ERE):
    def __init__(self, entity_embedding_dim, relation_embedding_dim, entity_num, relation_num):
        """init

        """
        super(Rescal, self).__init__(entity_embedding_dim, relation_embedding_dim, entity_num, relation_num, None)
        if entity_embedding_dim != relation_embedding_dim:
            raise ValueError("DistMult will not allow to separate entity_embedding_dim and relation_embedding_dim.")
        del self.relation_embeddings
        self.relation_embeddings = torch.normal(
            0., 1., size=(relation_num, relation_embedding_dim, relation_embedding_dim), requires_grad=True)

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
        relation_emb = self.relation_embeddings[torch.flatten(relation)].unsqueeze(1)
        tail_emb = self.entity_embeddings(tail).unsqueeze(3)

        return torch.sum(head_emb * torch.squeeze(torch.matmul(relation_emb, tail_emb), 3), dim=2)


class HolE(KGE_ERE):
    def __init__(self, entity_embedding_dim, relation_embedding_dim, entity_num, relation_num):
        """init

        """
        super(HolE, self).__init__(entity_embedding_dim, relation_embedding_dim, entity_num, relation_num, None)
        if entity_embedding_dim != relation_embedding_dim:
            raise ValueError("HolE will not allow to separate entity_embedding_dim and relation_embedding_dim.")
        del self.entity_embeddings

        self.head_embeddings = torch.nn.Embedding(entity_num, entity_embedding_dim, padding_idx=None)
        self.tail_embeddings = torch.nn.Embedding(entity_num, entity_embedding_dim, padding_idx=None)

    def init(self):
        """init

        """
        pass

    def forward(self, triple: torch.Tensor) -> torch.Tensor:
        """forward

        """
        tail_len = triple.shape[1] - 2
        head, relation, tail = torch.split(triple, [1, 1, tail_len], dim=1)

        head_emb = self.head_embeddings(head)
        relation_emb = self.relation_embeddings(relation)
        tail_emb = self.tail_embeddings(tail)
        return torch.sum(relation_emb * (head_emb * tail_emb), dim=2)


class ComplEx(KGE_ERE):
    """ TransE

    """

    def __init__(self, entity_embedding_dim, relation_embedding_dim, entity_num, relation_num):
        """

        Args:
            entity_num(int): the number of entity.
            relation_num(int): リレーション数
            emb_dim(int): エンベディングの次元数
        """
        super(ComplEx, self).__init__(entity_embedding_dim, relation_embedding_dim, entity_num, relation_num, None)
        if entity_embedding_dim != relation_embedding_dim:
            raise ValueError("ComplEx will not allow to separate entity_embedding_dim and relation_embedding_dim.")
        if entity_embedding_dim % 2 != 0:
            raise ValueError("ComplEx will not allow to separate entity_embedding_dim and relation_embedding_dim.")
        self.emb_dim = entity_embedding_dim // 2

    def init(self):
        """init

        """
        pass

    def forward(self, triple: torch.Tensor):
        """forward

        Args:
            triple(torch.Tensor): triple tensor.

        Returns:
            torch.Tensor: score. 学習が進むと平均的に上がる
        """
        tail_len = triple.shape[1] - 2
        head, relation, tail = torch.split(triple, [1, 1, tail_len], dim=1)

        head_re_emb, head_im_emb = torch.chunk(self.entity_embeddings(head), 2, dim=2)
        relation_re_emb, relation_im_emb = torch.chunk(self.relation_embeddings(relation), 2, dim=2)
        tail_re_emb, tail_im_emb = torch.chunk(self.relation_embeddings(tail), 2, dim=2)
        score = torch.sum(
            + head_re_emb * tail_re_emb * relation_re_emb
            + head_im_emb * tail_im_emb * relation_re_emb
            + head_re_emb * tail_im_emb * relation_im_emb
            - head_im_emb * tail_re_emb * relation_im_emb,
            dim=2)
        return score


def main():
    entity_num, relation_num, emb_dim = 2, 4, 8
    train_triple = torch.tensor([
        [0, 0, 0], [1, 1, 0], [0, 0, 0]
    ])  # batch*3
    batch_size = train_triple.shape[0]
    valid_triple = torch.cat((train_triple[:, :2], torch.arange(entity_num).repeat(batch_size, 1)), dim=1)

    _distmult = DistMult(emb_dim, emb_dim, entity_num, relation_num)
    _rescal = Rescal(emb_dim, emb_dim, entity_num, relation_num)
    _hole = HolE(emb_dim, emb_dim, entity_num, relation_num)
    _complex = ComplEx(emb_dim, emb_dim, entity_num, relation_num)

    with torch.no_grad():
        for model in (_distmult, _rescal, _hole, _complex):
            print(model.__class__)
            print(model(train_triple).shape)
            print(model(valid_triple).shape)


if __name__ == '__main__':
    main()
