#!/usr/bin/python
# -*- coding: utf-8 -*-
"""TransE, TransH, TransR

* translation models
"""
# noinspection PyUnresolvedReferences
from typing import List, Dict, Tuple, Optional, Union, Callable, Literal, Final, get_args
# pytorch
import torch
from torch import nn
from torch.nn import functional as F

# My abstract module
from models.KGModel.kg_model import KGE_ERE


class TransE(KGE_ERE):
    """ TransE

    """

    def __init__(self, entity_embedding_dim, relation_embedding_dim, entity_num, relation_num):
        """

        Args:
            entity_num(int): the number of entity.
            relation_num(int): リレーション数
            emb_dim(int): エンベディングの次元数
        """
        super(TransE, self).__init__(entity_embedding_dim, relation_embedding_dim, entity_num, relation_num, None)

        if entity_embedding_dim != relation_embedding_dim:
            raise ValueError("TransE will not allow to separate entity_embedding_dim and relation_embedding_dim.")
        self.emb_dim = entity_embedding_dim
        self.bias_head = torch.nn.Embedding(entity_num, 1, _weight=torch.zeros(entity_num, 1))
        self.bias_tail = torch.nn.Embedding(entity_num, 1, _weight=torch.zeros(entity_num, 1))

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

        head_emb = self.entity_embeddings(head)
        relation_emb = self.relation_embeddings(relation)
        tail_emb = self.entity_embeddings(tail)
        bias_head = self.bias_head(head).reshape(-1, 1)
        bias_tail = self.bias_tail(tail).reshape(-1, tail_len)
        # shape = [batch, tail_len, emb_dim]
        deviations: torch.Tensor = tail_emb - (head_emb + relation_emb)
        score = torch.linalg.norm(deviations, dim=2) + bias_head + bias_tail
        # score.shape = [batch, tail_len]
        return score


class TransH(KGE_ERE):
    """ TransE

    """

    def __init__(self, entity_embedding_dim, relation_embedding_dim, entity_num, relation_num):
        """

        Args:
            entity_num(int): the number of entity.
            relation_num(int): リレーション数
            emb_dim(int): エンベディングの次元数
        """
        super(TransH, self).__init__(entity_embedding_dim, relation_embedding_dim, entity_num, relation_num, None)

        if entity_embedding_dim != relation_embedding_dim:
            raise ValueError("TransH will not allow to separate entity_embedding_dim and relation_embedding_dim.")
        self.emb_dim = entity_embedding_dim
        self.wr_embeddings = torch.nn.Embedding(relation_num, self.emb_dim)
        self.bias_head = torch.nn.Embedding(entity_num, 1, _weight=torch.zeros(entity_num, 1))
        self.bias_tail = torch.nn.Embedding(entity_num, 1, _weight=torch.zeros(entity_num, 1))

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

        head_emb = self.entity_embeddings(head)
        relation_emb = self.relation_embeddings(relation)
        tail_emb = self.entity_embeddings(tail)
        emb_w_r = F.normalize(self.wr_embeddings(relation), dim=2)
        bias_head = self.bias_head(head).reshape(-1, 1)
        bias_tail = self.bias_tail(tail).reshape(-1, tail_len)
        #
        head_moved_emb = head_emb + emb_w_r * head_emb * emb_w_r
        tail_moved_emb = tail_emb + emb_w_r * tail_emb * emb_w_r
        # shape = [batch, tail_len, emb_dim]
        deviations: torch.Tensor = tail_moved_emb - (head_moved_emb + relation_emb)
        score = torch.linalg.norm(deviations, dim=2) + bias_head + bias_tail
        # score.shape = [batch, tail_len]
        return score


class TransR(KGE_ERE):
    """ TransE

    """

    def __init__(self, entity_embedding_dim, relation_embedding_dim, entity_num, relation_num):
        """

        Args:
            entity_num(int): the number of entity.
            relation_num(int): リレーション数
            emb_dim(int): エンベディングの次元数
        """
        super(TransR, self).__init__(entity_embedding_dim, relation_embedding_dim, entity_num, relation_num, None)
        self.W_r = nn.Linear(entity_embedding_dim, relation_embedding_dim, bias=False)
        self.bias_head = torch.nn.Embedding(entity_num, 1, _weight=torch.zeros(entity_num, 1))
        self.bias_tail = torch.nn.Embedding(entity_num, 1, _weight=torch.zeros(entity_num, 1))

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

        head_emb = self.entity_embeddings(head)
        relation_emb = self.relation_embeddings(relation)
        tail_emb = self.entity_embeddings(tail)
        bias_head = self.bias_head(head).reshape(-1, 1)
        bias_tail = self.bias_tail(tail).reshape(-1, tail_len)

        head_W_emb = self.W_r(head_emb)
        tail_W_emb = self.W_r(tail_emb)
        # shape = [batch, tail_len, entity_embedding_dim]
        deviations: torch.Tensor = tail_W_emb - (head_W_emb + relation_emb)
        score = torch.linalg.norm(deviations, dim=2) + bias_head + bias_tail
        # score.shape = [batch, tail_len]
        return score


def main():
    entity_num, relation_num, emb_dim = 2, 4, 8
    train_triple = torch.tensor([
        [0, 0, 0], [1, 1, 0], [0, 0, 0]
    ])  # batch*3
    batch_size = train_triple.shape[0]
    valid_triple = torch.cat((train_triple[:, :2], torch.arange(entity_num).repeat(batch_size, 1)), dim=1)

    _transe = TransE(emb_dim, emb_dim, entity_num, relation_num)
    _transh = TransH(emb_dim, emb_dim, entity_num, relation_num)
    _transr = TransR(emb_dim, emb_dim, entity_num, relation_num)
    with torch.no_grad():
        for model in (_transe, _transh, _transr):
            print(model.__class__)
            print(model(train_triple).shape)
            print(model(valid_triple).shape)


if __name__ == '__main__':
    main()
    pass
