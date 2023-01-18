#!/usr/bin/python
# -*- coding: utf-8 -*-
"""TransE だいたいこんな感じじゃないですかね? コード

"""
# noinspection PyUnresolvedReferences
from typing import List, Dict, Tuple, Optional, Union, Callable, Literal, Final, get_args
# pytorch
import torch
import torch.nn.functional as F

# 定数およびリテラル
MODEL_MODE_Literal = Literal['train.batch', 'test.batch']
TRAIN_MODE: Final = 'train.batch'
TEST_MODE: Final = 'test.batch'


# テストにのみ用いる簡単な関数
def all_same(*args):
    """
    return True if all item in args is same value.
    Args:
        *args:

    Returns:

    """
    tmp = args[0]
    for item in args[1:]:
        if item != tmp:
            return False
    return True


class TransE(torch.nn.Module):
    """ TransE

    """
    def __init__(self, entity_num, relation_num, emb_dim):
        """

        Args:
            entity_num(int): the number of entiy
            relation_num(int): リレーション数
            emb_dim(int): エンベディングの次元数
        """
        super(TransE, self).__init__()
        self.entity_num = entity_num
        self.relation_num = relation_num
        self.emb_dim = emb_dim
        self.emb_entity = torch.nn.Embedding(entity_num, emb_dim)
        self.emb_relation = torch.nn.Embedding(relation_num, emb_dim)
        self.bias_head = torch.nn.Embedding(entity_num, 1, _weight=torch.zeros(entity_num, 1))
        self.bias_tail = torch.nn.Embedding(entity_num, 1, _weight=torch.zeros(entity_num, 1))

    def forward(self, head, relation, tail, mode: MODEL_MODE_Literal):
        """

        Args:
            head(torch.Tensor): head tensor
            relation(torch.Tensor): relation tensor
            tail(torch.Tensor): tail tensor
            mode(MODEL_MODE): output mode

        Returns:
            torch.Tensor: score. 学習が進むと平均的に上がる
        """
        batch_size = head.shape[0]
        emb_dim = self.emb_dim

        emb_head: torch.Tensor = self.emb_entity(head)
        emb_relation: torch.Tensor = self.emb_relation(relation)
        emb_tail: torch.Tensor = self.emb_entity(tail)
        bias_head: torch.Tensor = self.bias_head(head)
        bias_tail: torch.Tensor = self.bias_tail(tail)

        if mode == TRAIN_MODE:
            assert all_same(emb_head.shape, emb_relation.shape, emb_tail.shape, (batch_size, emb_dim)), \
                "tensorサイズを確認してください. \n " \
                "emb_head.shape=>{}, emb_relation.shape=>{}, emb_tail.shape=>{}".format(
                    emb_head.shape, emb_relation.shape, emb_tail.shape)
            assert all_same(bias_head.shape, bias_tail.shape, (batch_size, 1)), \
                "tensorサイズを確認してください. \n " \
                "bias_head.shape=>{}, bias_tail.shape=>{}".format(bias_head.shape, bias_tail.shape)
            deviations = emb_tail - (emb_head + emb_relation)
            assert deviations.shape == (batch_size, emb_dim), \
                "deviations.shape={}, (batch_size, emb_dim)={}".format(deviations.shape, (batch_size, emb_dim))
            score = torch.linalg.norm(deviations, dim=1) + bias_head.reshape(batch_size) + bias_tail.reshape(batch_size)
            assert score.shape == (batch_size,), \
                "score.shape={}, (batch_size,)={}".format(score.shape, (batch_size,))
        elif mode == TEST_MODE:
            emb_tail_len = self.entity_num
            assert all_same(emb_head.shape, emb_relation.shape, (batch_size, emb_dim)) and \
                   emb_tail.shape == (emb_tail_len, emb_dim), \
                   "tensorサイズを確認してください. \n " \
                   "emb_head.shape=>{}, emb_relation.shape=>{}, emb_tail.shape=>{}".format(
                    emb_head.shape, emb_relation.shape, emb_tail.shape)
            assert bias_head.shape == (batch_size, 1) and bias_tail.shape == (emb_tail_len, 1), \
                "tensorサイズを確認してください. \n " \
                "bias_head.shape=>{}, bias_tail.shape=>{}".format(bias_head.shape, bias_tail.shape)

            emb_pred_tail = (emb_head + emb_relation).reshape(batch_size, 1, emb_dim)
            emb_true_tail = emb_tail.unsqueeze(0).repeat(batch_size, 1, 1)
            all_deviations = emb_true_tail - emb_pred_tail
            assert all_deviations.shape == (batch_size, emb_tail_len, emb_dim), \
                "all_deviations.shape={}, (batch_size, tail_len, emb_dim)={}".format(
                    all_deviations.shape, (batch_size, emb_tail_len, emb_dim))
            score = torch.linalg.norm(all_deviations, dim=2) + bias_head + bias_tail.reshape(1, emb_tail_len)
            assert score.shape == (batch_size, emb_tail_len), \
                "score.shape={}, (batch_size, entity_num)={}".format(score.shape, (batch_size, emb_tail_len))
        else:
            raise ValueError("mode=={}, しかしそのようなモードはありません.".format(mode))
            pass
        return score


def test01():
    entity_num, relation_num, emb_dim = 64, 32, 16
    batch_size, epoch, lr = 4, 10000, 1e-2
    # entities = ['entity{}'.format(i) for i in range(entity_num)]

    model = TransE(entity_num, relation_num, emb_dim)
    print("----- check model -----")
    print(model)

    # training
    tensor_triple = torch.tensor([
        [1, 0, 1], [1, 1, 2], [1, 2, 3], [1, 3, 4],
    ])  # batch*3
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def _train():
        mode = TRAIN_MODE
        model.train()
        # train
        batch = tensor_triple
        assert batch.shape == (batch_size, 3)
        opt.zero_grad()
        tensor_head = batch[:, 0]
        tensor_relation = batch[:, 1]
        tensor_tail = batch[:, 2]
        # print("----- input shape -----")
        # print(f"head.shape={tensor_head.shape}, relation.shape={tensor_relation.shape}, "
        #       f"tail.shape={tensor_tail.shape}")
        score = model(tensor_head, tensor_relation, tensor_tail, mode)
        # print("score.shape={}".format(score.shape))
        loss = F.logsigmoid(score)
        # print("loss={}".format(loss.tolist()))
        loss = -loss.mean()
        loss.backward()
        opt.step()
        # loss.backward とか
        # print("train complete")

    def _test():
        mode = TEST_MODE
        model.eval()
        # test
        batch = tensor_triple
        assert batch.shape == (batch_size, 3)
        tensor_head = batch[:, 0]
        tensor_relation = batch[:, 1]
        tensor_true_tail = batch[:, 2]
        score = model(tensor_head, tensor_relation, torch.arange(entity_num), mode)
        assert score.shape == (batch_size, entity_num)
        sorted_ = torch.argsort(score, dim=1, descending=True)
        for i in range(batch_size):
            for j in range(entity_num):
                # print("{}_{}={}".format(i, j, entities[sorted_[i][j]]))
                pass
        ranking = torch.nonzero(sorted_ == tensor_true_tail.reshape(batch_size, 1))[:, 1]

        for i in range(batch_size):
            # print("item{}_ranking={}".format(i, ranking[i]))
            pass
        print(ranking.tolist())

    for _epoch in range(epoch):
        _train()
        _test()


if __name__ == '__main__':
    test01()
    pass
