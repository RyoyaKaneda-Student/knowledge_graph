# coding: UTF-8
import os
import sys
from pathlib import Path

# noinspection PyUnresolvedReferences
from typing import List, Dict, Tuple, Optional, Union, Callable, Final

import abc

import torch
from torch.nn import functional as F, Parameter

from torch.nn.init import xavier_normal_
# from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import TransformerEncoderLayer, TransformerEncoder, Softmax

from utils.torch import MM

from models.utilModules.tranformer import PositionalEncoding
from models.utilModules.mlp_mixer import MlpMixer, MlpMixerLayer
from models.datasets.data_helper import SpecialPaddingTokens

HEAD: Final = 'head'
RELATION: Final = 'relation'
TAIL: Final = 'tail'


class KGE_ERTails(torch.nn.Module, metaclass=abc.ABCMeta):
    """The abstract model of knowledge graph.

    * This model's input is batch * [head, relation].
    * This model return all tail prediction as score.
    """

    def __init__(self, embedding_dim, num_entities, num_relations, special_tokens: Optional[SpecialPaddingTokens]):
        """__init__

        Args:
            embedding_dim(int): The dimensions of embedding.
            num_entities(int): The number of entities.
            num_relations(int): The number of relations.
            special_tokens(Optional[SpecialPaddingTokens]): SpecialTokens
        """
        super().__init__()
        if special_tokens is not None:
            padding_token_e, padding_token_r = special_tokens.padding_token_e, special_tokens.padding_token_r
        else:
            padding_token_e, padding_token_r = None, None

        self.emb_e = torch.nn.Embedding(num_entities, embedding_dim, padding_idx=padding_token_e)
        self.emb_rel = torch.nn.Embedding(num_relations, embedding_dim, padding_idx=padding_token_r)

    @abc.abstractmethod
    def init(self) -> None:
        """init parameters.

        """
        pass

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """input is batch * [entity, relation], return all tail prediction as score.

        Args:
            x(torch.Tensor): The tuple of entity and relation. The shape is (batch * 2).

        Returns:
            torch.Tensor: The shape is (batch * entity_size).

        """
        raise NotImplementedError()


class KGE_ERE(torch.nn.Module, metaclass=abc.ABCMeta):
    """The abstract model of knowledge graph.

    * This model's input is batch * [head, relation, tail].
    * This model return prediction as score.

    """

    def __init__(self, embedding_dim, num_entities, num_relations, padding_token_e, padding_token_r):
        """
        Args:
            embedding_dim: エンべディングの次数
            num_entities: エンべディングの個数
            num_relations: リレーションの個数
            padding_token_e: パディングのトークンのid (エンべディング)
            padding_token_r: パディングのトークンのid (リレーション)
        """
        super().__init__()
        self.emb_e = torch.nn.Embedding(num_entities, embedding_dim, padding_idx=padding_token_e)
        self.emb_rel = torch.nn.Embedding(num_relations, embedding_dim, padding_idx=padding_token_r)

    @abc.abstractmethod
    def init(self) -> None:
        pass

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x(torch.Tensor): x.shape is (batch * 3).

        Returns:
            torch.Tensor: The shape is (batch * entity_size).

        """
        pass


class Complex(KGE_ERTails):
    """Complex. This is all tail version.

    * This is almost same as this urls<https://github.com/TimDettmers/ConvE>
    * "Convolutional 2D Knowledge Graph Embeddings"

    """
    def __init__(self, args, num_entities, num_relations, special_tokens):
        super(Complex, self).__init__(args.embedding_dim, num_entities, num_relations, special_tokens)
        del self.emb_e, self.emb_rel
        embedding_dim = args.embedding_dim
        padding_idx_e, padding_idx_r = special_tokens.padding_token_e, special_tokens.padding_token_r

        self.emb_e_real = torch.nn.Embedding(num_entities, embedding_dim, padding_idx=padding_idx_e)
        self.emb_e_img = torch.nn.Embedding(num_entities, embedding_dim, padding_idx=padding_idx_e)
        self.emb_rel_real = torch.nn.Embedding(num_relations, embedding_dim, padding_idx=padding_idx_r)
        self.emb_rel_img = torch.nn.Embedding(num_relations, embedding_dim, padding_idx=padding_idx_r)
        self.inp_drop = torch.nn.Dropout(args.input_drop)
        self.loss = torch.nn.BCELoss()

    def init(self):
        """init embeddings.

        """
        xavier_normal_(self.emb_e_real.weight.data)
        xavier_normal_(self.emb_e_img.weight.data)
        xavier_normal_(self.emb_rel_real.weight.data)
        xavier_normal_(self.emb_rel_img.weight.data)

    def forward(self, x):
        """Forward function.

        """
        e1, rel = torch.split(x, 1, dim=1)
        e1_embedded_real = self.emb_e_real(e1).squeeze()
        rel_embedded_real = self.emb_rel_real(rel).squeeze()
        e1_embedded_img = self.emb_e_img(e1).squeeze()
        rel_embedded_img = self.emb_rel_img(rel).squeeze()

        e1_embedded_real = self.inp_drop(e1_embedded_real)
        rel_embedded_real = self.inp_drop(rel_embedded_real)
        e1_embedded_img = self.inp_drop(e1_embedded_img)
        rel_embedded_img = self.inp_drop(rel_embedded_img)

        # complex space bilinear product (equivalent to HolE)
        realrealreal = torch.mm(e1_embedded_real * rel_embedded_real, self.emb_e_real.weight.transpose(1, 0))
        realimgimg = torch.mm(e1_embedded_real * rel_embedded_img, self.emb_e_img.weight.transpose(1, 0))
        imgrealimg = torch.mm(e1_embedded_img * rel_embedded_real, self.emb_e_img.weight.transpose(1, 0))
        imgimgreal = torch.mm(e1_embedded_img * rel_embedded_img, self.emb_e_real.weight.transpose(1, 0))
        pred = realrealreal + realimgimg + imgrealimg - imgimgreal
        pred = torch.sigmoid(pred)

        return pred


class DistMult(KGE_ERTails):
    """DistMult. This is all tail version.

    * This is almost same as this urls<https://github.com/TimDettmers/ConvE>
    * "Convolutional 2D Knowledge Graph Embeddings"

    """
    def __init__(self, args, num_entities, num_relations, special_tokens):
        super(DistMult, self).__init__(args.embedding_dim, num_entities, num_relations, special_tokens)
        self.inp_drop = torch.nn.Dropout(args.input_drop)
        self.loss = torch.nn.BCELoss()

    def init(self):
        """init embeddings.

        """
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)

    def forward(self, x):
        """Forward function.

        """
        e1, rel = torch.split(x, 1, dim=1)
        e1_embedded = self.emb_e(e1)
        rel_embedded = self.emb_rel(rel)
        e1_embedded = e1_embedded.squeeze()
        rel_embedded = rel_embedded.squeeze()

        e1_embedded = self.inp_drop(e1_embedded)
        rel_embedded = self.inp_drop(rel_embedded)

        pred = torch.mm(e1_embedded * rel_embedded, self.emb_e.weight.transpose(1, 0))
        pred = torch.sigmoid(pred)

        return pred


class TransformerVer2E(KGE_ERTails):
    def __init__(self, args, num_entities, num_relations, special_tokens, **kwargs):
        super(TransformerVer2E, self).__init__(args.embedding_dim, num_entities, num_relations, special_tokens)
        embedding_dim = args.embedding_dim
        input_drop = args.input_drop
        hidden_drop = args.hidden_drop
        nhead = args.nhead
        position_encoder_drop = args.position_encoder_drop
        transformer_drop = args.transformer_drop
        dim_feedforward = args.dim_feedforward
        num_layers = args.num_layers
        padding_token = args.padding_token_h
        cls_token = args.cls_token_e

        assert padding_token is not None
        assert cls_token is not None
        assert padding_token != cls_token
        assert args.entity_special_num >= 2

        self.register_buffer('padding_token', torch.tensor([padding_token]))
        self.register_buffer('cls_token_num', torch.tensor([cls_token]))

        self.loss = torch.nn.BCELoss()

        encoder_layer = TransformerEncoderLayer(
            d_model=embedding_dim, nhead=nhead, dropout=transformer_drop, batch_first=True,
            dim_feedforward=dim_feedforward,
            activation=F.gelu, norm_first=True
        )

        self.inp_drop = torch.nn.Dropout(input_drop)
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout=position_encoder_drop, max_len=3)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers)
        self.hidden_drop1 = torch.nn.Dropout(hidden_drop)
        self.hidden_drop2 = torch.nn.Dropout(hidden_drop)
        self.bn01 = torch.nn.BatchNorm1d(embedding_dim)
        self.activate1 = torch.nn.GELU()

        self.fc1 = torch.nn.Linear(embedding_dim, embedding_dim * 4)
        self.activate2 = torch.nn.GELU()
        self.fc2 = torch.nn.Linear(embedding_dim * 4, embedding_dim)


        self.bn02 = torch.nn.BatchNorm1d(embedding_dim)
        self.activate3 = torch.nn.GELU()
        self.mm = MM()
        self.b = Parameter(torch.zeros(num_entities))
        self.sigmoid1 = torch.nn.Sigmoid()

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)

    def get_cls_emb_e(self) -> torch.Tensor:
        return self.emb_e(self.cls_token_num)

    def get_emb_e(self, e1) -> torch.Tensor:
        return self.emb_e(e1)

    def get_emb_rel(self, rel) -> torch.Tensor:
        return self.emb_rel(rel)

    def encoding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x[:, 0]  # cls
        return x

    def mlp_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.hidden_drop1(x)
        x = self.activate2(x)
        x = self.fc2(x)
        return x

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        e1, rel = x
        batch_size = e1.shape[0]
        # embedding
        e1_embedded = self.get_emb_e(e1)
        rel_embedded = self.get_emb_rel(rel)
        cls_embedded = self.get_cls_emb_e().repeat(batch_size, 1).view((batch_size, 1, -1))
        x = torch.cat([cls_embedded, e1_embedded, rel_embedded], dim=1)
        x = self.inp_drop(x)
        # transformer
        x = self.encoding(x)
        x = self.bn01(x)
        x = self.activate1(x)
        # mlp
        x = self.mlp_block(x)
        x = self.hidden_drop2(x)
        x = self.bn02(x)
        x = self.activate3(x)
        # check tail embedding
        x = self.mm(x, self.emb_e.weight.transpose(1, 0))
        x += self.b.expand_as(x)
        pred = self.sigmoid1(x)

        return pred


class TransformerVer3E(KGE_ERTails):
    def __init__(self, args, num_entities, num_relations, data_helper, **kwargs):
        from models.pytorch_geometric import make_geodata, separate_triples

        embedding_dim = args.embedding_dim
        input_drop = args.input_drop
        hidden_drop = args.hidden_drop
        nhead = args.nhead
        transformer_drop = args.transformer_drop
        num_layers = args.num_layers

        padding_token_e = args.padding_token_h
        cls_token_e = args.cls_token_e
        mask_token_e = args.mask_token_e
        padding_token_r = args.padding_token_r
        cls_token_r = args.cls_token_r
        self_loop_token_r = args.self_loop_token_r

        assert padding_token_e is not None
        assert cls_token_e is not None
        assert padding_token_e != cls_token_e
        assert args.entity_special_num >= 3

        assert padding_token_r is not None
        assert cls_token_r is not None
        assert self_loop_token_r is not None
        assert padding_token_r != cls_token_r
        assert cls_token_r != self_loop_token_r
        assert self_loop_token_r != padding_token_r
        assert args.relation_special_num >= 3

        super(TransformerVer3E, self).__init__(
            embedding_dim, num_entities, num_relations, padding_token_e, padding_token_r)

        self.padding_token_e = padding_token_e
        self.cls_token_e = cls_token_e
        self.mask_token_e = mask_token_e
        self.padding_token_r = padding_token_r
        self.cls_token_r = cls_token_r
        self.self_loop_token_r = self_loop_token_r

        self.num_entities = num_entities
        self.num_relations = num_relations
        self.loss = torch.nn.BCELoss()

        geo_data = make_geodata(data_helper=data_helper, is_del_reverse=False, is_add_self_loop=False,
                                logger=kwargs['logger'] if 'logger' in kwargs else None)

        node2neighbor_node, node2neighbor_attr = separate_triples(
            geo_data, padding_value=padding_token_e, self_loop_value=self_loop_token_r,
            logger=kwargs['logger'] if 'logger' in kwargs else None
        )

        self.node2neighbor_node = node2neighbor_node
        self.node2neighbor_attr = node2neighbor_attr
        self.node2neighbor_pad = node2neighbor_node == 0
        # self.neighbor_node_linear = torch.nn.Linear(embedding_dim, embedding_dim)
        # self.neighbor_edge_linear = torch.nn.Linear(embedding_dim, embedding_dim)

        encoder_layer = TransformerEncoderLayer(
            d_model=embedding_dim, nhead=nhead, dropout=transformer_drop, batch_first=True
        )

        self.inp_drop = torch.nn.Dropout(input_drop)
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout=0)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers)

        self.Softmax_dim2 = Softmax(dim=2)

        self.fc = torch.nn.Linear(embedding_dim, embedding_dim)
        self.hidden_drop = torch.nn.Dropout(hidden_drop)
        # self.bn2 = torch.nn.BatchNorm1d(embedding_dim)
        self.b = Parameter(torch.zeros(num_entities))

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)

    def get_cls_emb_e(self):
        return self.emb_e.weight[self.cls_token_e]

    def get_cls_emb_r(self):
        return self.emb_rel.weight[self.cls_token_r]

    def get_emb_e(self, e1):
        return self.emb_e(e1)

    def get_emb_rel(self, rel):
        return self.emb_rel(rel)

    def add_cls(self, _tensor, batch_size, cls_token):
        device = self.device
        return torch.cat([torch.full((batch_size, 1), cls_token, dtype=device), _tensor], dim=1)

    def combination_node_attr(self, neighbor_node, neighbor_attr):
        x = neighbor_node + 1.0 * neighbor_attr
        return x

    def forward(self, x):
        e1, rel = x
        device = rel.device
        batch_size = e1.shape[0]

        cls_token_e, cls_token_r = self.cls_token_e, self.cls_token_r

        neighbor_node = self.node2neighbor_node[e1].view(batch_size, -1).to(device)
        neighbor_attr = self.node2neighbor_attr[e1].view(batch_size, -1).to(device)
        neighbor_pad = self.node2neighbor_pad[e1].view(batch_size, -1).to(device)

        neighbor_node[(neighbor_attr == rel)] = self.mask_token_e

        neighbor_node = self.add_cls(neighbor_node, batch_size, cls_token_e)
        neighbor_attr = self.add_cls(neighbor_attr, batch_size, cls_token_r)
        neighbor_pad = self.add_cls(neighbor_pad, batch_size, False)

        neighbor_node = self.get_emb_e(neighbor_node)
        neighbor_attr = self.get_emb_rel(neighbor_attr)

        neighbor_node = self.Softmax_dim2(neighbor_node)
        neighbor_attr = self.Softmax_dim2(neighbor_attr)

        # 重いので
        neighbor_node = neighbor_node[:, :20]
        neighbor_attr = neighbor_attr[:, :20]
        neighbor_pad = neighbor_pad[:, :20]
        #

        x = self.combination_node_attr(neighbor_node, neighbor_attr)

        x = self.inp_drop(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder.forward(x, src_key_padding_mask=neighbor_pad
                                             ) if self.training else self.transformer_encoder.forward(x)
        x = x[:, 0]  # cls
        x = F.relu(x)
        x = self.fc(x)
        x = self.hidden_drop(x)
        # x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, self.emb_e.weight.transpose(1, 0))
        x += self.b.expand_as(x)
        pred = torch.sigmoid(x)
        return pred


class TransformerVer3E_1(TransformerVer3E):
    def __init__(self, args, num_entities, num_relations, data_helper, **kwargs):
        super(TransformerVer3E_1, self).__init__(args, num_entities, num_relations, data_helper, **kwargs)
        del self.pos_encoder, self.transformer_encoder, self.fc

        embedding_dim = args.embedding_dim
        nhead = args.nhead
        transformer_drop = args.transformer_drop
        num_layers = 4

        encoder_layer = TransformerEncoderLayer(
            d_model=embedding_dim * 2, nhead=nhead, dropout=transformer_drop, batch_first=True
        )
        self.pos_encoder = PositionalEncoding(embedding_dim * 2, dropout=0)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers)

        self.fc = torch.nn.Linear(embedding_dim * 2, embedding_dim)

    # override
    def combination_node_attr(self, neighbor_node, neighbor_attr):
        x = torch.concat((neighbor_node, neighbor_attr), dim=2)
        return x


class MlpMixE(TransformerVer2E):
    def __init__(self, args, num_entities, num_relations, **kwargs):
        super(MlpMixE, self).__init__(args, num_entities, num_relations, **kwargs)
        del self.transformer_encoder
        embedding_dim = args.embedding_dim
        num_layers = args.num_layers
        self.mlp_mixer = MlpMixer(MlpMixerLayer(embedding_dim, dim_feedforward=embedding_dim * 2), num_layers)

    # override
    def encoding(self, x: torch.Tensor):
        x = self.pos_encoder(x)
        x = self.mlp_mixer(x)
        x = x[:, 0]  # cls
        return x


def main():
    pass


if __name__ == '__main__':
    main()
