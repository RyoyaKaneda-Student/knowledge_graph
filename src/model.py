# coding: UTF-8
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(os.path.join(PROJECT_DIR, 'src'))

import abc

import torch
from torch.nn import functional as F, Parameter
from torch.autograd import Variable

from torch.nn.init import xavier_normal_, xavier_uniform_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import TransformerEncoderLayer, TransformerEncoder

from utils.torch import PositionalEncoding


class KGEmbedding(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def init(self) -> None:
        raise NotImplementedError()


class Complex(torch.nn.Module):
    def __init__(self, args, num_entities, num_relations):
        super(Complex, self).__init__()
        self.num_entities = num_entities
        self.emb_e_real = torch.nn.Embedding(num_entities, args.embedding_dim, padding_idx=0)
        self.emb_e_img = torch.nn.Embedding(num_entities, args.embedding_dim, padding_idx=0)
        self.emb_rel_real = torch.nn.Embedding(num_relations, args.embedding_dim, padding_idx=0)
        self.emb_rel_img = torch.nn.Embedding(num_relations, args.embedding_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(args.input_drop)
        self.loss = torch.nn.BCELoss()

    def init(self):
        xavier_normal_(self.emb_e_real.weight.data)
        xavier_normal_(self.emb_e_img.weight.data)
        xavier_normal_(self.emb_rel_real.weight.data)
        xavier_normal_(self.emb_rel_img.weight.data)

    def forward(self, x):
        e1, rel = x
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


class DistMult(torch.nn.Module):
    def __init__(self, args, num_entities, num_relations):
        super(DistMult, self).__init__()
        self.emb_e = torch.nn.Embedding(num_entities, args.embedding_dim, padding_idx=0)
        self.emb_rel = torch.nn.Embedding(num_relations, args.embedding_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(args.input_drop)
        self.loss = torch.nn.BCELoss()

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)

    def forward(self, x):
        e1, rel = x
        e1_embedded = self.emb_e(e1)
        rel_embedded = self.emb_rel(rel)
        e1_embedded = e1_embedded.squeeze()
        rel_embedded = rel_embedded.squeeze()

        e1_embedded = self.inp_drop(e1_embedded)
        rel_embedded = self.inp_drop(rel_embedded)

        pred = torch.mm(e1_embedded * rel_embedded, self.emb_e.weight.transpose(1, 0))
        pred = torch.sigmoid(pred)

        return pred


class ConvE(torch.nn.Module):
    def __init__(self, args, num_entities, num_relations):
        super(ConvE, self).__init__()
        entity_special_num = args.entity_special_num
        relation_special_num = args.relation_special_num

        self.emb_e = torch.nn.Embedding(num_entities, args.embedding_dim, padding_idx=0)
        self.emb_rel = torch.nn.Embedding(num_relations, args.embedding_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(args.input_drop)
        self.hidden_drop = torch.nn.Dropout(args.hidden_drop)
        self.feature_map_drop = torch.nn.Dropout2d(args.feat_drop)
        self.loss = torch.nn.BCELoss()
        self.emb_dim1 = args.embedding_shape1
        self.emb_dim2 = args.embedding_dim // self.emb_dim1

        self.conv1 = torch.nn.Conv2d(1, 32, (3, 3), 1, 0, bias=args.use_bias)
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.bn2 = torch.nn.BatchNorm1d(args.embedding_dim)

        self.register_parameter('b', Parameter(torch.zeros(num_entities)))
        self.fc = torch.nn.Linear(args.hidden_size, args.embedding_dim)

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)

    def forward(self, x):
        e1, rel = x
        e1_embedded = self.emb_e(e1).view(-1, 1, self.emb_dim1, self.emb_dim2)
        rel_embedded = self.emb_rel(rel).view(-1, 1, self.emb_dim1, self.emb_dim2)

        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 2)

        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, self.emb_e.weight.transpose(1, 0))
        x += self.b.expand_as(x)
        pred = torch.sigmoid(x)

        return pred


class TransformerE(torch.nn.Module):
    def __init__(self, args, num_entities, num_relations):
        super(TransformerE, self).__init__()
        raise "this mode is not supported"
        embedding_dim = args.embedding_dim
        input_drop = args.input_drop
        hidden_drop = args.hidden_drop
        nhead = args.nhead
        transformer_drop = args.transformer_drop
        num_layers = 4

        self.emb_e = torch.nn.Embedding(num_entities, embedding_dim)
        self.emb_rel = torch.nn.Embedding(num_relations, embedding_dim)
        self.inp_drop = torch.nn.Dropout(input_drop)
        self.hidden_drop = torch.nn.Dropout(hidden_drop)
        self.loss = torch.nn.BCELoss()

        self.encoder_layer = TransformerEncoderLayer(
            d_model=embedding_dim, nhead=nhead, dropout=transformer_drop, batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers)
        self.fc = torch.nn.Linear(embedding_dim * 2, embedding_dim)
        self.bn2 = torch.nn.BatchNorm1d(embedding_dim)
        self.register_parameter('b', Parameter(torch.zeros(num_entities)))

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)

    def forward(self, x):
        e1, rel = x
        e1_embedded = self.emb_e(e1)
        rel_embedded = self.emb_rel(rel)
        x = torch.cat([e1_embedded, rel_embedded], dim=1)
        x = self.inp_drop(x)

        x = self.transformer_encoder(x)
        x = F.relu(x)
        x = torch.flatten(x, start_dim=1)

        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, self.emb_e.weight.transpose(1, 0))
        x += self.b.expand_as(x)
        pred = torch.sigmoid(x)

        return pred


class TransformerVer2E(torch.nn.Module):
    def __init__(self, args, num_entities, num_relations):
        super(TransformerVer2E, self).__init__()
        embedding_dim = args.embedding_dim
        input_drop = args.input_drop
        hidden_drop = args.hidden_drop
        nhead = args.nhead
        transformer_drop = args.transformer_drop
        num_layers = 4
        padding_token = args.padding_token
        cls_token = args.cls_token

        assert padding_token is not None
        assert cls_token is not None
        assert padding_token != cls_token
        assert args.entity_special_num >= 2

        self.padding_token_num: int = padding_token  # padding
        self.cls_token_num: int = cls_token  # cls
        self.padding_token: torch.Tensor = torch.tensor([padding_token])
        self.cls_token_num: torch.Tensor = torch.tensor([cls_token])

        self.emb_e = torch.nn.Embedding(num_entities, embedding_dim)
        self.emb_rel = torch.nn.Embedding(num_relations, embedding_dim)
        self.inp_drop = torch.nn.Dropout(input_drop)
        self.hidden_drop = torch.nn.Dropout(hidden_drop)
        self.loss = torch.nn.BCELoss()

        encoder_layer = TransformerEncoderLayer(
            d_model=embedding_dim, nhead=nhead, dropout=transformer_drop, batch_first=True
        )
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout=0.1)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers)
        self.fc = torch.nn.Linear(embedding_dim, embedding_dim)
        # self.bn2 = torch.nn.BatchNorm1d(embedding_dim)
        self.b = Parameter(torch.zeros(num_entities))

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)

    def get_cls_emb_e(self):
        return self.emb_e.weight[self.cls_token_num]

    def get_emb_e(self, e1):
        return self.emb_e(e1)

    def get_emb_rel(self, rel):
        return self.emb_rel(rel)

    def forward(self, x):
        e1, rel = x

        e1_embedded = self.get_emb_e(e1)
        rel_embedded = self.get_emb_rel(rel)
        cls_embedded = self.get_cls_emb_e().expand_as(e1_embedded)

        x = torch.cat([cls_embedded, e1_embedded, rel_embedded], dim=1)
        x = self.inp_drop(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
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


class TransformerVer3E(torch.nn.Module):
    def __init__(self, args, num_entities, num_relations, data_helper):
        from models.pytorch_geometric import make_geodata, separate_triples
        super(TransformerVer3E).__init__()
        embedding_dim = args.embedding_dim
        input_drop = args.input_drop
        hidden_drop = args.hidden_drop
        nhead = args.nhead
        transformer_drop = args.transformer_drop
        num_layers = 4

        padding_token_e = args.padding_token_e
        cls_token_e = args.cls_token_e
        padding_token_r = args.padding_token_r
        cls_token_r = args.cls_token_r
        self_loop_token_r = args.self_loop_token_r

        assert padding_token_e is not None
        assert cls_token_e is not None
        assert padding_token_e != cls_token_e
        assert args.entity_special_num >= 2

        assert padding_token_r is not None
        assert cls_token_r is not None
        assert self_loop_token_r is not None

        assert padding_token_r != cls_token_r
        assert cls_token_r != self_loop_token_r
        assert self_loop_token_r != padding_token_r
        assert args.entity_special_num >= 3

        self.padding_token_e = padding_token_e
        self.cls_token_e = cls_token_e
        self.padding_token_r = padding_token_r
        self.cls_token_r = cls_token_r
        self.self_loop_token_r = self_loop_token_r

        geo_data = make_geodata(data_helper=data_helper, is_del_reverse=False, is_add_self_loop=True,
                                self_loop_weight=self_loop_token_r)
        node2neighbor_node, node2neighbor_attr = separate_triples(geo_data, padding_value=padding_token_e)

        self.node2neighbor_node = node2neighbor_node
        self.node2neighbor_attr = node2neighbor_attr
        self.node2neighbor_pad = node2neighbor_node == 0
        self.neighbor_node_linear = torch.nn.Linear(embedding_dim, embedding_dim)
        self.neighbor_edge_linear = torch.nn.Linear(embedding_dim, embedding_dim)

        encoder_layer = TransformerEncoderLayer(
            d_model=embedding_dim, nhead=nhead, dropout=transformer_drop, batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers)

        self.emb_e = torch.nn.Embedding(num_entities, embedding_dim)
        self.emb_e.weight[padding_token_e].requires_grad = False
        self.emb_r.weight[padding_token_r].requires_grad = False
        self.emb_rel = torch.nn.Embedding(num_relations, embedding_dim)

    def init(self):
        pass

    def get_cls_emb_e(self):
        return self.emb_e.weight[self.cls_token_e]

    def get_cls_emb_r(self):
        return self.emb_rel.weight[self.cls_token_r]

    def get_emb_e(self, e1):
        return self.emb_e(e1)

    def get_emb_rel(self, rel):
        return self.emb_rel(rel)

    def forward(self, x):
        e1, rel = x
        neighbor_node = self.node2neighbor_node[e1]
        neighbor_edge = self.node2neighbor_attr[e1]

        emb_e = self.get_emb_e(neighbor_node)
        neighbor_edge = self.get_emb_e(neighbor_node)

        self.transformer_encoder.forward(x, )

        return pred


# Add your own model here


class MyModel(torch.nn.Module):
    def __init__(self, args, num_entities, num_relations, nhead=8):
        super(DistMult, self).__init__()
        self.emb_e = torch.nn.Embedding(num_entities, args.embedding_dim, padding_idx=0)
        self.emb_rel = torch.nn.Embedding(num_relations, args.embedding_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(args.input_drop)
        self.loss = torch.nn.BCELoss()
        encoder_layer = TransformerEncoderLayer(d_model=args.embedding_dim, nhead=8, dropout=0.1, batch_first=True)
        transformer_encoder = TransformerEncoder(encoder_layer, 4)

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)

    def forward(self, x):
        e1, rel = x
        e1_embedded = self.emb_e(e1)
        rel_embedded = self.emb_rel(rel)

        # Add your model function here
        # The model function should operate on the embeddings e1 and rel
        # and output scores for all entities (you will need a projection layer
        # with output size num_relations (from constructor above)

        output = None
        # generate output scores here
        prediction = torch.sigmoid(output)

        return prediction
