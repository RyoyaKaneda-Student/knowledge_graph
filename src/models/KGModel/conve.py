# coding: UTF-8
import os
import sys
from pathlib import Path
# noinspection PyUnresolvedReferences
from typing import List, Dict, Tuple, Optional, Union, Callable

import abc

import torch
from torch.nn import functional as F, Parameter

from torch.nn.init import xavier_normal_

from models.KGModel.kg_model import KGE_ERTails


class ConvE(KGE_ERTails):
    """ConvE

    * This is almost same as  this urls<https://github.com/TimDettmers/ConvE>
    * "Convolutional 2D Knowledge Graph Embeddings"


    """
    def __init__(self, args, entity_num, relation_num, special_tokens):
        super(ConvE, self).__init__(args.embedding_dim, entity_num, relation_num, special_tokens)
        embedding_dim = args.embedding_dim
        input_drop = args.input_drop
        hidden_drop = args.hidden_drop
        feature_map_drop = args.feat_drop
        embedding_shape1 = args.embedding_shape1
        use_bias = args.use_bias
        hidden_size = args.hidden_size

        self.inp_drop = torch.nn.Dropout(input_drop)
        self.hidden_drop = torch.nn.Dropout(hidden_drop)
        self.feature_map_drop = torch.nn.Dropout2d(feature_map_drop)
        self.loss = torch.nn.BCELoss()
        self.emb_dim1 = embedding_shape1
        self.emb_dim2 = embedding_dim // embedding_shape1

        self.conv1 = torch.nn.Conv2d(1, 32, (3, 3), 1, 0, bias=use_bias)
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.bn2 = torch.nn.BatchNorm1d(embedding_dim)

        self.b = Parameter(torch.zeros(entity_num))
        self.fc = torch.nn.Linear(hidden_size, embedding_dim)

    def init(self):
        """init weight

        Returns:

        """
        xavier_normal_(self.entity_embeddings.weight.data)
        xavier_normal_(self.relation_embeddings.weight.data)

    def forward(self, x):
        """Forward function

        """
        e1, rel = torch.split(x, 1, dim=1)
        e1_embedded = self.entity_embeddings(e1).view(-1, 1, self.emb_dim1, self.emb_dim2)
        rel_embedded = self.relation_embeddings(rel).view(-1, 1, self.emb_dim1, self.emb_dim2)

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
        x = torch.mm(x, self.entity_embeddings.weight.transpose(1, 0))
        x += self.b.expand_as(x)
        pred = torch.sigmoid(x)

        return pred
