#!/usr/bin/python
# -*- coding: utf-8 -*-
"""

"""
from typing import Any

from torch.utils.tensorboard import SummaryWriter
import h5py
import pickle


class ResultPerEpoch:
    def __init__(self, keywords: list):
        self._dict: dict = {}
        for key in keywords:
            self._dict[key] = []
        self._completed_epoch_num = -1
        self._epoch = -1

    def start_epoch(self):
        self._epoch += 1
        [self._dict[key].append(None) for key in self._dict.keys()]
        return self._epoch

    def complete_epoch(self):
        self._completed_epoch_num = self._epoch

    def write(self, key: str, value: Any):
        self._dict[key][self._epoch] = value

    def write_all(self, key_value):
        for key, value in key_value.items():
            self.write(key, value)

    def save(self, file_):
        pass

    @classmethod
    def load_from(cls, file_):
        self = cls([])
        with open(file_, "rb") as f:
            dict_ = pickle.load(f)
        self._dict = dict_
        return dict_
