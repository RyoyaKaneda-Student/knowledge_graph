import os

import torch
import torch.nn as nn
from logging import Logger
# noinspection PyUnresolvedReferences
from typing import List, Dict, Tuple, Optional, Callable, Union
import math

from utils.progress_manager import ProgressHelper

INF = float('inf')


def get_device(device_name, *, logger: Logger = None):
    if device_name == "cuda" and torch.cuda.is_available():
        logger.info("use gpu")
        return torch.device("cuda")
    elif device_name == "cuda" and not torch.cuda.is_available():
        logger.info("GPU使えないのに使おうとすんな. cpuで")
        return torch.device("cpu")
    elif device_name == "mps":
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            logger.info("mpsは使えませんのでcpu")
            return torch.device("cpu")
    else:
        return torch.device("cpu")


def cuda_empty_cache():
    torch.cuda.empty_cache()


# decorator
def force_cuda_empty_cache_after_function(func):
    def wrapper(*args, **kwargs):
        rev = func(*args, **kwargs)
        cuda_empty_cache()
        return rev

    return wrapper


def load_model(model: nn.Module, model_path: str, device, *, delete_file=False):
    with force_cpu(model, device):
        model.load_state_dict(torch.load(model_path))
        if delete_file:
            os.remove(model_path)
    return model


def save_model(model: nn.Module, model_path: str, device):
    with force_cpu(model, device):
        torch.save(model.state_dict(), model_path)


def decorate_loader(_loader, no_show_bar):
    if no_show_bar:
        return enumerate(_loader)
    else:
        from tqdm import tqdm
        return tqdm(enumerate(_loader), total=len(_loader), leave=False)


def onehot(items: Union[List[int], torch.Tensor], num_classes) -> torch.Tensor:
    if not torch.is_tensor(items):
        items = torch.tensor(items)
    return torch.nn.functional.one_hot(items, num_classes=num_classes)


def param_count_check(model):
    params = 0
    for p in model.parameters():
        if p.requires_grad:
            params += p.numel()
    return params


class force_cpu(object):
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        if not isinstance(device, torch.device):
            assert "'device' type is not 'torch.device'. "

    def __enter__(self):
        self.model.to('cpu')
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.model.to(self.device)
        del self.model, self.device


class PositionalEncoding(nn.Module):
    # batch first
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class LossHelper:
    def __init__(self, update_min_d=1e-6, progress_helper: ProgressHelper = None, early_total_count=20):
        self._loss_list = [INF]
        self.min_loss = INF
        self.update_min_d = update_min_d
        self.not_update_count = 0
        self.progress_helper = progress_helper
        progress_helper.add_key('early_count', total=early_total_count)

    # return True if loss is down
    def update(self, loss) -> bool:
        self._loss_list.append(loss)
        rev = (self.min_loss - loss) > self.update_min_d
        if rev:
            self.not_update_count = 0
            self.progress_helper.reset_key('early_count')
        else:
            self.not_update_count += 1
            self.progress_helper.update_key('early_count')
        self.min_loss = min(self.min_loss, loss)

    @property
    def all_loss(self):
        return self._loss_list[1:]
