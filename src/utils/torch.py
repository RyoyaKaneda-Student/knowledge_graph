import os
import warnings
import random
import numpy as np

import torch
import torch.nn as nn
from logging import Logger
# noinspection PyUnresolvedReferences
from typing import List, Dict, Tuple, Optional, Callable, Union, Final, NamedTuple
import math

from utils.progress_manager import ProgressHelper
from utils.utils import none_count, is_same_len_in_list

INF = float('inf')

ndarray_Tensor = Union[np.ndarray, torch.Tensor]


class _DeviceName(NamedTuple):
    CPU: str = 'cpu'
    CUDA: str = 'cuda'
    MPS: str = 'mps'

    @classmethod
    def ALL_LIST(cls):
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return [cls.CPU, cls.CUDA, cls.MPS]
        else:
            return [cls.CPU, cls.CUDA]


DeviceName = _DeviceName()


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


def torch_fix_seed(seed=42):
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True


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
        from utils.utils import tqdm
        return tqdm(enumerate(_loader), total=len(_loader), leave=False)


def onehot(items: Union[List[int], torch.Tensor], num_classes) -> torch.Tensor:
    if not torch.is_tensor(items):
        items = torch.tensor(items)
    return torch.nn.functional.one_hot(items, num_classes=num_classes)


def insert_to_front_of_all_batch(tensor_, full_value):
    return torch.cat(torch.full((len(tensor_), 1), tensor_))


def param_count_check(model):
    params = 0
    for p in model.parameters():
        if p.requires_grad:
            params += p.numel()
    return params


def random_choice(x: torch.Tensor, n: Union[torch.Tensor, int], filter_: Optional[torch.Tensor] = None):
    indices = random_indices_choice(x, n, filter_)
    return torch.index_select(x, 0, indices)


def onehot_target(target_num, target_len, len_, dtype: torch.dtype = None):
    return torch.tensor([1 if i == target_num else 0 for i in range(target_len)], dtype=dtype).repeat(len_)


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
    """ PositionalEncoding
    batch first
    """

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
        x = x + self.pe[:, :x.size(1)]
        x = self.dropout(x)
        return x


class MM(nn.Module):
    """ PositionalEncoding
    batch first
    """

    def __init__(self):
        super(MM, self).__init__()

    def forward(self, x, y):
        return torch.mm(x, y)


class LossHelper:
    def __init__(self, update_min_d=1e-6, progress_helper: ProgressHelper = None, early_total_count=20):
        self._loss_list = [INF]
        self.min_loss = INF
        self.update_min_d = update_min_d
        self.not_update_count = 0
        self.progress_helper = progress_helper
        progress_helper.add_key('early_count', total=early_total_count)

    def update(self, loss):
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
