#!/usr/bin/python
# -*- coding: utf-8 -*-
"""utils related to pytorch
This module is the utils for pytorch basic operations and devices.
todo:

"""

import os
# noinspection PyUnresolvedReferences
import warnings
from logging import Logger
import random
# noinspection PyUnresolvedReferences
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
# noinspection PyUnresolvedReferences
from typing import List, Dict, Tuple, Optional, Callable, Union, Final, NamedTuple, Literal, cast, Iterable, TypeVar
# from utils.utils import none_count, is_same_len_in_list
from utils.progress_manager import ProgressHelper
from utils.utils import tqdm  # default tqdm or jupyter tqdm.

# region const value
INF: Final = float('inf')
ZERO_TENSOR: Final = torch.tensor(0)
ONE_TENSOR: Final = torch.tensor(1)

ZERO_FLOAT32_TENSOR: Final = torch.tensor(0., dtype=torch.float32)

ndarray_Tensor = Union[np.ndarray, torch.Tensor]
_T = TypeVar('_T')
_U = TypeVar('_U')


# endregion

# region functions related to device type
class _DeviceName:
    """
    const values for device name
    """
    CPU: Final[str] = 'cpu'
    CUDA: Final[str] = 'cuda'
    MPS: Final[str] = 'mps'

    @property
    def ALL_LIST(self) -> list:
        """
        list: the list of all usable device name.
        """
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return [self.CPU, self.CUDA, self.MPS]
        else:
            return [self.CPU, self.CUDA]

    @property
    def ALL_INFO(self) -> str:
        """
        str: all usable device name info.
        """
        return ', '.join(self.ALL_LIST)


DeviceName: Final[_DeviceName] = _DeviceName()
DeviceNameType: Final = Literal['cpu', 'cuda', 'mps']


def get_device(device_name: DeviceNameType, *, logger: Logger = None):
    assert device_name in DeviceName.ALL_LIST
    if device_name == DeviceName.CUDA:
        assert torch.cuda.is_available()
        logger.info("use gpu")
        return torch.device(DeviceName.CUDA)
    elif device_name == "mps":
        assert getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
        return torch.device("mps")
    else:
        return torch.device("cpu")
    pass


# endregion

# region functions related to gpu
def cuda_empty_cache():
    """torch.cuda.empty_cache()"""
    torch.cuda.empty_cache()


def force_cuda_empty_cache_after_function(func):
    """
    This is decorator.
    torch.cuda.empty_cache() after function.
    """

    def wrapper(*args, **kwargs):
        rev = func(*args, **kwargs)
        cuda_empty_cache()
        return rev

    return wrapper


def force_cuda_empty_cache_per_loop(iterable: Iterable[_T]):
    for x in iterable:
        yield x
        cuda_empty_cache()


class force_cpu(object):
    def __init__(self, model: nn.Module, device: torch.device, non_blocking=False):
        """
        Within with... , device is forced to be cpu.
        """
        self.model = model
        self.device = device
        self.non_blocking = non_blocking
        if not isinstance(device, torch.device):
            assert "'device' type is not 'torch.device'. "

    def __enter__(self):
        self.model.to('cpu', non_blocking=self.non_blocking)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.model.to(self.device)
        del self.model, self.device


def force_cpu_decorator(model, device, non_blocking=False):
    def _force_cpu_decorator(func):
        def wrapper(*args, **kwargs):
            with force_cpu(model, device, non_blocking=False):
                func(*args, **kwargs)
        return wrapper
    return _force_cpu_decorator


# endregion

# region functions related to load and save model.
def load_model(model: nn.Module, model_path: str, device: torch.device, *, delete_file=False):
    """
    load model function. if `delete_file' is True, delete file after loaded.
    Returns: model.
    """
    with force_cpu(model, device):
        model.load_state_dict(torch.load(model_path))
        if delete_file:
            os.remove(model_path)
    return model


def save_model(model: nn.Module, model_path: str, device: torch.device):
    """
    Save cpu model. After saving, the model device is returned to the input `device' value.
    Args:
        model:
        model_path:
        device:

    Returns:

    """
    device = device or model.d
    with force_cpu(model, device):
        torch.save(model.state_dict(), model_path)


# endregion

# region util functions.
def torch_fix_seed(seed: int = 42) -> None:
    """
    Fix the seed and expect reproducibility when using Pytorch.

    Args:
        seed (int): seed value

    Returns:
        None
    """
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True


def requires_grad_param_num(model: nn.Module):
    """
    Get the number of gradable params.
    Args:
        model (nn.Module): PyTorch model.

    Returns:
        The number of gradable params.

    """
    params = 0
    for p in model.parameters():
        if p.requires_grad:
            params += p.numel()
    return params


def all_same_shape(*args: torch.Tensor):
    _shape = args[0].shape
    for _tensor in args:
        assert type(_tensor) is torch.Tensor
        if not _tensor.shape == _shape: return False
    return True


class MM(nn.Module):
    """
    Only torch.mm. It's just that it can be visualized by making it a module.
    """

    def __init__(self):
        super(MM, self).__init__()

    @staticmethod
    def forward(x, y):
        return torch.mm(x, y)


class LossHelper:
    """
    This is helper of loss manage.
    """

    def __init__(self, update_min_d=1e-6, progress_helper: ProgressHelper = None, early_total_count=20):
        """
        Args:
            update_min_d (:obj:`float`, optional):
                If the updated value is smaller than this value, it is not considered updated.
            progress_helper (ProgressHelper): todo. write this.
            early_total_count (:obj:`int`, optional):
                if no updated count >= early_total_count,
        """
        self._loss_list = []
        self.min_loss = INF
        self.update_min_d = update_min_d
        self.not_update_count = 0
        self.progress_helper = progress_helper
        self.early_total_count = early_total_count
        progress_helper.add_key('early_count', total=early_total_count)

    def update(self, loss: Union[float, torch.Tensor]):
        """
        update per loss checked.
        Args:
            loss (float or torch.Tensor): loss param.

        Returns:
            True if min loss is update, else False
        """
        loss = loss.item() if type(loss) is torch.Tensor else loss
        self._loss_list.append(loss)
        rev = (self.min_loss - loss) > self.update_min_d
        if rev:
            self.not_update_count = 0
            self.progress_helper.reset_key('early_count')
            self.min_loss = loss
        else:
            self.not_update_count += 1
            self.progress_helper.update_key('early_count')
        return rev

    @property
    def is_early_stopping(self):
        """
        bool: True when the number of non-updates exceeds self.early_total_count.
        """
        if self.early_total_count <= 0:
            return False
        else:
            return self.not_update_count >= self.early_total_count

    @property
    def all_loss(self):
        """
        list[float]: The list of all loss
        """
        return self._loss_list[:]

# endregion
