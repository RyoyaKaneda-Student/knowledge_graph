import torch
import torch.nn as nn
from logging import Logger
from typing import List, Dict, Tuple, Optional, Callable, Union


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


class force_cpu(object):
    """ 自作のクラス """
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
