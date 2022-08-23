import torch
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
