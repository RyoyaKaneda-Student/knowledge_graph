#!/usr/bin/python
# -*- coding: utf-8 -*-
# ========== python ==========
from logging import Logger
# noinspection PyUnresolvedReferences
from typing import List, Dict, Tuple, Optional, Union, Callable, Any, Final, Literal, Iterable, get_args
# noinspection PyUnresolvedReferences
import numpy as np
import h5py
from h5py import Group

from utils.log_helper import logger_is_optional


@logger_is_optional
def del_data_if_exist(group: Group, names: Iterable[str], *, logger: Logger = None):
    logger.debug("exist keys: {}".format(list(group.keys())))
    for name in names:
        if name in group.keys():
            del group[name]
            logger.debug("{} was deleted.".format(name))


def str_list_for_hdf5(p_object: Union[np.ndarray, Iterable, int, float], *args: Any, **kwargs: Any):
    return np.array(p_object, dtype=h5py.special_dtype(vlen=str), *args, **kwargs)


def read_one_item(path_, func: Callable[[h5py.File], Any]):
    with h5py.File(path_, 'r') as f:
        return func(f)


if __name__ == '__main__':
    pass
