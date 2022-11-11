#!/usr/bin/python
# -*- coding: utf-8 -*-
# ========== python ==========
from logging import Logger
# noinspection PyUnresolvedReferences
from typing import List, Dict, Tuple, Optional, Union, Callable, Final, Literal, Iterable, get_args
# noinspection PyUnresolvedReferences
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


if __name__ == '__main__':
    pass
