#!/usr/bin/python
# coding: UTF-8
"""Utils of Utils

* This file is used to store various useful functions.
* utils/utils, so there are various functions and classes.

Todo:
    * Improving.
    * Cleaning.
    * Erase what you don't need.
"""
import gc
from pathlib import Path
import random
# noinspection PyUnresolvedReferences
from typing import Tuple, List, Any, TypeVar, Generic, Iterable, Callable
from logging import Logger

from tqdm import tqdm as tqdm_default

tqdm = tqdm_default

_T = TypeVar('_T')
_U = TypeVar('_U')


def tqdm2notebook_tqdm():
    global tqdm
    from tqdm.notebook import tqdm as tqdm_notebook
    tqdm = tqdm_notebook


def get_pure_path(_path):
    """

    Args:
        _path:

    Returns:

    """
    return Path(_path).resolve().as_posix()


class FakeLogger(Logger):
    def __init__(self):
        pass

    def debug(self, msg: object, *args: object, ) -> None:
        print(msg)

    def info(self, msg: object, *args: object, ) -> None:
        print(msg)


def add_logger_if_logger_is_none(func):
    """
    this is decorator
    """

    def wrapper(*args, **kwargs):
        if 'logger' not in kwargs:
            logger = FakeLogger()
            kwargs['logger'] = logger
        return func(*args, **kwargs)

    return wrapper


def elapsed_time_str(seconds: float) -> str:
    """

    Args:
        seconds:

    Returns:

    """
    seconds = int(seconds + 0.5)  # 秒数を四捨五入
    h = seconds // 3600  # 時の取得
    m = (seconds - h * 3600) // 60  # 分の取得
    s = seconds - h * 3600 - m * 60  # 秒の取得
    return f"{h:02}:{m:02}:{s:02}"  # hh:mm:ss形式の文字列で返す


class EternalGenerator(Generic[_T]):
    def __init__(self, queue_generator: Callable[[], Iterable[_T]]):
        self.i = 0
        self.queue_generator = queue_generator
        self.queue_ = iter(queue_generator())

    def get_next(self, conditional=None):
        if conditional is None:
            conditional = lambda _: True
        while True:
            try:
                x = self.queue_.__next__()
                if conditional(x):
                    return x
            except StopIteration:
                self.queue_ = iter(self.queue_generator())


def dict_to_list(dict_: dict[int, _T]) -> list[_T]:
    for key in dict_:
        assert type(key) is int
    sorted_ = sorted(dict_.keys())
    assert sorted_[0] >= 0
    list_ = [None for _ in range(sorted_[-1])]
    for key, value in dict_.items():
        list_[key] = value
    return list_


def del_none(list_: list[Any]) -> list[Any]:
    return [x for x in list_ if x is not None]


def get_true_position_items_using_getter(
        items: list[Callable[[], _T]],
        bool_list: list[bool]
) -> list[_T]:
    """

    Args:
        items:
        bool_list:

    Returns:

    """
    rev: list[_T] = []
    for item_getter, is_true in zip(items, bool_list):
        if is_true: rev.append(item_getter())
    return rev


def get_true_position_items(
        items: list[_T],
        bool_list: list[bool]
) -> list[_T]:
    rev: list[_T] = []
    for item, is_true in zip(items, bool_list):
        if is_true: rev.append(item)
    return rev


def replace_list_value(list_: list[_T], before_after_list: Iterable[tuple[_T, _T]]) -> list[_T]:
    list_ = list_[:] # copy
    for before_, after_ in before_after_list:
        list_ = [v if v != before_ else after_ for v in list_]
    return list_


def remove_duplicate_as_same_order(list_):
    return sorted(set(list_), key=list_.index)
    # return list(dict.fromkeys(list_))


def true_count(*args):
    return len([True for item in args if item is True])


def none_count(*args):
    return len([True for x in args if x is not None])


def len_in_lists(*args) -> List:
    rep = []
    for item in args:
        rep.append(len(item))
    return rep


def is_same_item_in_list(*args) -> bool:
    rep = True
    for i in range(len(args) - 1):
        rep = rep and args[i] == args[i + 1]
        if not rep: break
    return rep


def is_same_len_in_list(*args) -> bool:
    return is_same_item_in_list(*len_in_lists(*args))


def optional_chaining1(func, x):
    if x is None:
        return None
    else:
        return func(x)


@add_logger_if_logger_is_none
def version_check(*args, logger):
    logger.debug("====================version check====================")
    for item in args:
        try:
            logger.debug(f"{item.__name__} == {item.__version__}")
        except AttributeError as e:
            logger.debug(f"{item.__name__} has no attribute 'version'.")

    logger.debug("====================version check====================")


def get_from_dict(dict_: dict, names: Tuple[str, ...]):
    return map(lambda x: dict_[x], names)


def force_gc():
    """
    強制的にgcするだけ
    Returns: None

    """
    gc.collect()


def force_gc_after_function(func):
    """
    Decorator to force gc as soon as the function finishes. Since I don't know the default spec.
    """

    def wrapper(*args, **kwargs):
        rev = func(*args, **kwargs)
        force_gc()
        return rev

    return wrapper
