import gc
import random
from typing import Tuple, List
from logging import Logger

from tqdm import tqdm as tqdm_default

tqdm = tqdm_default


def tqdm2notebook_tqdm():
    global tqdm
    from tqdm.notebook import tqdm as tqdm_notebook
    tqdm = tqdm_notebook


class FakeLogger(Logger):
    def __init__(self):
        pass

    def debug(self, msg: object, *args: object, ) -> None:
        print(msg)

    def info(self, msg: object, *args: object, ) -> None:
        print(msg)


def add_logger_if_logger_is_None(func):
    """
    this is decorator
    """

    def wrapper(*args, **kwargs):
        if 'logger' not in kwargs:
            logger = FakeLogger()
            kwargs['logger'] = logger
        return func(*args, **kwargs)

    return wrapper


def logger_is_optional(func):
    """
    this is decorator
    """

    def wrapper(*args, **kwargs):
        def _ignore_func(*_args, **_kwargs):
            return None

        if 'logger' not in kwargs:
            logger = object()
            logger.info = _ignore_func
            logger.debug = _ignore_func
            kwargs['logger'] = logger
        return func(*args, **kwargs)

    return wrapper


def random_num_choice(count_: int, min_: int, max_: int, not_choice_list_=None):
    list_ = [i for i in range(min_, max_) if i in not_choice_list_]
    rep = random.sample(list_, len(list_))[count_]
    assert len(rep) == count_
    return rep


def random_num_choice2(count_: int, min_: int, max_: int, not_choice_list_=None):
    list_ = [i for i in range(min_, max_) if i in not_choice_list_]
    rep = random.sample(list_, len(list_))[count_]
    assert len(rep) == count_
    return rep


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


def version_check(*args, logger=None):
    if logger is None: logger = FakeLogger()
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
    関数が終わり次第強制的にgcするためのデコレータ
    メモリやばい時のため
    """

    def wrapper(*args, **kwargs):
        rev = func(*args, **kwargs)
        force_gc()
        return rev

    return wrapper
