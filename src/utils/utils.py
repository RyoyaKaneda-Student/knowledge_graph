import gc
from typing import Tuple
from logging import Logger


class FakeLogger:
    def debug(self, msg: object, *args: object, ) -> None:
        print(msg)

    def info(self, msg: object, *args: object, ) -> None:
        print(msg)


def version_check(*args, logger=None):
    if logger is None: logger = FakeLogger()
    logger.debug("====================version check====================")
    for item in args:
        logger.debug(f"{item.__name__} == {item.__version__}")
    logger.debug("====================version check====================")


def get_from_dict(dict_: dict, names: Tuple[str, ...]):
    return map(lambda x: dict_[x], names)


def force_gc():
    gc.collect()


# decorator
def force_gc_after_function(func):
    def wrapper(*args, **kwargs):
        rev = func(*args, **kwargs)
        force_gc()
        return rev

    return wrapper
