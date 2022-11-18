# noinspection PyUnresolvedReferences
from typing import (
    List, Dict, Tuple, Callable, Iterable,
    Final, Literal, Optional, Union,
    Generic, TypeVar,
    get_args, cast, )

_T = TypeVar('_T')
_T1 = TypeVar('_T1')
_T2 = TypeVar('_T2')
_K = TypeVar('_K')
_V = TypeVar('_V')

StrList = list[str]
SkeyDict = dict[str, _T]


class ConstValueClass:
    def __init__(self):
        raise "This class is only Const Value"


def type_map1(__func: Callable[[_T], _V], __iter1: Iterable[_T]) -> tuple[_T, ...]:
    values = map(__func, __iter1)
    return tuple(values)


def main():
    a: SkeyDict[int]

    pass


if __name__ == '__main__':
    main()
