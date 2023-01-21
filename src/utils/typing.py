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


class ConstMeta(type):
    """Const parameter metaclass.

    * This is only Const parameter's metaclass.
    """
    def __setattr__(self, name, value):
        if name in self.__dict__:
            raise TypeError(f'Can\'t rebind const ({name})')
        else:
            self.__setattr__(name, value)


def type_map1(__func: Callable[[_T], _V], __iter1: Iterable[_T]) -> tuple[_T, ...]:
    values = map(__func, __iter1)
    return tuple(values)


def notNone(x: Optional[_T], error_msg='it must not None, but is is None.') -> _T:
    if x is None: raise TypeError(error_msg)
    return x


def main():
    a: SkeyDict[int]

    pass


if __name__ == '__main__':
    main()
