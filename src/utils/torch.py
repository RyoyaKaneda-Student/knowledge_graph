import os
import warnings
import random
import numpy as np

import torch
import torch.nn as nn
from logging import Logger
# noinspection PyUnresolvedReferences
from typing import List, Dict, Tuple, Optional, Callable, Union
import math

from utils.progress_manager import ProgressHelper
from utils.utils import none_count, is_same_len_in_list

INF = float('inf')

ndarray_Tensor = Union[np.ndarray, torch.Tensor]


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


def torch_fix_seed(seed=42):
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True


def cuda_empty_cache():
    torch.cuda.empty_cache()


# decorator
def force_cuda_empty_cache_after_function(func):
    def wrapper(*args, **kwargs):
        rev = func(*args, **kwargs)
        cuda_empty_cache()
        return rev

    return wrapper


def load_model(model: nn.Module, model_path: str, device, *, delete_file=False):
    with force_cpu(model, device):
        model.load_state_dict(torch.load(model_path))
        if delete_file:
            os.remove(model_path)
    return model


def save_model(model: nn.Module, model_path: str, device):
    with force_cpu(model, device):
        torch.save(model.state_dict(), model_path)


def decorate_loader(_loader, no_show_bar):
    if no_show_bar:
        return enumerate(_loader)
    else:
        from tqdm import tqdm
        return tqdm(enumerate(_loader), total=len(_loader), leave=False)


def onehot(items: Union[List[int], torch.Tensor], num_classes) -> torch.Tensor:
    if not torch.is_tensor(items):
        items = torch.tensor(items)
    return torch.nn.functional.one_hot(items, num_classes=num_classes)


def insert_to_front_of_all_batch(tensor_, full_value):
    return torch.cat(torch.full((len(tensor_), 1), tensor_))


def param_count_check(model):
    params = 0
    for p in model.parameters():
        if p.requires_grad:
            params += p.numel()
    return params


def random_choice(x: torch.Tensor, n: Union[torch.Tensor, int], filter_: Optional[torch.Tensor] = None):
    indices = random_indices_choice(x, n, filter_)
    return torch.index_select(x, 0, indices)


_x_indexes_raw: torch.Tensor = torch.tensor([i for i in range(1000000)], dtype=torch.long)


def random_indices_choice(
        x: torch.Tensor,
        n: Union[torch.Tensor, int],
        p: Union[torch.Tensor, np.ndarray, list] = None,
):
    assert x.dim() == 1
    assert none_count(filter_, not_choice_num_list) > 0, "only one choice. (filter_, not_choice_num_list)"
    assert filter_ is None or x.shape == filter_.shape

    x_indexes: torch.Tensor = _x_indexes_raw[:len(x)]
    x_indexes = x_indexes[filter_]
    indices = x_indexes[torch.randperm(len(x_indexes))][:n]
    if len(indices) != n:
        print(x_indexes)
        warnings.warn("すくな！")
        raise "saa"
    return indices


def onehot_target(target_num, target_len, len_, dtype: torch.dtype = None):
    return torch.tensor([1 if i == target_num else 0 for i in range(target_len)], dtype=dtype).repeat(len_)


class force_cpu(object):
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


class SparceData:
    """
    Args:
        self._sparce_data: [ [array(...), array(...)], [array(...), array(...)]  ]

    """

    def __init__(self,
                 _sparce_data: List[Tuple[np.ndarray, ndarray_Tensor]],
                 _tmp: Union[np.ndarray, torch.Tensor],
                 *, is_torch=False
                 ):
        """
        Args:
            data: [ [(0,1), (1,1) ], [(0,2), (2,2)] ]
        """
        self._sparce_data: List[Tuple[np.ndarray, ndarray_Tensor]] = _sparce_data
        self._tmp: Union[np.ndarray, torch.Tensor] = _tmp
        self._is_torch = is_torch
        if is_torch:
            for i in range(100):
                output, counts = torch.unique(self._sparce_data[i][1], return_counts=True)
                # print(f"self._sparce_data[0], {output=}, {counts=}")

    def add_to_data_index(self, item):
        self._sparce_data = [
            (_data_index + item, _data) for (_data_index, _data) in self._sparce_data]
        len_tmp = len(self._tmp) + item
        self._tmp = np.zeros(len_tmp, dtype=np.int8) if not self._is_torch else torch.zeros(len_tmp, dtype=torch.int8)

    def get_data(self, index: int, filter_func: Callable = None, fill_value=None) -> Union[np.ndarray, ndarray_Tensor]:
        _data_index, _data = self._sparce_data[index]
        if filter_func is not None:
            filter_ = filter_func(_data_index, _data)
            _data_index, _data = _data_index[filter_], _data[filter_]
        rep = self._tmp.clone()
        rep[_data_index] = fill_value if fill_value is not None else _data
        return rep

    def get_raw_data(self, index: int):
        return self._sparce_data[index]

    def del_indices(self, *,
                    del_indices: Union[List[int], np.ndarray, None] = None,
                    save_indices: Union[List[int], np.ndarray, None] = None,
                    key: Callable[[np.ndarray, np.ndarray], bool] = None
                    ) -> List:
        """
        Args:
            del_indices: del if
            save_indices:
            key: delete if key(i) is True.
        Returns: The list of indices (True if the index is deleted else False.).
        """
        sparce_data = self._sparce_data
        tmp = none_count([del_indices, save_indices, key])
        tmp_list = np.zeros(len(sparce_data), dtype=bool)

        if tmp >= 2 or tmp == 0:
            raise "select only one value."
        elif del_indices is not None:
            tmp_list[del_indices] = True
        elif save_indices is not None:
            tmp_list[:] = True
            tmp_list[save_indices] = False
            pass
        elif key is not None:
            list_ = []
            for i, (_data_index, _data) in enumerate(sparce_data):
                if key(_data_index, _data): list_.append(i)
            tmp_list[list_] = True

        new_sparce_data = [
            item for i, item in enumerate(self._sparce_data) if not tmp_list[i]
        ]

        assert len(new_sparce_data) == len(self._sparce_data) - np.count_nonzero(tmp_list)

        self._sparce_data = new_sparce_data
        return tmp_list

    def __len__(self):
        return len(self._sparce_data)

    @classmethod
    def from_row_index_value(cls, rows: np.ndarray, indices: np.ndarray, values: np.ndarray,
                             max_value: Optional[int] = None):

        assert is_same_len_in_list(rows, indices, values)
        assert rows.ndim == 1 and indices.ndim == 1 and values.ndim == 1
        max_value = max_value if max_value is not None else max(values).item()

        list_ = [([], []) for _ in range(max(rows))]
        for r, i, v in zip(rows.tolist(), indices.tolist(), values.tolist()):
            list_[r][0].append(i)
            list_[r][1].append(v)

        tmp_ = np.zeros(max_value)

        return cls([(np.array(l_[0]), np.array(l_[1])) for l_ in list_], tmp_)

    @classmethod
    def from_rawdata(cls, data: List[List[Tuple[int, int]]], max_len=None):
        _sparce_data = [list(zip(*tails)) for tails in data]
        _sparce_data = [(np.array(_data_index, dtype=np.int64), np.array(_data, dtype=np.int8)) for _data_index, _data
                        in _sparce_data]
        max_len = max([max(_data_index) for _data_index, _data in _sparce_data]) if max_len is None else max_len
        return cls(_sparce_data, np.zeros(max_len))

    @classmethod
    def make_clone(cls, old, to_tensor: bool):
        assert type(old) == SparceData
        _sparce_data, _tmp = old._sparce_data.copy(), old._tmp.copy()
        if to_tensor:
            _sparce_data = [
                (_data_index, torch.from_numpy(_data).to(torch.int8)) for _data_index, _data in _sparce_data
            ]
            _tmp = torch.zeros(len(_tmp))
        return SparceData(_sparce_data, _tmp, is_torch=True)


class PositionalEncoding(nn.Module):
    """ PositionalEncoding
    batch first
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :x.size(1)]
        x = self.dropout(x)
        return x


class MM(nn.Module):
    """ PositionalEncoding
    batch first
    """

    def __init__(self):
        super(MM, self).__init__()

    def forward(self, x, y):
        return torch.mm(x, y)


class LossHelper:
    def __init__(self, update_min_d=1e-6, progress_helper: ProgressHelper = None, early_total_count=20):
        self._loss_list = [INF]
        self.min_loss = INF
        self.update_min_d = update_min_d
        self.not_update_count = 0
        self.progress_helper = progress_helper
        progress_helper.add_key('early_count', total=early_total_count)

    def update(self, loss):
        self._loss_list.append(loss)
        rev = (self.min_loss - loss) > self.update_min_d
        if rev:
            self.not_update_count = 0
            self.progress_helper.reset_key('early_count')
        else:
            self.not_update_count += 1
            self.progress_helper.update_key('early_count')
        self.min_loss = min(self.min_loss, loss)

    @property
    def all_loss(self):
        return self._loss_list[1:]
