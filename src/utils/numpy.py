import numpy as np

from typing import Union


def negative_sampling(items: np.ndarray, count_per_items: np.ndarray, size: Union[int, np.ndarray], replace=True):
    assert items.ndim == 1 and count_per_items.ndim == 1
    assert items.shape == count_per_items.shape
    return np.random.choice(
        items, size=size, replace=replace,
        p=np.power(count_per_items, 0.75) / np.sum(np.power(count_per_items, 0.75))
    )
