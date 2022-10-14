# region !import area!
# ========== python OS level ==========
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(os.path.join(PROJECT_DIR, 'src'))
# ========== python ==========
from tqdm import tqdm
from logging import Logger
# noinspection PyUnresolvedReferences
from typing import List, Dict, Tuple, Optional, Union, Callable


# endregion

def remake_filename(file_name, pid=None):
    if pid is not None and r'.{pid}.' in file_name:
        file_name = file_name.replace(r'.{pid}.', f'.{pid}.')
    return file_name


class ProgressHelper:
    def __init__(self, file_name, pid=None):
        file_name = remake_filename(file_name, pid)
        self.file_name = file_name
        self.f = open(file_name, mode='w')
        self.f.write(f"{pid if pid is not None else ''}\n")
        self.dict_: Dict[str, tqdm] = {}

        self._finish = False
        self.disable = False

    def add_key(self, key, total, description=None, *args, **kwargs):
        pbar = tqdm(total=total, desc=key, leave=False, file=self.f, disable=self.disable, **kwargs)
        if description is None: description = key
        pbar.set_description(description)
        self.dict_[key] = pbar
        pbar.update(0)

    def update_key(self, key, add_value=1):
        self.dict_[key].update(add_value)

    def reset_key(self, key):
        self.dict_[key].reset()

    def finish_key(self, key):
        self.dict_[key].close()

    def progress(self, iterable, key, total=None, description=None, *args, **kwargs):
        if total is None:
            try:
                total=len(iterable)
            except AttributeError as e:
                total = None
        assert total is not None
        self.add_key(key, total, description, *args, **kwargs)

        for obj in iterable:
            yield obj
            self.update_key(key)

        self.finish_key(key)

    # decorator
    def update_progress_after_function(self, key, add_value=1):
        def updater(func):
            def wrapper(*args, **kwargs):
                res = func(*args, **kwargs)
                self.update_key(key, add_value=add_value)
                return res
            return wrapper
        return updater

    def finish(self, delete=True):
        self.f.close()
        if delete: os.remove(self.file_name)
        self._finish = True

    def __del__(self):
        if not self._finish: self.finish(delete=False)


def main():
    pw = ProgressHelper(file_name="./test_progress.txt")
    total_ = 1000
    for i in pw.progress(range(total_), 'i'):
        for j in pw.progress(range(total_), 'j'):
            for k in pw.progress(range(total_), 'k'):
                pass
            pass
        pass
    pw.finish(delete=True)


if __name__ == '__main__':
    main()

