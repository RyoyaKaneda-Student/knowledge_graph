import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(os.path.join(PROJECT_DIR, 'src'))

from unittest import TestCase

import utils.utils as utils


class Test(TestCase):
    def test_is_same_item_in_list(self):
        list_ = ['a', 'a', 'a']
        self.assertEqual(utils.is_same_item_in_list(*list_), True)
        list_ = ['a', 'b', 'a']
        self.assertEqual(utils.is_same_item_in_list(*list_), False)
        list_ = ['a', 'a', 'b']
        self.assertEqual(utils.is_same_item_in_list(*list_), False)

    def test_is_same_len_in_list(self):
        list_ = [[1], [2], [3]]
        self.assertEqual(utils.is_same_len_in_list(*list_), True)
        list_ = [[1, ], (2, ), [3, ]]
        self.assertEqual(utils.is_same_len_in_list(*list_), True)
        list_ = [[1, ], (2, 3), [4, ]]
        self.assertEqual(utils.is_same_len_in_list(*list_), False)
