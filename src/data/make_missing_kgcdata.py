from utils.setup import setup_logger
from models.datasets.data_helper import MyDataHelperForStory
import torch
import numpy as np
import h5py
from utils.hdf5 import str_list_for_hdf5

from const.const_values import JA_TITLE2LEN_INFO, SRO_ALL_INFO_FILE, SRO_ALL_TRAIN_FILE
from models.datasets.data_helper import TRAIN_INDEX
L090, L080, L075 = 'l090', 'l080', 'l075'

logger = None

def missing_make(data_helper, _title_ja, _title_en, _l100, _l, _l_name):
    triple = data_helper.data.train_triple
    entities = data_helper.processed_entities
    relations = data_helper.processed_relations
    triple_str = [(entities[h], relations[r], entities[t]) for h, r, t in triple]
    triple_head, triple_relation, triple_tail = zip(*triple_str)
    triple_str_np = str_list_for_hdf5(triple_str)

    i=_l+1
    while f'{_title_en}:{i}' not in triple_head:
        i+=1
        
    del_from_index = triple_head.index(f'{_title_en}:{i}')
    del_to_index = len(triple_head)-triple_head[::-1].index(f'{_title_en}:{_l100}')

    path_ = SRO_ALL_TRAIN_FILE.replace('train.hdf5', f'train_{_title_en}_{_l_name}.hdf5')
    logger.debug(f"delete index of {_title_ja}: from {del_from_index}, {del_to_index}")
    logger.debug(f"save path: {path_}")
    
    with h5py.File(path_, 'w') as f:
        f.create_dataset(TRAIN_INDEX.TRIPLE, data=np.delete(triple, slice(del_from_index, del_to_index), 0))
        f.create_dataset(TRAIN_INDEX.TRIPLE_RAW, data=np.delete(triple_str_np, slice(del_from_index, del_to_index), 0))

def main():
    """Main function

    """
    global logger
    logger = setup_logger(__name__, 'log/make_missing_all_data.log', console_level='debug')
    data_helper = MyDataHelperForStory(SRO_ALL_INFO_FILE, SRO_ALL_TRAIN_FILE, None, None, logger=logger,
                                       entity_special_dicts={}, relation_special_dicts={})
    
    for title_ja, (title_en, l100, l090, l080, l075) in JA_TITLE2LEN_INFO.items():
        missing_make(data_helper, title_ja, title_en, l100, l090, L090)
        missing_make(data_helper, title_ja, title_en, l100, l080, L080)
        missing_make(data_helper, title_ja, title_en, l100, l075, L075)

if __name__ == '__main__':
    main()
    pass

