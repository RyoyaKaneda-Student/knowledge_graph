# coding: UTF-8
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(os.path.join(PROJECT_DIR, 'src'))

# python
from logging import Logger
from typing import List, Dict, Tuple, Optional, Callable, Union
import dataclasses
from tqdm import tqdm
# Machine learning
import h5py
import numpy as np
import pandas as pd
# torch
import torch
from torch.utils.data import Dataset
from torch import nn
from torch.utils.data.dataloader import DataLoader
# Made by me
from utils.setup import setup, save_param
from model import ConvE, DistMult, Complex, TransformerE

PROCESSED_DATA_PATH = './data/processed/'
EXTERNAL_DATA_PATH = './data/external/'

KGDATA_ALL = ['FB15k-237', 'WN18RR', 'YAGO3-10']
name2model = {
    'conve': ConvE,
    'distmult': DistMult,
    'complex': Complex,
    'transformere': TransformerE
}


def debug(*, logger):
    logger.debug("====================version check====================")
    logger.debug(np.__version__)
    logger.debug(pd.__version__)
    logger.debug(torch.__version__)
    logger.debug("====================version check====================")


def setup_parser():
    import argparse  # 1. argparseをインポート
    parser = argparse.ArgumentParser(description='データの初期化')
    parser.add_argument('--logfile', help='ログファイルのパス', type=str)
    parser.add_argument('--param-file', help='パラメータを保存するファイルのパス', type=str)
    parser.add_argument('--console-level', help='コンソール出力のレベル', type=str, default='info')
    parser.add_argument('--no-show-bar', help='バーを表示しない', action='store_true')
    parser.add_argument('--device-name', help='cpu or cuda or mps', type=str, default='cpu',
                        choices=['cpu', 'cuda', 'mps'])

    parser.add_argument('--KGdata', help=' or '.join(KGDATA_ALL), type=str,
                        choices=KGDATA_ALL)
    parser.add_argument('--model', type=str,
                        help='Choose from: {conve, distmult, complex, transformere}')
    parser.add_argument('--embedding-dim', type=int, default=200,
                        help='The embedding dimension (1D). Default: 200')
    parser.add_argument('--batch-size', help='batch size', type=int)
    parser.add_argument('--epoch', help='max epoch', type=int)

    parser.add_argument('--model-path', type=str, help='model path')
    parser.add_argument('--do-train', help='do-train', action='store_true')
    parser.add_argument('--do-valid', help='do-valid', action='store_true')
    parser.add_argument('--do-test', help='do-test', action='store_true')

    parser.add_argument('--embedding-shape1', type=int, default=20,
                        help='The first dimension of the reshaped 2D embedding. '
                             'The second dimension is inferred. Default: 20')
    # optimizer
    parser.add_argument('--lr', type=float, default=0.003, help='learning rate (default: 0.003)')
    parser.add_argument('--l2', type=float, default=0.0,
                        help='Weight decay value to use in the optimizer. Default: 0.0')
    parser.add_argument('--label-smoothing', type=float, default=0.1, help='Label smoothing value to use. Default: 0.1')
    # convE
    parser.add_argument('--hidden-drop', type=float, default=0.3, help='Dropout for the hidden layer. Default: 0.3.')
    parser.add_argument('--input-drop', type=float, default=0.2, help='Dropout for the input embeddings. Default: 0.2.')
    parser.add_argument('--feat-drop', type=float, default=0.2,
                        help='Dropout for the convolutional features. Default: 0.2.')
    parser.add_argument('--use-bias', action='store_true', help='Use a bias in the convolutional layer. Default: True')
    parser.add_argument('--lr-decay', type=float, default=0.995,
                        help='Decay the learning rate by this factor every epoch. Default: 0.995')
    parser.add_argument('--hidden-size', type=int, default=9728,
                        help='The side of the hidden layer. The required size changes with the size of the embeddings. '
                             'Default: 9728 (embedding size 200).')
    # transformere
    parser.add_argument('--nhead', type=int, default=8, help='nhead. Default: 8.')
    parser.add_argument('--transformer-drop', type=float, default=0.1, help='transformer-drop. Default: 0.1.')


    # コマンドライン引数をパースして対応するハンドラ関数を実行
    _args = parser.parse_args()

    return _args


def _select_model(args, model_name, e_length, r_length) -> nn.Module:
    if model_name in name2model.keys():
        model = name2model[model_name](args, e_length, r_length)
        pass
    else:
        raise Exception(f"Unknown model! :{model_name}")
        pass
    return model


def _load_model(model: nn.Module, model_path: str, device, *, delete_file=False):
    model.to('cpu')
    model.load_state_dict(torch.load(model_path))
    if delete_file:
        os.remove(model_path)
    model.to(device)


def _save_model(model: nn.Module, model_path: str, device):
    model.to('cpu')
    torch.save(model.state_dict(), model_path)
    model.to(device)


def _get_loader(_loader, no_show_bar):
    if no_show_bar:
        return enumerate(_loader)
    else:
        return tqdm(enumerate(_loader), total=len(_loader), leave=False)


def _get_from_dict(dict_: dict, names: Tuple[str, ...]):
    return map(lambda x: dict_[x], names)


# cuda cache remove
def _ccr():
    torch.cuda.empty_cache()


@dataclasses.dataclass(init=False)
class MyTrainTestData:
    train_path: str
    valid_path: str
    test_path: str
    is_use_zero: bool
    e_length: int
    r_length: int
    er_list: np.ndarray
    er2index: Dict[Tuple[int, int], int]
    er_tails_data: np.ndarray
    er_tails_row: np.ndarray
    er_tails_data_type: np.ndarray
    _train_triple: Optional[np.ndarray]
    _valid_triple: Optional[np.ndarray]
    _test_triple: Optional[np.ndarray]

    def __init__(self, info_path, train_path, valid_path, test_path, del_zero2zero):
        self.train_path, self.valid_path, self.test_path = (
            train_path, valid_path, test_path
        )
        self.is_use_zero = not del_zero2zero
        with h5py.File(info_path, 'r') as f:
            self.e_length = f['e_length'][()] - (1 if del_zero2zero else 0)
            self.r_length = f['r_length'][()] - (1 if del_zero2zero else 0)

        self._get_er_tails()
        self._train_triple = None
        # self._get_train()
        # self._get_valid()
        # self._get_test()

    def _get_er_tails(self):
        tmp = 0 if self.is_use_zero else 1
        with h5py.File(self.train_path) as f:
            self.er_list = f['er_list'][tmp:] - tmp
            self.er_tails_data = f['er_tails_data'][tmp:] - tmp
            self.er_tails_row = f['er_tails_row'][tmp:] - tmp
            self.er_tails_data_type = f['er_tails_data_type'][tmp:]

        self.er2index = {tuple(er.tolist()): i for i, er in enumerate(self.er_list)}

    def _get_train(self):
        tmp = 0 if self.is_use_zero else 1
        with h5py.File(self.train_path) as f:
            self._train_triple = f['triple'][tmp:] - tmp
            pass

    def _get_valid(self):
        tmp = 0 if self.is_use_zero else 1
        with h5py.File(self.valid_path) as f:
            self._valid_triple = f['triple'][tmp:] - tmp
            pass

    def _get_test(self):
        tmp = 0 if self.is_use_zero else 1
        with h5py.File(self.test_path) as f:
            self._test_triple = f['triple'][tmp:] - tmp
            pass

    @property
    def train_triple(self):
        self._get_train()
        return self._train_triple

    @property
    def valid_triple(self):
        self._get_valid()
        return self._valid_triple

    @property
    def test_triple(self):
        self._get_test()
        return self._test_triple

    def __del__(self):
        pass


@dataclasses.dataclass(init=False)
class MyDataset(Dataset):
    data: torch.Tensor
    label: torch.Tensor

    def __init__(self, data: np.ndarray, label: np.ndarray, target_num: int, del_if_no_tail=False):
        if del_if_no_tail:
            tmp = np.count_nonzero(label == target_num, axis=1) > 0
            data = data[tmp]
            label = label[tmp]

        self.data = torch.from_numpy(data)
        self.label = torch.from_numpy(label == target_num).to(torch.float32)
        self.del_if_no_tail = del_if_no_tail

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        er = self.data[index]
        e2s = self.label[index]
        return er, e2s

    def __len__(self) -> int:
        return len(self.data)


@dataclasses.dataclass(init=False)
class MyDatasetEcoMemory(Dataset):
    data: torch.Tensor
    label_all: torch.Tensor
    target_num: int

    def __init__(self, data: np.ndarray, label: np.ndarray, target_num: int, del_if_no_tail=False):
        if del_if_no_tail:
            tmp = np.count_nonzero(label == target_num, axis=1) > 0
            data = data[tmp]
            label = label[tmp]
        self.data = torch.from_numpy(data)
        self.label_all = torch.from_numpy(label)
        self.target_num = target_num

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        er = self.data[index]
        e2s = self.label_all[index]
        e2s = (e2s == self.target_num).to(torch.float32)
        return er, e2s

    def __len__(self) -> int:
        return len(self.data)


@dataclasses.dataclass(init=False)
class MyDatasetWithFilter(MyDatasetEcoMemory):
    def __getitem__(self, index: int
                    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        er = self.data[index]
        e2s_all = self.label_all[index]
        e2s_target = (e2s_all == self.target_num).to(torch.float32)
        return er, e2s_target, e2s_all


@dataclasses.dataclass(init=False)
class MyTripleDataset(Dataset):
    er: torch.Tensor
    tail: torch.Tensor
    er_list_index: torch.Tensor

    def __init__(self, triples: np.ndarray, er2index: Dict[Tuple[int, int], int]):
        self.er, self.tail = torch.from_numpy(triples).split(2, 1)
        self.er_list_index = torch.tensor(
            [er2index[(e.item(), r.item())] for e, r in self.er], requires_grad=False, dtype=torch.int64)

    def __getitem__(
            self,
            index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        er = self.er[index]
        tail = self.tail[index]
        er_list_index = self.er_list_index[index]
        return er, tail, er_list_index

    def __len__(self) -> int:
        return len(self.er)


class MyDataHelper:
    def __init__(self, info_path, train_path, valid_path, test_path, del_zero2zero=True, *, logger=None):
        super().__init__()
        my_data = MyTrainTestData(info_path, train_path, valid_path, test_path, del_zero2zero)

        e_length = my_data.e_length
        r_length = my_data.r_length
        er_list: np.ndarray = my_data.er_list
        er_tails_data: np.ndarray = my_data.er_tails_data
        er_tails_row: np.ndarray = my_data.er_tails_row
        er_tails_data_type: np.ndarray = my_data.er_tails_data_type
        # debug
        if logger is not None:
            logger.debug(f"e_length = {e_length}, r_length = {r_length}")
        #
        er_tails: List[List[Tuple[int, int]]] = [[] for _ in er_list]
        for row, data, data_type in zip(er_tails_row, er_tails_data, er_tails_data_type):
            row, data, data_type = row.item(), data.item(), data_type.item()
            er_tails[row].append((data, data_type))

        labels = np.zeros((len(er_list), e_length), dtype=np.int8)

        for index, tails in enumerate(er_tails):
            tails_data, tails_data_type = zip(*tails)
            np.put(labels[index], tails_data, tails_data_type)
            del tails

        self._data = my_data
        self._er_list = er_list
        self._label = labels
        # dataset
        """
        self._train_dataset: Dataset = None
        self._valid_dataset: Dataset = None
        self._test_dataset: DataLoader = None
        """
        # dataloader
        self._train_dataloader: Optional[DataLoader] = None
        self._valid_dataloader: Optional[DataLoader] = None
        self._test_dataloader: Optional[DataLoader] = None
        # dataloader getter (メモリ節約)
        self._valid_dataloader_getter: Optional[Callable[[], DataLoader]] = None
        self._test_dataloader_getter: Optional[Callable[[], DataLoader]] = None

    def _get_dataset(self, eco_memory: bool, target_num) -> Union[MyDataset, MyDatasetEcoMemory]:
        if eco_memory:
            MyDatasetEcoMemory(self._er_list, self.label, target_num=target_num)
        else:
            return MyDataset(self._er_list, self.label, target_num=target_num)

    def get_train_dataset(self, eco_memory: bool) -> Union[MyDataset, MyDatasetEcoMemory]:
        if eco_memory:
            return MyDatasetEcoMemory(self._er_list, self.label, target_num=1)
        else:
            return MyDataset(self._er_list, self.label, target_num=1)

    def get_valid_dataset(self) -> Union[MyDatasetWithFilter]:
        return MyDatasetWithFilter(self._er_list, self.label, target_num=2, del_if_no_tail=True)

    def get_test_dataset(self) -> Union[MyDatasetWithFilter]:
        return MyDatasetWithFilter(self._er_list, self.label, target_num=3, del_if_no_tail=True)

    def get_train_triple_dataset(self) -> MyTripleDataset:
        return MyTripleDataset(self.data.train_triple, self.data.er2index)

    def get_valid_triple_dataset(self) -> MyTripleDataset:
        return MyTripleDataset(self.data.valid_triple, self.data.er2index)

    def get_test_triple_dataset(self) -> MyTripleDataset:
        return MyTripleDataset(self.data.test_triple, self.data.er2index)

    def del_loaders(self):
        del self._train_dataloader, self._valid_dataloader, self._test_dataloader

    def set_loaders(self, train_dataloader: DataLoader, valid_dataloader: DataLoader, test_dataloader: DataLoader):
        self._train_dataloader = train_dataloader
        self._valid_dataloader = valid_dataloader
        self._test_dataloader = test_dataloader

    def set_loader_getters(self, valid_dataloader_getter, test_dataloader_getter):
        self._valid_dataloader_getter = valid_dataloader_getter
        self._test_dataloader_getter = test_dataloader_getter

    @property
    def data(self) -> MyTrainTestData:
        return self._data

    @property
    def label(self) -> np.ndarray:
        return self._label

    @property
    def train_dataloader(self) -> DataLoader:
        return self._train_dataloader

    @property
    def valid_dataloader(self) -> DataLoader:
        _valid_dataloader = self._valid_dataloader
        _valid_dataloader_getter = self._valid_dataloader_getter
        if (_valid_dataloader is not None) and (_valid_dataloader_getter is None):
            return _valid_dataloader
        elif (_valid_dataloader is None) and (_valid_dataloader_getter is not None):
            return _valid_dataloader_getter()
        else:
            raise "_valid_dataloader or _valid_dataloader_getter must not be none"

    @property
    def test_dataloader(self) -> DataLoader:
        _test_dataloader = self._test_dataloader
        _test_dataloader_getter = self._test_dataloader_getter
        if (_test_dataloader is not None) and (_test_dataloader_getter is None):
            return _test_dataloader
        elif (_test_dataloader is None) and (_test_dataloader_getter is not None):
            return _test_dataloader_getter()
        else:
            raise "_test_dataloader or _test_dataloader_getter must not be none"


def load_data(kg_data):
    info_path = os.path.join(PROCESSED_DATA_PATH, 'KGdata', kg_data, f"info.hdf5")
    train_path = os.path.join(PROCESSED_DATA_PATH, 'KGdata', kg_data, f"train.hdf5")
    valid_path = os.path.join(PROCESSED_DATA_PATH, 'KGdata', kg_data, f"valid.hdf5")
    test_path = os.path.join(PROCESSED_DATA_PATH, 'KGdata', kg_data, f"test.hdf5")
    data_helper = MyDataHelper(info_path, train_path, valid_path, test_path)
    return data_helper


def make_dataloader(data_helper: MyDataHelper, batch_size):
    train = DataLoader(data_helper.get_train_dataset(eco_memory=False), batch_size=batch_size, shuffle=True)
    # this is debug
    # valid_dataset = MyDatasetWithFilter(data_helper._er_list, data_helper.label, target_num=1, del_if_no_tail=True)
    # valid = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    valid = DataLoader(data_helper.get_valid_dataset(), batch_size=batch_size, shuffle=False)
    test = DataLoader(data_helper.get_test_dataset(), batch_size=batch_size, shuffle=False)

    data_helper.set_loaders(train, valid, test)  # debug


def get_model(args, data_helper):
    model = _select_model(args, args.model, data_helper.data.e_length, data_helper.data.r_length)
    model.init()
    return model


def training(
        args, *, logger,
        model, data_helper,
        do_valid,
        no_show_bar=False,
):
    device = args.device
    max_epoch = args.epoch
    checkpoint_path = f"saved_models/.tmp/check-point.pid={args.pid}.model"
    # data
    train = data_helper.train_dataloader
    # len_dataset = len(train.dataset)
    #
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    model.to(device)
    result = {
        'train_loss': [],
        'mrr': [],
        'hit_': [],
        'completed_epoch': -1
    }

    def append_to_result(_name, _value, *, _epoch=None):
        if not _epoch:
            result[_name].append(_value)
        else:
            result[_name][_epoch] = _value

    for epoch in range(max_epoch):
        # train
        logger.info(f"{'-' * 10}epoch {epoch + 1} start. {'-' * 10}")
        model.train()
        loss = torch.tensor(0., requires_grad=False, device=device)
        sum_train = 0

        for idx, (er, e2s) in _get_loader(train, no_show_bar=no_show_bar):
            opt.zero_grad()
            er, e2s = er.to(device), e2s.to(device)
            e, r = er.split(1, 1)
            # e2s = ((1.0 - args.label_smoothing) * e2s) + (1.0 / e2s.size(1))
            sum_train += (e2s[e2s != 0]).sum()

            pred: torch.Tensor = model.forward(e, r)
            _loss = model.loss(pred, e2s)
            _loss.backward()
            opt.step()
            # loss check
            _loss = _loss.detach().sum()
            loss += _loss
            del e, r, er, pred, e2s, _loss
            _ccr()

        logger.debug(f"sum_train: {sum_train}")
        loss /= len(train)
        append_to_result('train_loss', loss.to('cpu').item())
        logger.info("-----train result (epoch={}): loss = {}".format(epoch + 1, loss))
        logger.debug(f"{'-' * 5}epoch {epoch + 1} train end. {'-' * 5}")
        del loss
        _ccr()
        # valid
        if do_valid:
            logger.debug(f"{'-' * 10}epoch {epoch + 1} valid start. {'-' * 5}")
            _result = testing(args, logger=logger, model=model, data_helper=data_helper, is_valid=True,
                              no_show_bar=no_show_bar)
            mrr, hit_ = _get_from_dict(_result, ('mrr', 'hit_'))
            append_to_result('mrr', mrr)
            append_to_result('hit_', hit_)
            logger.info("-----valid result (epoch={}): mrr = {}".format(epoch + 1, mrr))
            logger.info("-----valid result (epoch={}): hit = {}".format(epoch + 1, hit_))
            logger.debug(f"{'-' * 5}epoch {epoch + 1} valid end. {'-' * 5}")

        logger.info(f"{'-' * 10}epoch {epoch + 1} end.{'-' * 10}")
        _save_model(model, checkpoint_path, device=device)
        result['completed_epoch'] = epoch + 1

    _load_model(model, checkpoint_path, device=device, delete_file=True)
    return model


@torch.no_grad()
def testing(
        args, *, logger,
        model, data_helper,
        is_valid=False, is_test=False,
        no_show_bar=False,
):
    device = args.device
    if is_valid:
        logger.debug(f"{'-' * 5}This is valid{'-' * 5}")
        test = data_helper.valid_dataloader
    elif is_test:
        logger.debug(f"{'-' * 5}This is test{'-' * 5}")
        test = data_helper.test_dataloader
    else:
        raise "Either valid or test must be specified."
        pass

    len_test = 0
    zero_tensor = torch.tensor(0., dtype=torch.float32, device=device)

    # test
    model.to(device)
    model.eval()

    mrr = torch.tensor(0., device=device, dtype=torch.float32, requires_grad=False)
    hit_ = torch.tensor([0.] * 10, device=device, dtype=torch.float32, requires_grad=False)

    with torch.no_grad():
        for idx, (er, e2s, e2s_all) in _get_loader(test, no_show_bar=no_show_bar):
            er = er.to(device)
            e, r = er.split(1, 1)
            pred: torch.Tensor = model.forward(e, r)
            del e, r, er
            # make filter
            e2s = e2s.to(device)
            row, column = torch.where(e2s == 1)
            del e2s
            e2s_all_binary: torch.Tensor = e2s_all != 0
            del e2s_all
            #
            pred = pred[row]  # 複製
            e2s_all_binary = e2s_all_binary[row]  # 複製
            # row is change
            len_row = len(row)
            row = [i for i in range(len_row)]
            #
            e2s_all_binary[row, column] = False
            pred[e2s_all_binary] = zero_tensor
            del e2s_all_binary
            #
            ranking = torch.argsort(pred, dim=1, descending=True)  # これは0 始まり
            del pred
            _ccr()
            ranks = torch.argsort(ranking, dim=1)[row, column]
            del ranking
            _ccr()
            ranks += 1
            # mrr and hit
            mrr += (1. / ranks).sum()
            for i in range(10):
                hit_[i] += torch.count_nonzero(ranks <= (i + 1))
            del ranks, row, column
            # after
            _ccr()
            len_test += len_row

    mrr = (mrr / len_test)
    hit_ = (hit_ / len_test)
    rev = {
        'mrr': mrr.item(), 'hit_': hit_.tolist()
    }
    del mrr, hit_
    _ccr()
    logger.debug("=====Test result: mrr = {}".format(rev['mrr']))
    logger.debug("=====Test result: hit = {}".format(rev['hit_']))
    return rev


def _info_str(s):
    return s.ljust(25).center(40, '=')


def conv1train(args, *, logger: Logger):
    kg_data = args.KGdata
    do_train, do_valid, do_test = args.do_train, args.do_valid, args.do_test
    model_path = args.model_path
    batch_size = args.batch_size
    device = args.device
    no_show_bar = args.no_show_bar

    logger.info(f"Function start".center(40, '='))

    # load data
    logger.info(_info_str(f"load data start."))
    data_helper = load_data(kg_data)
    logger.info(_info_str(f"load data complete."))

    # dataloader
    logger.info(_info_str(f"make dataloader start."))
    make_dataloader(data_helper, batch_size)
    logger.info(_info_str(f"make dataloader complete."))

    # model
    logger.info(_info_str(f"make model start."))
    model = get_model(args, data_helper)
    logger.info(_info_str(f"make model complete."))

    if do_train:
        logger.info(_info_str(f"Train start."))
        training(
            args, logger=logger,
            data_helper=data_helper,
            model=model,
            do_valid=do_valid,
            no_show_bar=no_show_bar
        )
        _save_model(model, args.model_path, device=device)
        logger.info(_info_str(f"Train complete."))
    _load_model(model, model_path, device=device)
    if do_valid:
        logger.info(_info_str(f"Test valid start."))
        testing(
            args, logger=logger,
            data_helper=data_helper, model=model,
            is_valid=True,
            no_show_bar=no_show_bar
        )
        logger.info(_info_str(f"Test valid complete."))
    if do_test:
        logger.info(_info_str(f"Test start."))
        testing(
            args, logger=logger,
            data_helper=data_helper, model=model,
            is_test=True,
            no_show_bar=no_show_bar,
        )
        logger.info(_info_str(f"Test complete."))

    logger.info(f"Function finish".center(40, '='))


def main():
    args, logger, device = setup(setup_parser, PROJECT_DIR)
    try:
        args.project_dir = PROJECT_DIR
        args.logger = logger
        args.device = device
        args.completed = {}
        logger.debug(vars(args))
        logger.debug(f"process id = {args.pid}")
        conv1train(args, logger=logger)

    finally:
        save_param(args)


if __name__ == '__main__':
    main()
