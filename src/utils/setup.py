# coding: UTF-8
import dataclasses
import os
from typing import Tuple, Any
import sys
from pathlib import Path
import pickle
from logging import Logger
from argparse import Namespace

import torch
from .torch import get_device


class ChangeDisableNamespace(Namespace):
    def __setattr__(self, name, value):
        if name == '__dict__':
            # 初回設定用
            super(ChangeDisableNamespace, self).__setattr__('__dict__', value)
        else:
            raise TypeError(f'Can\'t set value')

    def __init__(self, args, **kwargs: Any):
        super().__init__(**kwargs)
        self.__dict__ = args.__dict__

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass


def command_help(args):
    print(args.parser.parse_args([args.command, '--help']))


def setup_logger(name, logfile, console_level=None) -> Logger:
    import logging
    name = name
    console_level = logging.DEBUG if console_level == 'debug' else logging.INFO

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # create file handler which logs even DEBUG messages
    fh = logging.FileHandler(logfile)
    fh.setLevel(logging.DEBUG)
    fh_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(filename)s - %(name)s - %(funcName)s - %(message)s')
    fh.setFormatter(fh_formatter)

    # create console handler with a INFO log level
    ch = logging.StreamHandler()
    ch.setLevel(console_level)
    ch_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(filename)s - %(lineno)d - %(message)s', '%Y-%m-%d %H:%M:%S')
    ch.setFormatter(ch_formatter)

    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.propagate = False
    return logger


def setup(setup_parser, project_dir, *, parser_args=None) -> Tuple[Namespace, Logger, torch.device]:
    from dotenv import load_dotenv
    load_dotenv()
    args: Namespace = setup_parser(parser_args)
    logger: Logger = setup_logger(__name__, f"{project_dir}/{args.logfile}", console_level=args.console_level)
    device: torch.device = get_device(device_name=args.device_name, logger=logger)
    # process id
    args.pid = os.getpid()
    return args, logger, device


def save_param(args: Namespace):
    args = vars(args)
    if 'param_file' in args and args['param_file'] is not None:
        param_file = args['param_file']
        project_dir = args['project_dir']
        param_file = f"{project_dir}/{param_file}"

        os.makedirs(os.path.dirname(param_file), exist_ok=True)
        with open(param_file, "wb") as tf:
            pickle.dump(args, tf)
        args['logger'].info("save params")


def main(setup_parser, project_dir):
    try:
        args, logger, device = setup(setup_parser, project_dir)
        args.project_dir = project_dir
        args.logger = logger
        args.device = device
        logger.info(vars(args))

        if hasattr(args, 'handler'):
            args.handler(args)
            pass
        else:
            args.parser.print_help(args)

    except BaseException as e:
        logger.error(e)

    finally:
        save_param(args)


