# coding: UTF-8
import os
from typing import Tuple
import sys
from pathlib import Path
import pickle
from logging import Logger
from argparse import Namespace

import torch

from .torch import get_device


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


def setup(setup_parser, project_dir) -> Tuple[Namespace, Logger, torch.device]:
    from dotenv import load_dotenv
    load_dotenv()
    _args = setup_parser()
    _logger = setup_logger(__name__, f"{project_dir}/{_args.logfile}", console_level=_args.console_level)
    _device = get_device(device_name=_args.device_name, logger=_logger)
    # process id
    _args.pid = os.getpid()
    return _args, _logger, _device


def save_param(args):
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
