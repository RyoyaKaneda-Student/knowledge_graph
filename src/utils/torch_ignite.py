#!/usr/bin/python
# -*- coding: utf-8 -*-
"""torch ignite utils.

Todo:
    * 色々

"""
# ========== python ==========
# noinspection PyUnresolvedReferences
from typing import Optional, Union, Callable, Final, Iterable, Literal, get_args, cast, Sequence, TypedDict
from logging import Logger

import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.tensorboard.writer import SummaryWriter

from ignite.engine import Events, Engine
from ignite.handlers import Timer, Checkpoint, DiskSaver, global_step_from_engine

from utils.torch import force_cpu
from utils.utils import elapsed_time_str

from const.const_values import (LOSS, MODEL, OPTIMIZER)
from const.const_values import (MOST_GOOD_CHECKPOINT_PATH, LATEST_CHECKPOINT_PATH)

TRAINER: Final[str] = 'trainer'
EVALUATOR: Final[str] = 'evaluator'
GOOD_LOSS_CHECKPOINTE: Final[str] = 'good_loss_checkpointe'
LAST_CHECKPOINTE: Final[str] = 'last_checkpointe'


def save_metrics(epoch, metrics, get_tags, metric_names, summary_writer):
    """save metrics function

    Args:
        epoch(int):
        metrics(dict):
        get_tags(Callable):
        metric_names(Iterable):
        summary_writer():

    Returns:

    """
    if summary_writer is not None:
        for _name in metric_names:
            _value = metrics[_name]
            summary_writer.add_scalar(get_tags(_name), _value, global_step=epoch)


def set_start_epoch_function(trainer, *, logger: Logger, optional_func: Optional[Callable] = None):
    """set start function

    Args:
        trainer(Engine):
        logger(Logger):
        optional_func(Callable):

    """

    logger.debug("Set set_start_epoch_function")

    @trainer.on(Events.EPOCH_STARTED)
    def start_epoch_func(engine: Engine):
        """start epoch function

        * start epoch function. Move at the beginning of each epoch.

        """
        epoch = engine.state.epoch
        logger.debug("----- epoch: {:>5} start -----".format(epoch))
        optional_func()
        pass


def set_end_epoch_function(trainer, get_tag_function, *, logger: Logger, summary_writer=None, metric_names, **_):
    """set end function

    Args:
        trainer(Engine):
        logger(Logger):
        get_tag_function(Callable):
        metric_names(list|tuple):
        summary_writer(SummaryWriter):
        logger:

    """

    logger.debug("Set set_end_epoch_function")

    @trainer.on(Events.EPOCH_COMPLETED)
    def end_epoch_func(engine: Engine):
        """end epoch function

        * end epoch function. Move at the ending of each epoch.

        """
        epoch = engine.state.epoch
        metrics = engine.state.metrics
        save_metrics(epoch, metrics, get_tag_function, metric_names, summary_writer)


def set_valid_function(trainer, evaluator, valid, valid_interval, get_tag_func, *,
                       logger: Logger, summary_writer=None, metric_names, **_):
    """set end function

    Args:
        trainer(Engine):
        evaluator(Engine):
        valid(Iterable):
        valid_interval(int)
        logger(Logger):
        get_tag_func(Callable):
        metric_names(list|tuple):
        summary_writer(SummaryWriter):

    """

    logger.debug("Set set_valid_function")

    @trainer.on(Events.STARTED)
    @trainer.on(Events.EPOCH_COMPLETED(every=valid_interval))
    def valid_func(engine: Engine):
        """valid_func

        * valid function. Move at the ending of each valid_interval.

        """
        evaluator.run(valid)
        epoch = engine.state.epoch
        metrics = evaluator.state.metrics
        save_metrics(epoch, metrics, get_tag_func, metric_names, summary_writer)


def set_write_model_param_function(trainer, model, get_tag_func, get_item_func, *,
                                   logger: Logger, summary_writer=None, param_names, **_):
    """set the function. the function write some parameters per epoch.

    Args:
        trainer(Engine):
        model(nn.Module):
        get_item_func(Callable):
        get_tag_func(Callable):
        logger(Logger):
        summary_writer(SummaryWriter):
        param_names(list|tuple):

    """

    logger.debug("Set write_model_params")

    @trainer.on(Events.EPOCH_COMPLETED)
    def write_model_params(engine: Engine):
        """write model parameters

        """
        epoch = engine.state.epoch
        if summary_writer is not None:
            [summary_writer.add_scalar(get_tag_func(key), get_item_func(key), global_step=epoch)
             for key in param_names if getattr(model, key, None) is not None]


def set_timer_and_logging(trainer: Engine, evaluator: Engine, *, logger: Logger):
    """set timer and set logging function.

    Args:
        evaluator(Engine): evaluator.
        trainer(Engine): trainer.
        logger(Logger): logger.

    Returns:

    """
    assert trainer is not None and evaluator is not None, "Must not None."
    total_timer = Timer(average=False)

    @trainer.on(Events.STARTED)
    def start_train(engine: Engine):
        """start train

        * move only ones (when train started)

        """
        total_timer.reset()
        logger.info("pre training start. epoch length = {}".format(engine.state.max_epochs))

    @trainer.on(Events.COMPLETED)
    def complete_train(engine: Engine):
        """start train

        * move only ones (when train completed)

        """
        epoch = engine.state.epoch
        time_str = elapsed_time_str(total_timer.value())
        logger.info("pre training complete. finish epoch: {:>5}, time: {:>7}".format(epoch, time_str))

    @trainer.on(Events.EPOCH_COMPLETED)
    def print_time_per_epoch(engine: Engine):
        """print time per epoch function

        * end epoch function. Move at the ending of each epoch.

        """
        epoch = engine.state.epoch
        logger.info(
            "----- epoch: {:>5} complete. time: {:>8.2f}. total time: {:>7} -----".format(
                epoch, engine.state.times['EPOCH_COMPLETED'], elapsed_time_str(total_timer.value()))
        )

    @trainer.on(Events.ITERATION_COMPLETED(every=100))
    def print_info_per_some_iter(engine: Engine):
        """print info per some iter function

        * Move at the ending of some iter.

        """
        epoch = engine.state.epoch
        logger.debug("----- epoch: {:>5} iter {:>6} complete. total time: {:>7} -----".format(
            epoch, engine.state.iteration, elapsed_time_str(total_timer.value())))


def make_get_checkpoints(
        model: nn.Module, opt: Union[nn.Module, Optimizer], trainer: Engine, checkpoint_dir: str
) -> tuple[Checkpoint, Checkpoint, dict]:
    """make and get checkpoints

    Args:
        model(nn.Module):
        opt(Union[nn.Module, Optimizer]):
        trainer(Engine):
        checkpoint_dir(str):

    Returns:
        tuple[Checkpoint, Checkpoint, dict]: good_checkpoint, last_checkpoint, to_save

    """
    # about checkpoint
    to_save = {MODEL: model, OPTIMIZER: opt, TRAINER: trainer}
    good_checkpoint = Checkpoint(
        to_save, DiskSaver(MOST_GOOD_CHECKPOINT_PATH.format(checkpoint_dir), require_empty=False),
        global_step_transform=global_step_from_engine(trainer),
        score_name=LOSS, score_function=Checkpoint.get_default_score_fn(LOSS, -1.0))

    last_checkpoint = Checkpoint(
        to_save, DiskSaver(LATEST_CHECKPOINT_PATH.format(checkpoint_dir), require_empty=False),
    )
    to_save.update({GOOD_LOSS_CHECKPOINTE: good_checkpoint, LAST_CHECKPOINTE: last_checkpoint})
    return good_checkpoint, last_checkpoint, to_save


def load_from_checkpoints_if_you_need(
        load_path, is_resume_from_good_checkpoint, is_resume_from_last_checkpoint, to_load, *,
        good_checkpoint: Checkpoint = None, last_checkpoint: Checkpoint = None, logger: Logger
) -> tuple[Checkpoint, Checkpoint, dict]:
    """load from_checkpoints if you need.

    Args:
        load_path(str):
        is_resume_from_good_checkpoint(bool):
        is_resume_from_last_checkpoint(bool):
        to_load(dict):
        good_checkpoint(Checkpoint):
        last_checkpoint(Checkpoint):
        logger(Logger):

    Returns:
        tuple[Checkpoint, Checkpoint, dict]: good_checkpoint, last_checkpoint, loaded_dict

    """
    good_checkpoint_save_handler = good_checkpoint.save_handler
    last_checkpoint_save_handler = last_checkpoint.save_handler
    if is_resume_from_good_checkpoint and is_resume_from_last_checkpoint:
        raise ValueError("resume-from-checkpoint or resume-from-last-point can be 'True', not both.")
        pass
    elif is_resume_from_good_checkpoint:
        if load_path is None: raise "checkpoint_path must not None."
        logger.info(f"----- resume from path: {load_path}")
        Checkpoint.load_objects(to_load=to_load, checkpoint=load_path)

    elif is_resume_from_last_checkpoint:
        if load_path is None:
            raise ValueError("--checkpoint-path must not None.")
            pass
        logger.info(f"----- resume from last. last_path: {load_path}")
        Checkpoint.load_objects(to_load=to_load, checkpoint=load_path)
    good_checkpoint.save_handler = good_checkpoint_save_handler
    last_checkpoint.save_handler = last_checkpoint_save_handler
    return good_checkpoint, last_checkpoint, to_load


def training_with_ignite(
        model, opt, max_epoch, trainer, evaluator, checkpoint_dir, resume_checkpoint_path,
        is_resume_from_good_checkpoint, is_resume_from_last_checkpoint, *, train, device, non_blocking, logger
) -> tuple[Checkpoint, Checkpoint, dict]:
    """training

    Args:
        model:
        opt:
        max_epoch:
        trainer:
        evaluator:
        checkpoint_dir:
        resume_checkpoint_path:
        is_resume_from_good_checkpoint:
        is_resume_from_last_checkpoint:
        train:
        device:
        non_blocking:
        logger:

    Returns:

    """

    def force_cpu_deco(_func: Callable):
        """force_cpu decorator

        Args:
            _func(Callable):

        Returns:
            Callable: the decorated function.

        """

        def _new_func(*_args, **_kwargs):
            with force_cpu(model, device, non_blocking=non_blocking):
                return _func(*_args, **_kwargs)

        return _new_func

    # timer setting
    set_timer_and_logging(trainer, evaluator, logger=logger)
    # checkpoints setting
    good_checkpoint, last_checkpoint, to_load = make_get_checkpoints(model, opt, trainer, checkpoint_dir)
    good_checkpoint, last_checkpoint, to_load = load_from_checkpoints_if_you_need(
        resume_checkpoint_path, is_resume_from_good_checkpoint, is_resume_from_last_checkpoint, to_load,
        good_checkpoint=good_checkpoint, last_checkpoint=last_checkpoint, logger=logger
    )
    trainer.add_event_handler(Events.EPOCH_COMPLETED, last_checkpoint)
    evaluator.add_event_handler(Events.COMPLETED, force_cpu_deco(good_checkpoint)) if evaluator is not None else None

    if max_epoch > trainer.state.epoch:
        trainer.run(train, max_epochs=max_epoch)

    return good_checkpoint, last_checkpoint, to_load
