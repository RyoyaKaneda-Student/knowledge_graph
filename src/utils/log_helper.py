from logging import Logger
from typing import Literal


class NoOutputLogger:
    def info(self, *_args, **_kwargs):
        pass

    def debug(self, *_args, **_kwargs):
        pass


def logger_is_optional(func):
    """
    this is decorator
    """

    def wrapper(*args, **kwargs):
        if 'logger' not in kwargs:
            kwargs['logger'] = NoOutputLogger
        return func(*args, **kwargs)

    return wrapper


class LoggerStartEnd:
    def __init__(self, logger: Logger, start_item_name: str, hyphen_num=5, log_level: Literal['info', 'debug'] = 'info',
                 start_message='start.', end_message='end.'
                 ):
        self.start_item_name = start_item_name
        self.hyphen_num = hyphen_num
        self.log_level = log_level
        self.start_message = start_message
        self.end_message = end_message
        if log_level == 'info':
            self.log = logger.info
        elif log_level == 'debug':
            self.log = logger.debug
        else:
            raise "error."

    def __enter__(self):
        self.log(f"{'-' * self.hyphen_num}{self.start_item_name} {self.start_message} {'-' * self.hyphen_num}")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.log(f"{'-' * self.hyphen_num}{self.start_item_name} {self.end_message} {'-' * self.hyphen_num}")
