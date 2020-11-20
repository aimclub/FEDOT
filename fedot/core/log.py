import json
import logging
import os
import sys
from functools import wraps
from logging.config import dictConfig
from logging.handlers import RotatingFileHandler

from fedot.core.utils import default_fedot_data_dir


def default_log(logger_name: str,
                log_file=None) -> 'Log':
    """
    :param logger_name: string name for logger
    :param log_file: path to the file where log messages will be recorded to
    :return Log: Log object
    """
    if not log_file:
        log_file = os.path.join(default_fedot_data_dir(), 'log.log')
    log = Log(logger_name=logger_name,
              config_json_file='default',
              log_file=log_file)
    return log


class Log:
    """
    This class provides with basic logging object

    :param str logger_name: name of the logger object
    :param str config_json_file: json file with configuration for logger setup
    :param str log_file: file where log messages are recorded to
    """

    def __init__(self, logger_name: str,
                 config_json_file: str,
                 log_file: str = None):
        if not log_file:
            self.log_file = os.path.join(default_fedot_data_dir(), 'log.log')
        else:
            self.log_file = log_file

        self.name = logger_name
        self.config_file = config_json_file
        self.logger = logging.getLogger(self.name)

        if self.config_file != 'default':
            self._setup_logger_from_json_file()
        else:
            self._setup_default_logger()

    def _setup_default_logger(self):
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler = RotatingFileHandler(self.log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        console_handler = logging.StreamHandler(sys.stdout)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def _setup_logger_from_json_file(self):
        """Setup logging configuration from file"""
        try:
            with open(self.config_file, 'rt') as file:
                config = json.load(file)
            dictConfig(config)
        except Exception as ex:
            raise Exception(f'Can not open the log config file because of {ex}')

    def info(self, message):
        """Record the INFO log massage"""
        self.logger.info(message)

    def debug(self, message):
        """Record the DEBUG log massage"""
        self.logger.debug(message)

    def warn(self, message):
        """Record the WARN log massage"""
        self.logger.warning(message)

    def error(self, message):
        """Record the ERROR log massage"""
        self.logger.error(message, exc_info=True)

    @property
    def handlers(self):
        return self.logger.handlers

    def release_handlers(self):
        """This function closes handlers of logger"""
        for handler in self.handlers:
            handler.close()

    def __getstate__(self):
        """
        Define the attributes to be pickled via deepcopy or pickle

        :return: dict: state
        """
        state = dict(self.__dict__)
        del state['logger']
        return state

    def __setstate__(self, state):
        """
        Restore an unpickled dict state and assign state items
        to the new instance’s dictionary.

        :param state: pickled class attributes
        """
        self.__dict__.update(state)
        self.logger = logging.getLogger(self.name)


def start_end_log_decorator(start_msg='Starting...', end_msg='Finished'):
    def decorator(method):
        @wraps(method)
        def wrapper(*args, **kwargs):
            args[0].log.info(f'{start_msg}')
            value = method(*args, **kwargs)
            args[0].log.info(f'{end_msg}')
            return value

        return wrapper

    return decorator
