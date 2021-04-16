import json
import logging
import os
import sys
from functools import wraps
from logging.config import dictConfig
from logging.handlers import RotatingFileHandler
from threading import RLock
from typing import Optional

from fedot.core.utils import default_fedot_data_dir


class SingletonMeta(type):
    """
    This meta class can provide other classes with the Singleton pattern.
    It guarantees to create one and only class instance.
    Pass it to the metaclass parameter when defining your class as follows:

    class YourClassName(metaclass=SingletonMeta)
    """
    _instances = {}

    _lock: RLock = RLock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]


class LogManager(metaclass=SingletonMeta):
    __logger_dict = {}

    def __init__(self):
        pass

    def get_logger(self, name, config_file: str, log_file: str = None):
        if name not in self.__logger_dict.keys():
            self.__logger_dict[name] = logging.getLogger(name)
            if config_file != 'default':
                self._setup_logger_from_json_file(config_file)
            else:
                self._setup_default_logger(log_file, name)

        return self.__logger_dict[name]

    def _setup_default_logger(self, log_file, logger_name):
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler = RotatingFileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        console_handler = logging.StreamHandler(sys.stdout)
        self.__logger_dict[logger_name].setLevel(logging.DEBUG)
        self.__logger_dict[logger_name].addHandler(file_handler)
        self.__logger_dict[logger_name].addHandler(console_handler)

    def _setup_logger_from_json_file(self, config_file):
        """Setup logging configuration from file"""
        try:
            with open(config_file, 'rt') as file:
                config = json.load(file)
            dictConfig(config)
        except Exception as ex:
            raise Exception(f'Can not open the log config file because of {ex}')

    @property
    def debug(self):
        """Returns the information about available loggers"""
        debug_info = {
            'loggers_number': len(self.__logger_dict),
            'loggers_names': [self.__logger_dict.keys()],
            'loggers': [self.__logger_dict.values()]
        }
        return debug_info

    def clear_cache(self):
        self.__logger_dict.clear()


def default_log(logger_name: str,
                log_file: Optional[str] = None,
                verbose_level: int = 2) -> 'Log':
    """
    :param logger_name: string name for logger
    :param log_file: path to the file where log messages will be recorded to
    :param verbose_level level of detalization
    :return Log: Log object
    """
    if not log_file:
        log_file = os.path.join(default_fedot_data_dir(), 'log.log')
    log = Log(logger_name=logger_name,
              config_json_file='default',
              log_file=log_file,
              output_verbosity_level=verbose_level)
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
                 output_verbosity_level=1,
                 log_file: str = None, ):
        if not log_file:
            self.log_file = os.path.join(default_fedot_data_dir(), 'log.log')
        else:
            self.log_file = log_file

        self.name = logger_name
        self.config_file = config_json_file
        self.logger = LogManager().get_logger(logger_name,
                                              config_file=self.config_file,
                                              log_file=self.log_file)
        self.verbosity_level = output_verbosity_level

    def message(self, message):
        """Record the message to user"""
        for_verbosity = 1
        if self.verbosity_level >= for_verbosity:
            self.logger.info(message)

    def info(self, message):
        """Record the INFO log message"""
        for_verbosity = 2
        if self.verbosity_level >= for_verbosity:
            self.logger.info(message)

    def debug(self, message):
        """Record the DEBUG log message"""
        for_verbosity = 3
        if self.verbosity_level >= for_verbosity:
            self.logger.debug(message)

    def ext_debug(self, message):
        """Record the extended DEBUG log message"""
        for_verbosity = 4
        if self.verbosity_level >= for_verbosity:
            self.logger.debug(message)

    def warn(self, message):
        """Record the WARN log message"""
        for_verbosity = 2
        if self.verbosity_level >= for_verbosity:
            self.logger.warning(message)

    def error(self, message):
        """Record the ERROR log message"""
        for_verbosity = 0
        if self.verbosity_level >= for_verbosity:
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

    def __str__(self):
        return f'Log object for {self.name} module'

    def __repr__(self):
        return self.__str__()


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
