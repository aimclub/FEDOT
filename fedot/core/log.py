import json
import os
import sys

import logging
from logging.config import dictConfig
from logging.handlers import RotatingFileHandler
from threading import RLock

from fedot.core.utils import default_fedot_data_dir


DEFAULT_LOG_PATH = os.path.join(default_fedot_data_dir(), 'log.log')


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


class Log(metaclass=SingletonMeta):
    """ Log object to store logger singleton and log adapters
    :param logger_name: name of the logger
    :param config_json_file: json file from which to collect the logger if specified
    :param output_verbosity_level: verbosity level of logger
    :param log_file: file to write logs in """

    __log_adapters = {}

    def __init__(self, logger_name: str,
                 config_json_file: str = 'default',
                 output_verbosity_level: int = logging.DEBUG,
                 log_file: str = None):
        if not log_file:
            self.log_file = os.path.join(default_fedot_data_dir(), 'log.log')
        else:
            self.log_file = log_file
        self.logger = self._get_logger(name=logger_name, config_file=config_json_file,
                                       verbosity_level=output_verbosity_level)

    def get_adapter(self, prefix: str) -> 'LoggerAdapter':
        """ Get adapter to pass contextual information to log messages.
        :param prefix: prefix to log messages with this adapter. Usually this prefix is the name of the class
        where the log came from """
        if prefix not in self.__log_adapters.keys():
            self.__log_adapters[prefix] = LoggerAdapter(self.logger,
                                                        {'class_name': prefix})
        return self.__log_adapters[prefix]

    def _get_logger(self, name, config_file: str, verbosity_level: int) -> logging.Logger:
        """ Get logger object """
        logger = logging.getLogger(name)
        if config_file != 'default':
            self._setup_logger_from_json_file(config_file)
        else:
            logger = self._setup_default_logger(logger, verbosity_level)
        return logger

    def _setup_default_logger(self, logger: logging.Logger, verbosity_level: int) -> logging.Logger:
        """ Define console and file handlers for logger """
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter('%(asctime)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        file_handler = RotatingFileHandler(self.log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)

        logger.setLevel(verbosity_level)

        return logger

    @staticmethod
    def _setup_logger_from_json_file(config_file):
        """ Setup logging configuration from file """
        try:
            with open(config_file, 'rt') as file:
                config = json.load(file)
            dictConfig(config)
        except Exception as ex:
            raise Exception(f'Can not open the log config file because of {ex}')

    @property
    def handlers(self):
        return self.logger.handlers

    def release_handlers(self):
        """This function closes handlers of logger"""
        for handler in self.handlers:
            handler.close()

    def getstate(self):
        """ Define the attributes to be pickled via deepcopy or pickle
        :return: dict: state """
        state = dict(self.__dict__)
        del state['logger']
        return state

    def __str__(self):
        return f'LoggerAdapter object for {self.logger.name} module'

    def __repr__(self):
        return self.__str__()


class LoggerAdapter(logging.LoggerAdapter):
    """ This class looks like logger but used to pass contextual information
    to the output along with logging event information """

    def __init__(self, logger, extra):
        super().__init__(logger=logger, extra=extra)
        self.setLevel(logger.level)
        self.verbosity_level = logger.level

    def process(self, msg, kwargs):
        return '%s - %s' % (self.extra['class_name'], msg), kwargs

    def __str__(self):
        return f'LoggerAdapter object for {self.extra["class_name"]} module'

    def __repr__(self):
        return self.__str__()


def default_log(prefix: str = 'default', verbose_level: int = logging.DEBUG) -> logging.LoggerAdapter:
    """
    Default logger
    :param prefix: adapter prefix to add it to log messages.
    :param verbose_level: level of detailing
    :return: LoggerAdapter: LoggerAdapter object
    """

    log = Log(logger_name='default',
              config_json_file='default',
              output_verbosity_level=verbose_level)

    return log.get_adapter(prefix=prefix)
