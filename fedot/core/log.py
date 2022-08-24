import json
import logging
import pathlib
import sys
from logging.config import dictConfig
from logging.handlers import RotatingFileHandler
from typing import Optional

from fedot.core.utilities.singleton_meta import SingletonMeta
from fedot.core.utils import default_fedot_data_dir

DEFAULT_LOG_PATH = pathlib.Path(default_fedot_data_dir(), 'log.log')


class Log(metaclass=SingletonMeta):
    """ Log object to store logger singleton and log adapters
    :param logger_name: name of the logger
    :param config_json_file: json file from which to collect the logger if specified
    :param output_logging_level: logging levels are the same as in 'logging':
    critical -- 50, error -- 40, warning -- 30, info -- 20, debug -- 10, nonset -- 0.
    Logs with a level HIGHER than set will be displayed.
    :param log_file: file to write logs in """

    __log_adapters = {}

    def __init__(self, logger_name: str,
                 config_json_file: str = 'default',
                 output_logging_level: int = logging.INFO,
                 log_file: str = None,
                 write_logs: bool = True):
        if not log_file:
            self.log_file = pathlib.Path(default_fedot_data_dir(), 'log.log')
        else:
            self.log_file = log_file
        self.logger = self._get_logger(name=logger_name, config_file=config_json_file,
                                       logging_level=output_logging_level,
                                       write_logs=write_logs)

    def get_adapter(self, prefix: str, logging_level: int = logging.INFO) -> 'LoggerAdapter':
        """ Get adapter to pass contextual information to log messages.
        :param prefix: prefix to log messages with this adapter. Usually this prefix is the name of the class
        where the log came from
        :param logging_level: level of logging """
        if prefix not in self.__log_adapters.keys():
            self.__log_adapters[prefix] = LoggerAdapter(self.logger,
                                                        {'prefix': prefix},
                                                        logging_level=logging_level)
        return self.__log_adapters[prefix]

    def _get_logger(self, name, config_file: str, logging_level: int, write_logs: bool) -> logging.Logger:
        """ Get logger object """
        logger = logging.getLogger(name)
        if config_file != 'default':
            self._setup_logger_from_json_file(config_file)
        else:
            logger = self._setup_default_logger(logger=logger, logging_level=logging_level, write_logs=write_logs)
        return logger

    def _setup_default_logger(self, logger: logging.Logger, logging_level: int, write_logs: bool) -> logging.Logger:
        """ Define console and file handlers for logger """
        if not write_logs or logging_level > logging.CRITICAL:
            return logger

        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter('%(asctime)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        file_handler = RotatingFileHandler(self.log_file, maxBytes=100000000, backupCount=1)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)

        logger.setLevel(logging_level)

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
        return f'Log object for {self.logger.name} module'

    def __repr__(self):
        return self.__str__()


class LoggerAdapter(logging.LoggerAdapter):
    """ This class looks like logger but used to pass contextual information
    to the output along with logging event information """

    def __init__(self, logger: logging.Logger, extra: dict, logging_level: int = None):
        super().__init__(logger=logger, extra=extra)
        self.setLevel(logging_level or logger.level)
        self.logging_level = logging_level or logger.level

    def process(self, msg, kwargs):
        self.logger.setLevel(self.logging_level)
        return '%s - %s' % (self.extra['prefix'], msg), kwargs

    def __str__(self):
        return f'LoggerAdapter object for {self.extra["prefix"]} module'

    def __repr__(self):
        return self.__str__()


def default_log(class_object=None, prefix: str = 'default', logging_level: Optional[int] = logging.INFO,
                write_logs: bool = True) -> logging.LoggerAdapter:
    """
    Default logger

    :param class_object: instance of class
    :param prefix: adapter prefix to add it to log messages
    :param logging_level: logging levels are the same as in 'logging'
    :param write_logs: bool indicating whenever to write logs in console or not
    :return: LoggerAdapter: LoggerAdapter object
    """
    log = Log(logger_name='default',
              config_json_file='default',
              output_logging_level=logging_level,
              write_logs=write_logs)

    if class_object:
        prefix = class_object.__class__.__name__

    return log.get_adapter(prefix=prefix, logging_level=log.logger.level)
