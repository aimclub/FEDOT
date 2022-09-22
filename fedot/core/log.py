import json
import logging
import multiprocessing
import pathlib
import sys
from logging.config import dictConfig
from logging.handlers import RotatingFileHandler
from typing import Optional, Tuple, Union

from fedot.core.utilities.singleton_meta import SingletonMeta
from fedot.core.utils import default_fedot_data_dir

DEFAULT_LOG_PATH = pathlib.Path(default_fedot_data_dir(), 'log.log')


class Log(metaclass=SingletonMeta):
    """Log object to store logger singleton and log adapters

    Args:
        logger_name: name of the logger
        config_json_file: ``json`` file from which to collect the logger if specified
        output_logging_level: logging levels are the same as in 'logging'

            .. details:: more details..

                * ``50`` -> critical
                * ``40`` -> error
                * ``30`` -> warning
                * ``20`` -> info
                * ``10`` -> debug
                * ``0`` -> nonset

                Attention!
                    logs with a level HIGHER than set will be displayed

        log_file: file to write logs in
    """

    __log_adapters = {}

    def __init__(self,
                 config_json_file: str = 'default',
                 output_logging_level: int = logging.INFO,
                 log_file: Optional[Union[str, pathlib.Path]] = None,
                 use_console: bool = True):
        self.log_file = log_file or DEFAULT_LOG_PATH
        self.logger = self._get_logger(config_file=config_json_file,
                                       logging_level=output_logging_level,
                                       use_console=use_console)

    @staticmethod
    def setup_in_mp(logging_level: int, logs_dir: pathlib.Path):
        """
        Preserves logger level and its records in a separate file for each process only if it's a child one

        Args:
            logging_level: level of the logger from the main process
            logs_dir: path to the logs directory
        """

        cur_proc = multiprocessing.current_process().name
        log_file_name = logs_dir.joinpath(f'log_{cur_proc}.log')
        Log(output_logging_level=logging_level, log_file=log_file_name, use_console=False)

    def get_parameters(self) -> Tuple[int, pathlib.Path]:
        return self.logger.level, pathlib.Path(self.log_file).parent

    def reset_logging_level(self, logging_level: int):
        """ Resets logging level for logger and its handlers """
        # Resets logging level is needed because before initialization with API params Singleton
        # can be initialized somewhere else with default ones
        self.logger.setLevel(logging_level)
        for handler in self.handlers:
            handler.setLevel(logging_level)

    def get_adapter(self, prefix: str) -> 'LoggerAdapter':
        """ Get adapter to pass contextual information to log messages

        Args:
            prefix: prefix to log messages with this adapter. Usually, the prefix is the name of the class
                where the log came from
        """

        if prefix not in self.__log_adapters.keys():
            self.__log_adapters[prefix] = LoggerAdapter(self.logger,
                                                        {'prefix': prefix})
        return self.__log_adapters[prefix]

    def _get_logger(self, config_file: str, logging_level: int, use_console: bool = True) -> logging.Logger:
        """ Get logger object """
        logger = logging.getLogger()
        if config_file != 'default':
            self._setup_logger_from_json_file(config_file)
        else:
            logger = self._setup_default_logger(logger=logger, logging_level=logging_level, use_console=use_console)
        return logger

    def _setup_default_logger(self, logger: logging.Logger, logging_level: int,
                              use_console: bool = True) -> logging.Logger:
        """ Define console and file handlers for logger
        """

        if use_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging_level)
            console_formatter = logging.Formatter('%(asctime)s - %(message)s')
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)

        file_handler = RotatingFileHandler(self.log_file, maxBytes=100000000, backupCount=1)
        file_handler.setLevel(logging_level)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)

        logger.setLevel(logging_level)

        return logger

    @staticmethod
    def _setup_logger_from_json_file(config_file):
        """ Setup logging configuration from file
        """

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
        """ This function closes handlers of logger
        """

        for handler in self.handlers:
            handler.close()

    def getstate(self):
        """ Define the attributes to be pickled via deepcopy or pickle

        Returns:
            dict: ``dict`` of state
        """

        state = dict(self.__dict__)
        del state['logger']
        return state

    def __str__(self):
        return f'Log object for {self.logger.name} module'

    def __repr__(self):
        return self.__str__()


class LoggerAdapter(logging.LoggerAdapter):
    """ This class looks like logger but used to pass contextual information
    to the output along with logging event information
    """

    def __init__(self, logger: logging.Logger, extra: dict):
        super().__init__(logger=logger, extra=extra)
        self.logging_level = logger.level
        self.setLevel(self.logging_level)

    def process(self, msg, kwargs):
        self.logger.setLevel(self.logging_level)
        return '%s - %s' % (self.extra['prefix'], msg), kwargs

    def message(self, message: str):
        """ Record the message to user.
        Message is an intermediate logging level between info and warning
        to display main info about optimization process """
        message_logging_level = 45
        if message_logging_level >= self.logging_level:
            self.critical(msg=message)

    def __str__(self):
        return f'LoggerAdapter object for {self.extra["prefix"]} module'

    def __repr__(self):
        return self.__str__()


def default_log(prefix: Optional[object] = 'default') -> 'LoggerAdapter':
    """ Default logger

    Args:
        class_object: instance of class
        prefix: adapter prefix to add it to log messages
        logging_level: logging levels are the same as in 'logging'
        write_logs: bool indicating whenever to write logs in console or not

    Returns:
        :obj:`LoggerAdapter`: :obj:`LoggerAdapter` object

    """

    # get log prefix
    if not isinstance(prefix, str):
        prefix = prefix.__class__.__name__

    return Log().get_adapter(prefix=prefix)
