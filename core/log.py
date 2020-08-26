import json
import logging
import os
import sys
from logging.config import dictConfig
from logging.handlers import RotatingFileHandler


def default_log(logger_name: str,
                log_file=os.path.join(os.path.dirname(__file__), 'log.log')) -> 'Log':
    log = Log(logger_name=logger_name,
              config_json_file='default',
              log_file=log_file)
    return log


class Log:
    """A class provides with basic logging object"""

    def __init__(self, logger_name: str,
                 config_json_file: str,
                 log_file: str = os.path.join(os.path.dirname(__file__), 'log.log')):
        """

        :param logger_name:
        :param config_json_file:
        :param log_file:
        """
        self.name = logger_name
        self.log_file = log_file
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
        self.logger.info(message)

    def debug(self, message):
        self.logger.debug(message)

    def warn(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message, exc_info=True)

    def exception(self, message):
        self.logger.exception(message)

    @property
    def handlers(self):
        return self.logger.handlers

    def release_handlers(self):
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
