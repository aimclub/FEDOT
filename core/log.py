import json
import logging
from logging.handlers import RotatingFileHandler
import os
import sys
from logging.config import dictConfig

main_log_file_path = os.path.dirname(__file__)


class Logger:
    def __init__(self, logger_name,
                 path=os.path.join(main_log_file_path, 'logging.json')):
        self.logger = logging.getLogger(logger_name)
        self._setup_logger(path)

    def _config(self):
        self.main_log_file = os.path.join(main_log_file_path, 'log.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler = RotatingFileHandler(self.main_log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        console_handler = logging.StreamHandler(sys.stdout)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def _setup_logger(self, path):
        """Setup logging configuration from file"""
        if os.path.exists(path):
            with open(path, 'rt') as file:
                config = json.load(file)
            dictConfig(config)
        else:
            self._config()

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
