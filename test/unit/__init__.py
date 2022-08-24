import logging

from fedot.core.log import Log

logging_level = logging.DEBUG
Log(logger_name='unit_test', console_logging_level=logging_level, file_logging_level=logging_level)
