import logging
import tracemalloc
from typing import Optional

from golem.core.log import default_log


class MemoryAnalytics:
    is_active = False
    active_session_label = 'main'

    @classmethod
    def start(cls):
        """
        Start memory monitoring session
        """
        cls.is_active = True
        tracemalloc.start()

    @classmethod
    def finish(cls):
        """
        Finish memory monitoring session
        """
        cls.log(additional_info='finish')
        tracemalloc.stop()
        cls.is_active = False

    @classmethod
    def get_measures(cls):
        """
        Estimates Python-related system memory consumption in MiB
        :return: current and maximal consumption
        """
        current_memory, max_memory = tracemalloc.get_traced_memory()
        return current_memory / 1024 / 1024, max_memory / 1024 / 1024

    @classmethod
    def log(cls, logger: Optional[logging.LoggerAdapter] = None,
            additional_info: str = 'location', logging_level: int = logging.INFO) -> str:
        """
        Print the message about current and maximal memory consumption to the log or console.
        :param logger: optional logger that should be used in output.
        :param additional_info: label for current location in code.
        :param logging_level: level of the message
        :return: text of the message.
        """
        message = ''
        if cls.is_active:
            memory_consumption = cls.get_measures()
            message = f'Memory consumption for {additional_info} in {cls.active_session_label} session: ' \
                      f'current {round(memory_consumption[0], 1)} MiB, ' \
                      f'max: {round(memory_consumption[1], 1)} MiB'
            if logger is None:
                logger = default_log(prefix=cls.__name__)
            logger.log(logging_level, message)
        return message
