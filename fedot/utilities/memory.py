import logging
import tracemalloc
from typing import Optional

import numpy as np
import pandas as pd
from golem.core.log import default_log

from fedot.preprocessing.data_types import ID_TO_TYPE


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


def reduce_mem_usage(features, initial_types):
    df = pd.DataFrame(features)
    types_array = [ID_TO_TYPE[_type] for _type in initial_types]

    for index, col in enumerate(df.columns):
        df[col] = df[col].astype(types_array[index])
        col_type = df[col].dtype.name

        if col_type not in ['object', 'category', 'datetime64[ns, UTC]']:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    return df
