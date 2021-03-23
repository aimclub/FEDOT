from copy import copy
from typing import Optional

import numpy as np
import tensorflow as tf

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.evaluation import EvaluationStrategy
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import extract_task_param


class AutoMlStrategy(EvaluationStrategy):
    __operations_by_types = {
        'tpot': None,
        'h2o': None}

    def __init__(self, operation_type: str, params: Optional[dict] = None):
        self.operation = self._convert_to_operation(operation_type)
        self.params = params
        super().__init__(operation_type, params)

    def _convert_to_operation(self, operation_type: str):
        if operation_type in self.__operations_by_types.keys():
            return self.__operations_by_types[operation_type]
        else:
            raise ValueError(f'Impossible to obtain AutoMl strategy for {operation_type}')

    @property
    def implementation_info(self) -> str:
        return str(self._convert_to_operation(self.operation_type))