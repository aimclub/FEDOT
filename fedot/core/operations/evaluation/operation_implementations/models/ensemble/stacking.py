from abc import abstractmethod
from typing import Optional

import numpy as np
from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import \
    ModelImplementation
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.tasks import TaskTypesEnum

from sklearn.linear_model import (
    Ridge, LogisticRegression
)

from fedot.utilities.custom_errors import AbstractMethodNotImplementError


class StackingIndustrialImplementation(ModelImplementation):
    """Class ensemble predictions."""

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)

    def fit(self, input_data: InputData):
        """ Method fit model on a dataset

        :param input_data: data with features, target and ids to process
        """
        return self

    @abstractmethod
    def predict(self, input_data: InputData) -> OutputData:
        """Abstract method. Should be override in child class"""
        raise AbstractMethodNotImplementError
