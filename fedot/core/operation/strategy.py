
from abc import ABC
from datetime import timedelta

import numpy as np

from fedot.core.algorithms.time_series.prediction import post_process_forecasted_ts
from fedot.core.data.data import InputData
from fedot.core.log import Log, default_log
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.model_types_repository import ModelMetaInfo, ModelTypesRepository
from fedot.core.operation.operation import Model, Preprocessing


class StrategyOperator:
    """
    Base class for determining what type of operation should be defined
    in the node

    """

    def __init__(self, model_type):
        self.model_type = model_type
        self.operation_type = self._define_operation_type()

    def get_model(self):
        """
        Factory method returns the desired object of the 'Preprocessing' or
        'Model' class which depends on model_type variable

        """

        if self.operation_type == 'model':
            print('model')
            operator = Model(model_type=self.model_type)
        else:
            operator = Preprocessing(model_type=self.model_type)

        return operator

    def _define_operation_type(self) -> str:
        """
        The method determines what type of operation is set for this node

        :return : operation type 'model' or 'preprocessing'
        TODO need to add a flag for whether preprocessing is used in the node
         or not
        """

        # Get available models
        models_repo = ModelTypesRepository()
        models = models_repo.models

        # If there is a such model in the list
        if any(self.model_type == model.id for model in models):
            operation_type = 'model'
        # Overwise - it is preprocessing operation
        else:
            operation_type = 'preprocessing'
        return operation_type