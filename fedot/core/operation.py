
from abc import ABC
from datetime import timedelta

import numpy as np

from fedot.core.algorithms.time_series.prediction import post_process_forecasted_ts
from fedot.core.data.data import InputData
from fedot.core.log import Log, default_log
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.model_types_repository import ModelMetaInfo, ModelTypesRepository
from fedot.core.repository.tasks import Task, TaskTypesEnum, compatible_task_types

DEFAULT_PARAMS_STUB = 'default_params'


class AbstractOperation:
    """
    Base object for operators in nodes. Operators could be machine learning
    models or data "preparators"

    """

    @property
    def acceptable_task_types(self):
        model_info = ModelTypesRepository().model_info_by_id(self.model_type)
        return model_info.task_type

    @property
    def metadata(self) -> ModelMetaInfo:
        model_info = ModelTypesRepository().model_info_by_id(self.model_type)
        if not model_info:
            raise ValueError(f'Model {self.model_type} not found')
        return model_info

    def output_datatype(self, input_datatype: DataTypesEnum) -> DataTypesEnum:
        output_types = self.metadata.output_types
        if input_datatype in output_types:
            return input_datatype
        elif input_datatype == DataTypesEnum.ts:
            return DataTypesEnum.forecasted_ts
        else:
            return output_types[0]

    @property
    def description(self):
        model_type = self.model_type
        model_params = self.params
        return f'n_{model_type}_{model_params}'

    def _init(self, task: Task, **kwargs):
        params_for_fit = None
        if self.params != DEFAULT_PARAMS_STUB:
            params_for_fit = self.params

        try:
            self._eval_strategy = _eval_strategy_for_task(self.model_type,
                                                          task.task_type)(
                self.model_type,
                params_for_fit)
        except Exception as ex:
            self.log.error(f'Can not find evaluation strategy because of {ex}')
            raise ex

        if 'output_mode' in kwargs:
            self._eval_strategy.output_mode = kwargs['output_mode']

    def fit(self, data: InputData):
        raise NotImplementedError()

    def predict(self, fitted_model, data: InputData,
                output_mode: str):
        raise NotImplementedError()

    def fine_tune(self, data: InputData, iterations: int,
                  max_lead_time: timedelta):
        raise NotImplementedError()

    def __str__(self):
        return f'{self.model_type}'

# TODO impement class
class DataOperation(AbstractOperation):

    def __init__(self, model_type: str, log: Log = None):
        super().__init__(model_type, log)

    def apply(self):
        """
        Combine fit and predict methods
        """
        pass


# TODO impement class
class Model(AbstractOperation):

    def __init__(self, model_type: str, log: Log = None):
        super().__init__(model_type, log)


class Operation:
    """
    Factory which allows determining what type of operation should be defined
    in the node

    """

    def __init__(self, model_type):
        self.model_type = model_type

    def _define_operation_type(self):
        pass

    def got_model(self):
        pass


def _eval_strategy_for_task(model_type: str, task_type_for_data: TaskTypesEnum):
    """
    The function returns the strategy for the selected model and task type

    """
    models_repo = ModelTypesRepository()
    model_info = models_repo.model_info_by_id(model_type)

    task_type_for_model = task_type_for_data
    task_types_acceptable_for_model = model_info.task_type

    # if the model can't be used directly for the task type from data
    if task_type_for_model not in task_types_acceptable_for_model:
        # search the supplementary task types, that can be included in chain which solves original task
        globally_compatible_task_types = compatible_task_types(task_type_for_model)
        compatible_task_types_acceptable_for_model = list(set(task_types_acceptable_for_model).intersection
                                                          (set(globally_compatible_task_types)))
        if len(compatible_task_types_acceptable_for_model) == 0:
            raise ValueError(f'Model {model_type} can not be used as a part of {task_type_for_model}.')
        task_type_for_model = compatible_task_types_acceptable_for_model[0]

    strategy = models_repo.model_info_by_id(model_type).current_strategy(task_type_for_model)
    return strategy


def _post_process_prediction_using_original_input(prediction, input_data: InputData):
    # TODO add docstring description
    processed_predict = prediction
    if input_data.task.task_type == TaskTypesEnum.ts_forecasting:
        processed_predict = post_process_forecasted_ts(prediction, input_data)
    else:
        if np.array([np.isnan(_) for _ in prediction]).any():
            processed_predict = np.nan_to_num(prediction)

    return processed_predict