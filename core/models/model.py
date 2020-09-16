from datetime import timedelta

import numpy as np

from core.log import default_log, Log
from core.models.data import InputData
from core.repository.dataset_types import DataTypesEnum
from core.repository.model_types_repository import ModelMetaInfo, ModelTypesRepository
from core.repository.tasks import Task, TaskTypesEnum, compatible_task_types

DEFAULT_PARAMS_STUB = 'default_params'


class Model:
    def __init__(self, model_type: str, log: Log = default_log(__name__)):
        self.model_type = model_type
        self._eval_strategy, self._data_preprocessing = None, None
        self.params = DEFAULT_PARAMS_STUB
        self.log = log

    @property
    def acceptable_task_types(self):
        model_info = ModelTypesRepository().model_info_by_id(self.model_type)
        return model_info.task_type

    def compatible_task_type(self, base_task_type: TaskTypesEnum):
        # if the model can't be used directly for the task type from data
        if base_task_type not in self.acceptable_task_types:
            # search the supplementary task types, that can be included in chain which solves original task
            globally_compatible_task_types = compatible_task_types(base_task_type)
            compatible_task_types_acceptable_for_model = list(set(self.acceptable_task_types).intersection
                                                              (set(globally_compatible_task_types)))
            if len(compatible_task_types_acceptable_for_model) == 0:
                raise ValueError(f'Model {self.model_type} can not be used as a part of {base_task_type}.')
            task_type_for_model = compatible_task_types_acceptable_for_model[0]
            return task_type_for_model
        return base_task_type

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
        else:
            return output_types[0]

    @property
    def description(self):
        model_type = self.model_type
        model_params = self.params
        return f'n_{model_type}_{model_params}'

    def _init(self, task: Task):

        params_for_fit = None
        if self.params != DEFAULT_PARAMS_STUB:
            params_for_fit = self.params

        try:
            self._eval_strategy = _eval_strategy_for_task(self.model_type, task.task_type)(self.model_type,
                                                                                           params_for_fit)
        except Exception as ex:
            self.log.error(f'Can not find evaluation strategy because of {ex}')
            raise Exception

    def fit(self, data: InputData):
        self._init(data.task)

        fitted_model = self._eval_strategy.fit(train_data=data)
        predict_train = self._eval_strategy.predict(trained_model=fitted_model,
                                                    predict_data=data)

        if np.array([np.isnan(_) for _ in predict_train]).any():
            predict_train = np.nan_to_num(predict_train)

        return fitted_model, predict_train

    def predict(self, fitted_model, data: InputData):
        self._init(data.task)

        prediction = self._eval_strategy.predict(trained_model=fitted_model,
                                                 predict_data=data)

        if np.array([np.isnan(_) for _ in prediction]).any():
            return np.nan_to_num(prediction)

        return prediction

    def fine_tune(self, data: InputData, iterations: int,
                  max_lead_time: timedelta = timedelta(minutes=5)):
        self._init(data.task)

        try:
            fitted_model, tuned_params = self._eval_strategy.fit_tuned(train_data=data,
                                                                       iterations=iterations,
                                                                       max_lead_time=max_lead_time)
            self.params = tuned_params
            if not self.params:
                self.params = DEFAULT_PARAMS_STUB
        except Exception as ex:
            print(f'Tuning failed because of {ex}')
            fitted_model = self._eval_strategy.fit(train_data=data)
            self.params = DEFAULT_PARAMS_STUB

        predict_train = self._eval_strategy.predict(trained_model=fitted_model,
                                                    predict_data=data)
        return fitted_model, predict_train

    def __str__(self):
        return f'{self.model_type}'


def _eval_strategy_for_task(model_type: str, task_type_for_data: TaskTypesEnum):
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
