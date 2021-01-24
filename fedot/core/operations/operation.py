from abc import abstractmethod
from datetime import timedelta

import numpy as np

from fedot.core.algorithms.time_series.prediction import post_process_forecasted_ts
from fedot.core.data.data import InputData
from fedot.core.log import Log, default_log
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.operation_types_repository import OperationMetaInfo, \
    ModelTypesRepository, DataOperationTypesRepository
from fedot.core.repository.tasks import Task, TaskTypesEnum, compatible_task_types

DEFAULT_PARAMS_STUB = 'default_params'


class Operation:
    """
    Base object for operators in nodes. Operators could be machine learning
    models or data "preparators"

    """

    def __init__(self, model_type: str, log: Log = None):
        self.model_type = model_type
        self.log = log

        self._eval_strategy, self._data_preprocessing = None, None
        self.params = DEFAULT_PARAMS_STUB

        if not log:
            self.log = default_log(__name__)
        else:
            self.log = log

    @property
    def description(self):
        model_type = self.model_type
        model_params = self.params
        return f'n_{model_type}_{model_params}'

    def output_datatype(self, input_datatype: DataTypesEnum) -> DataTypesEnum:
        output_types = self.metadata.output_types
        if input_datatype in output_types:
            return input_datatype
        elif input_datatype == DataTypesEnum.ts:
            return DataTypesEnum.forecasted_ts
        else:
            return output_types[0]

    @abstractmethod
    def fit(self, data: InputData):
        raise NotImplementedError()

    @abstractmethod
    def predict(self, fitted_model, data: InputData,
                output_mode: str):
        raise NotImplementedError()

    @abstractmethod
    def fine_tune(self, data: InputData, iterations: int,
                  max_lead_time: timedelta):
        raise NotImplementedError()

    def __str__(self):
        return f'{self.model_type}'


class Model(Operation):
    """
    Base object with fit/predict methods defining the evaluation strategy for the task

    :param model_type: str type of the model defined in model repository
    :param log: Log object to record messages
    """

    def __init__(self, model_type: str, log: Log = None):
        super().__init__(model_type=model_type, log=log)

    def _init(self, task: Task, **kwargs):
        operations_repo = ModelTypesRepository()
        params_for_fit = None
        if self.params != DEFAULT_PARAMS_STUB:
            params_for_fit = self.params

        try:
            self._eval_strategy = _eval_strategy_for_task(self.model_type,
                                                          task.task_type,
                                                          operations_repo)(
                self.model_type,
                params_for_fit)
        except Exception as ex:
            self.log.error(f'Can not find evaluation strategy because of {ex}')
            raise ex

        if 'output_mode' in kwargs:
            self._eval_strategy.output_mode = kwargs['output_mode']

    @property
    def acceptable_task_types(self):
        model_info = ModelTypesRepository().model_info_by_id(self.model_type)
        return model_info.task_type

    @property
    def metadata(self) -> OperationMetaInfo:
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

    def fit(self, data: InputData):
        """
        This method is used for defining and running of the evaluation strategy
        to train the model with the data provided

        :param data: data used for model training
        :return: tuple of trained model and prediction on train data
        """
        self._init(data.task)

        prepared_data = data.prepare_for_modelling(is_for_fit=True)

        fitted_model = self._eval_strategy.fit(train_data=prepared_data)

        predict_train = self.predict(fitted_model, data)

        return fitted_model, predict_train

    def predict(self, fitted_model, data: InputData, output_mode: str = 'default'):
        """
        This method is used for defining and running of the evaluation strategy
        to predict with the data provided

        :param fitted_model: trained model object
        :param data: data used for prediction
        """
        self._init(data.task, output_mode=output_mode)

        prepared_data = data.prepare_for_modelling(is_for_fit=False)

        prediction = self._eval_strategy.predict(trained_model=fitted_model,
                                                 predict_data=prepared_data)

        prediction = _post_process_prediction_using_original_input(prediction=prediction, input_data=data)

        return prediction

    def fine_tune(self, data: InputData, iterations: int,
                  max_lead_time: timedelta = timedelta(minutes=5)):
        """
        This method is used for hyperparameter searching

        :param data: data used for hyperparameter searching
        :param iterations: max number of iterations evaluable for hyperparameter optimization
        :param max_lead_time: max time(seconds) for tuning evaluation
        """
        self._init(data.task)

        prepared_data = data.prepare_for_modelling(is_for_fit=True)

        try:
            fitted_model, tuned_params = self._eval_strategy.fit_tuned(train_data=prepared_data,
                                                                       iterations=iterations,
                                                                       max_lead_time=max_lead_time)
            if fitted_model is None:
                raise ValueError(f'{self.model_type} can not be fitted')

            self.params = tuned_params
            if not self.params:
                self.params = DEFAULT_PARAMS_STUB
        except Exception as ex:
            print(f'Tuning failed because of {ex}')
            fitted_model = self._eval_strategy.fit(train_data=data)
            self.params = DEFAULT_PARAMS_STUB

        predict_train = self.predict(fitted_model, data)

        return fitted_model, predict_train

    def __str__(self):
        return f'{self.model_type}'


class DataOperation(Operation):

    def __init__(self, model_type: str, log: Log = None):
        super().__init__(model_type, log)

    def _init(self, task: Task, **kwargs):
        operations_repo = DataOperationTypesRepository()
        params_for_fit = None
        if self.params != DEFAULT_PARAMS_STUB:
            params_for_fit = self.params

        try:
            self._eval_strategy = _eval_strategy_for_task(self.model_type,
                                                          task.task_type,
                                                          operations_repo)(
                self.model_type,
                params_for_fit)
        except Exception as ex:
            self.log.error(f'Can not find evaluation strategy because of {ex}')
            raise ex

        if 'output_mode' in kwargs:
            self._eval_strategy.output_mode = kwargs['output_mode']

    @property
    def acceptable_task_types(self):
        model_info = DataOperationTypesRepository().model_info_by_id(self.model_type)
        return model_info.task_type

    @property
    def metadata(self) -> OperationMetaInfo:
        model_info = DataOperationTypesRepository().model_info_by_id(self.model_type)
        if not model_info:
            d=DataOperationTypesRepository()
            print(model_info)
            raise ValueError(f'Data operation {self.model_type} not found')
        return model_info

    def output_datatype(self, input_datatype: DataTypesEnum) -> DataTypesEnum:
        output_types = self.metadata.output_types
        if input_datatype in output_types:
            return input_datatype
        elif input_datatype == DataTypesEnum.ts:
            return DataTypesEnum.forecasted_ts
        else:
            return output_types[0]

    def fit(self, data: InputData):
        """
        This method is used for defining and running of the evaluation strategy
        to train the model with the data provided

        :param data: data used for model training
        :return: tuple of trained model and prediction on train data
        """
        self._init(data.task)

        prepared_data = data.prepare_for_modelling(is_for_fit=True)

        fitted_model = self._eval_strategy.fit(train_data=prepared_data)

        predict_train = self.predict(fitted_model, data)

        return fitted_model, predict_train

    def predict(self, fitted_model, data: InputData,
                output_mode: str = 'default'):
        """
        This method is used for defining and running of the evaluation strategy
        to predict with the data provided

        :param fitted_model: trained model object
        :param data: data used for prediction
        """
        self._init(data.task, output_mode=output_mode)

        prepared_data = data.prepare_for_modelling(is_for_fit=False)

        prediction = self._eval_strategy.predict(trained_model=fitted_model,
                                                 predict_data=prepared_data)

        prediction = _post_process_prediction_using_original_input(
            prediction=prediction, input_data=data)

        return prediction

    def fine_tune(self, data: InputData, iterations: int,
                  max_lead_time: timedelta = timedelta(minutes=5)):
        """
        This method is used for hyperparameter searching

        :param data: data used for hyperparameter searching
        :param iterations: max number of iterations evaluable for hyperparameter optimization
        :param max_lead_time: max time(seconds) for tuning evaluation
        """
        self._init(data.task)

        prepared_data = data.prepare_for_modelling(is_for_fit=True)

        try:
            fitted_model, tuned_params = self._eval_strategy.fit_tuned(
                train_data=prepared_data,
                iterations=iterations,
                max_lead_time=max_lead_time)
            if fitted_model is None:
                raise ValueError(f'{self.model_type} can not be fitted')

            self.params = tuned_params
            if not self.params:
                self.params = DEFAULT_PARAMS_STUB
        except Exception as ex:
            print(f'Tuning failed because of {ex}')
            fitted_model = self._eval_strategy.fit(train_data=data)
            self.params = DEFAULT_PARAMS_STUB

        predict_train = self.predict(fitted_model, data)

        return fitted_model, predict_train

    def __str__(self):
        return f'{self.model_type}'


def _eval_strategy_for_task(model_type: str, task_type_for_data: TaskTypesEnum,
                            operations_repo):
    """
    The function returns the strategy for the selected model and task type
    """

    model_info = operations_repo.model_info_by_id(model_type)

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

    strategy = operations_repo.model_info_by_id(model_type).current_strategy(task_type_for_model)
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