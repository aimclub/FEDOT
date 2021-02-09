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
    (or statistical) models or data operations

    """

    def __init__(self, operation_type: str, log: Log = None):
        self.operation_type = operation_type
        self.log = log

        self._eval_strategy, self._data_preprocessing = None, None
        self.params = DEFAULT_PARAMS_STUB

        if not log:
            self.log = default_log(__name__)
        else:
            self.log = log

    @property
    def description(self):
        operation_type = self.operation_type
        operation_params = self.params
        return f'n_{operation_type}_{operation_params}'

    def output_datatype(self, input_datatype: DataTypesEnum) -> DataTypesEnum:
        output_types = self.metadata.output_types
        if input_datatype in output_types:
            return input_datatype
        elif input_datatype == DataTypesEnum.ts:
            return DataTypesEnum.forecasted_ts
        else:
            return output_types[0]

    @abstractmethod
    def fit(self, data: InputData, is_fit_chain_stage: bool):
        raise NotImplementedError()

    @abstractmethod
    def predict(self, fitted_operation, data: InputData, is_fit_chain_stage: bool,
                output_mode: str):
        raise NotImplementedError()

    @abstractmethod
    def fine_tune(self, data: InputData, iterations: int,
                  max_lead_time: timedelta):
        raise NotImplementedError()

    def __str__(self):
        return f'{self.operation_type}'


class Model(Operation):
    """
    Base object with fit/predict methods defining the evaluation strategy for the task

    :param operation_type: str type of the model defined in model repository
    :param log: Log object to record messages
    """

    def __init__(self, operation_type: str, log: Log = None):
        super().__init__(operation_type=operation_type, log=log)

    def _init(self, task: Task, **kwargs):
        operations_repo = ModelTypesRepository()
        params_for_fit = None
        if self.params != DEFAULT_PARAMS_STUB:
            params_for_fit = self.params

        try:
            self._eval_strategy = _eval_strategy_for_task(self.operation_type,
                                                          task.task_type,
                                                          operations_repo)(
                self.operation_type,
                params_for_fit)
        except Exception as ex:
            self.log.error(f'Can not find evaluation strategy because of {ex}')
            raise ex

        if 'output_mode' in kwargs:
            self._eval_strategy.output_mode = kwargs['output_mode']

    @property
    def acceptable_task_types(self):
        model_info = ModelTypesRepository().operation_info_by_id(self.operation_type)
        return model_info.task_type

    @property
    def metadata(self) -> OperationMetaInfo:
        model_info = ModelTypesRepository().operation_info_by_id(self.operation_type)
        if not model_info:
            raise ValueError(f'Model {self.operation_type} not found')
        return model_info

    def output_datatype(self, input_datatype: DataTypesEnum) -> DataTypesEnum:
        output_types = self.metadata.output_types
        if input_datatype in output_types:
            return input_datatype
        elif input_datatype == DataTypesEnum.ts:
            return DataTypesEnum.forecasted_ts
        else:
            return output_types[0]

    def fit(self, data: InputData, is_fit_chain_stage: bool = True):
        """
        This method is used for defining and running of the evaluation strategy
        to train the model with the data provided

        :param data: data used for model training
        :return: tuple of trained model and prediction on train data
        :param is_fit_chain_stage: is this fit or predict stage for chain
        """
        print(f'{self.operation_type} fit stage')
        self._init(data.task)

        prepared_data = data.prepare_for_modelling(is_for_fit=True)

        fitted_model = self._eval_strategy.fit(train_data=prepared_data)

        predict_train = self.predict(fitted_model, data, is_fit_chain_stage)

        return fitted_model, predict_train

    def predict(self, fitted_operation, data: InputData, is_fit_chain_stage: bool,
                output_mode: str = 'default'):
        """
        This method is used for defining and running of the evaluation strategy
        to predict with the data provided

        :param fitted_operation: trained model object
        :param data: data used for prediction
        :param is_fit_chain_stage: is this fit or predict stage for chain
        """
        self._init(data.task, output_mode=output_mode)

        prepared_data = data.prepare_for_modelling(is_for_fit=False)

        prediction = self._eval_strategy.predict(trained_operation=fitted_operation,
                                                 predict_data=prepared_data,
                                                 is_fit_chain_stage=False)

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
                raise ValueError(f'{self.operation_type} can not be fitted')

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
        return f'{self.operation_type}'


class DataOperation(Operation):

    def __init__(self, operation_type: str, log: Log = None):
        super().__init__(operation_type, log)

    def _init(self, task: Task, **kwargs):
        operations_repo = DataOperationTypesRepository()
        params_for_fit = None
        if self.params != DEFAULT_PARAMS_STUB:
            params_for_fit = self.params

        try:
            self._eval_strategy = _eval_strategy_for_task(self.operation_type,
                                                          task.task_type,
                                                          operations_repo)(
                self.operation_type,
                params_for_fit)
        except Exception as ex:
            self.log.error(f'Can not find evaluation strategy because of {ex}')
            raise ex

        if 'output_mode' in kwargs:
            self._eval_strategy.output_mode = kwargs['output_mode']

    @property
    def acceptable_task_types(self):
        operation_info = DataOperationTypesRepository().operation_info_by_id(self.operation_type)
        return operation_info.task_type

    @property
    def metadata(self) -> OperationMetaInfo:
        operation_info = DataOperationTypesRepository().operation_info_by_id(self.operation_type)
        if not operation_info:
            d=DataOperationTypesRepository()
            print(operation_info)
            raise ValueError(f'Data operation {self.operation_type} not found')
        return operation_info

    def output_datatype(self, input_datatype: DataTypesEnum) -> DataTypesEnum:
        output_types = self.metadata.output_types
        if input_datatype in output_types:
            return input_datatype
        elif input_datatype == DataTypesEnum.ts:
            return DataTypesEnum.forecasted_ts
        else:
            return output_types[0]

    def fit(self, data: InputData, is_fit_chain_stage: bool = True):
        """
        This method is used for defining and running of the evaluation strategy
        to train the operation with the data provided

        :param data: data used for operation training
        :return: tuple of trained operation and prediction on train data
        :param is_fit_chain_stage: is this fit or predict stage for chain
        """
        print(f'Fit model {self.operation_type}, {is_fit_chain_stage}')
        self._init(data.task)

        prepared_data = data.prepare_for_modelling(is_for_fit=True)

        fitted_operation = self._eval_strategy.fit(train_data=prepared_data)

        predict_train = self.predict(fitted_operation, data, is_fit_chain_stage)

        return fitted_operation, predict_train

    def predict(self, fitted_operation, data: InputData, is_fit_chain_stage: bool,
                output_mode: str = 'default'):
        """
        This method is used for defining and running of the evaluation strategy
        to predict with the data provided

        :param fitted_operation: trained operation object
        :param data: data used for prediction
        :param is_fit_chain_stage: is this fit or predict stage for chain
        """
        print(f'{self.operation_type}, {is_fit_chain_stage}')
        self._init(data.task, output_mode=output_mode)

        prepared_data = data.prepare_for_modelling(is_for_fit=False)

        prediction = self._eval_strategy.predict(trained_operation=fitted_operation,
                                                 predict_data=prepared_data,
                                                 is_fit_chain_stage=is_fit_chain_stage)

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
            fitted_operation, tuned_params = self._eval_strategy.fit_tuned(
                train_data=prepared_data,
                iterations=iterations,
                max_lead_time=max_lead_time)
            if fitted_operation is None:
                raise ValueError(f'{self.operation_type} can not be fitted')

            self.params = tuned_params
            if not self.params:
                self.params = DEFAULT_PARAMS_STUB
        except Exception as ex:
            print(f'Tuning failed because of {ex}')
            fitted_operation = self._eval_strategy.fit(train_data=data)
            self.params = DEFAULT_PARAMS_STUB

        predict_train = self.predict(fitted_operation, data)

        return fitted_operation, predict_train

    def __str__(self):
        return f'{self.operation_type}'


def _eval_strategy_for_task(operation_type: str, task_type_for_data: TaskTypesEnum,
                            operations_repo):
    """
    The function returns the strategy for the selected operation and task type
    """

    operation_info = operations_repo.operation_info_by_id(operation_type)

    task_type_for_operation = task_type_for_data
    task_types_acceptable_for_operation = operation_info.task_type

    # if the operation can't be used directly for the task type from data
    if task_type_for_operation not in task_types_acceptable_for_operation:
        # search the supplementary task types, that can be included in chain which solves original task
        globally_compatible_task_types = compatible_task_types(task_type_for_operation)
        compatible_task_types_acceptable_for_operation = list(set(task_types_acceptable_for_operation).intersection(set(globally_compatible_task_types)))
        if len(compatible_task_types_acceptable_for_operation) == 0:
            raise ValueError(f'Operation {operation_type} can not be used as a part of {task_type_for_operation}.')
        task_type_for_operation = compatible_task_types_acceptable_for_operation[0]

    strategy = operations_repo.operation_info_by_id(operation_type).current_strategy(task_type_for_operation)
    return strategy


def _post_process_prediction_using_original_input(prediction, input_data: InputData):
    # TODO add docstring description
    processed_predict = prediction
    if input_data.task.task_type == TaskTypesEnum.ts_forecasting:
        processed_predict = post_process_forecasted_ts(prediction, input_data)
    # else:
    #     if np.array([np.isnan(_) for _ in prediction]).any():
    #         processed_predict = np.nan_to_num(prediction)
    # TODO у меня возникают проблемы во время этой проверки после encoding'а
    # + я не очень понимаю зачем она, ведь при нормальной работе моделей и
    # методов предобработки пропусков вообще появляться не должно. Если смысл
    # в том, чтобы заполнять пропуски в исходных данных. то эта операция
    # дублирует Imputation стратегию -> надо разобраться

    return processed_predict