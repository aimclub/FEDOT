from abc import abstractmethod

from fedot.core.data.data import InputData
from fedot.core.log import Log, default_log
from fedot.core.repository.operation_types_repository import \
    OperationMetaInfo, OperationTypesRepository
from fedot.core.repository.tasks import Task, TaskTypesEnum, compatible_task_types

DEFAULT_PARAMS_STUB = 'default_params'


class Operation:
    """
    Base class for operators in nodes. Operators could be machine learning
    (or statistical) models or data operations

    """

    def __init__(self, operation_type: str, log: Log = None):
        self.operation_type = operation_type
        self.log = log

        self._eval_strategy = None
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

    @abstractmethod
    def fit(self, data: InputData, is_fit_chain_stage: bool):
        raise NotImplementedError()

    @abstractmethod
    def predict(self, fitted_operation, data: InputData,
                is_fit_chain_stage: bool, output_mode: str = 'default'):
        raise NotImplementedError()

    def __str__(self):
        return f'{self.operation_type}'


class Model(Operation):
    """
    Class with fit/predict methods defining the evaluation strategy for the task

    :param operation_type: str type of the model defined in model repository
    :param log: Log object to record messages
    """

    def __init__(self, operation_type: str, log: Log = None):
        super().__init__(operation_type=operation_type, log=log)

    def _init(self, task: Task, **kwargs):
        operations_repo = OperationTypesRepository()
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
        operations_repo = OperationTypesRepository()
        model_info = operations_repo.operation_info_by_id(self.operation_type)
        return model_info.task_type

    @property
    def metadata(self) -> OperationMetaInfo:
        operations_repo = OperationTypesRepository()
        model_info = operations_repo.operation_info_by_id(self.operation_type)
        if not model_info:
            raise ValueError(f'Model {self.operation_type} not found')
        return model_info

    def fit(self, data: InputData, is_fit_chain_stage: bool = True):
        """
        This method is used for defining and running of the evaluation strategy
        to train the model with the data provided

        :param data: data used for model training
        :return: tuple of trained model and prediction on train data
        :param is_fit_chain_stage: is this fit or predict stage for chain
        """
        self._init(data.task)

        fitted_model = self._eval_strategy.fit(train_data=data)

        predict_train = self.predict(fitted_model, data, is_fit_chain_stage)

        return fitted_model, predict_train

    def predict(self, fitted_operation, data: InputData,
                is_fit_chain_stage: bool, output_mode: str = 'default'):
        """
        This method is used for defining and running of the evaluation strategy
        to predict with the data provided

        :param fitted_operation: trained model object
        :param data: data used for prediction
        :param is_fit_chain_stage: is this fit or predict stage for chain
        :param output_mode: string with information about output of operation,
        for example, is the operation predict probabilities or class labels
        """
        self._init(data.task, output_mode=output_mode)

        prediction = self._eval_strategy.predict(trained_operation=fitted_operation,
                                                 predict_data=data,
                                                 is_fit_chain_stage=is_fit_chain_stage)

        return prediction


class DataOperation(Operation):
    """
    Class with fit/predict methods defining the evaluation strategy for the task

    :param operation_type: str type of the data operation defined in data_operation
    repository
    :param log: Log object to record messages
    """

    def __init__(self, operation_type: str, log: Log = None):
        super().__init__(operation_type, log)

    def _init(self, task: Task, **kwargs):
        operations_repo = OperationTypesRepository(repository_name='data_operation_repository.json')
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
        operations_repo = OperationTypesRepository(repository_name='data_operation_repository.json')
        operation_info = operations_repo.operation_info_by_id(self.operation_type)
        return operation_info.task_type

    @property
    def metadata(self) -> OperationMetaInfo:
        operations_repo = OperationTypesRepository(repository_name='data_operation_repository.json')
        operation_info = operations_repo.operation_info_by_id(self.operation_type)
        if not operation_info:
            raise ValueError(f'Data operation {self.operation_type} not found')
        return operation_info

    def fit(self, data: InputData, is_fit_chain_stage: bool = True):
        """
        This method is used for defining and running of the evaluation strategy
        to train the operation with the data provided

        :param data: data used for operation training
        :return: tuple of trained operation and prediction on train data
        :param is_fit_chain_stage: is this fit or predict stage for chain
        """
        self._init(data.task)

        fitted_operation = self._eval_strategy.fit(train_data=data)

        predict_train = self.predict(fitted_operation, data, is_fit_chain_stage)

        return fitted_operation, predict_train

    def predict(self, fitted_operation, data: InputData,
                is_fit_chain_stage: bool, output_mode: str = 'default'):
        """
        This method is used for defining and running of the evaluation strategy
        to predict with the data provided

        :param fitted_operation: trained operation object
        :param data: data used for prediction
        :param is_fit_chain_stage: is this fit or predict stage for chain
        :param output_mode: string with information about output of operation,
        for example, is the operation predict probabilities or class labels
        """
        self._init(data.task, output_mode=output_mode)

        prediction = self._eval_strategy.predict(trained_operation=fitted_operation,
                                                 predict_data=data,
                                                 is_fit_chain_stage=is_fit_chain_stage)

        return prediction


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

        set_types_acceptable_for_operation = set(task_types_acceptable_for_operation)
        globally_set = set(globally_compatible_task_types)
        comp_types_acceptable_for_operation = list(set_types_acceptable_for_operation.intersection(globally_set))
        if len(comp_types_acceptable_for_operation) == 0:
            raise ValueError(f'Operation {operation_type} can not be used as a part of {task_type_for_operation}.')
        task_type_for_operation = comp_types_acceptable_for_operation[0]

    strategy = operations_repo.operation_info_by_id(operation_type).current_strategy(task_type_for_operation)
    return strategy
