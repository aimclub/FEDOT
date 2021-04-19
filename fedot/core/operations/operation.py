from sklearn.impute import SimpleImputer

from fedot.core.data.data import InputData
from fedot.core.log import Log, default_log
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.operation_types_repository import OperationMetaInfo
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.repository.tasks import compatible_task_types
from fedot.core.operations.evaluation.operation_implementations.data_operations.\
    sklearn_transformations import OneHotEncodingImplementation

DEFAULT_PARAMS_STUB = 'default_params'


class Operation:
    """
    Base class for operators in nodes. Operators could be machine learning
    (or statistical) models or data operations

    :param operation_type: name of the operation
    :param log: Log object to record messages
    """

    def __init__(self, operation_type: str, log: Log = None):
        self.operation_type = operation_type
        self.log = log

        self._eval_strategy = None
        self.operations_repo = None
        self.params = DEFAULT_PARAMS_STUB

        if not log:
            self.log = default_log(__name__)
        else:
            self.log = log

    def _init(self, task: Task, **kwargs):
        params_for_fit = None
        if self.params != DEFAULT_PARAMS_STUB:
            params_for_fit = self.params

        try:
            self._eval_strategy = _eval_strategy_for_task(self.operation_type,
                                                          task.task_type,
                                                          self.operations_repo)(
                self.operation_type,
                params_for_fit)
        except Exception as ex:
            self.log.error(f'Can not find evaluation strategy because of {ex}')
            raise ex

        if 'output_mode' in kwargs:
            self._eval_strategy.output_mode = kwargs['output_mode']

    @property
    def description(self):
        operation_type = self.operation_type
        operation_params = self.params
        return f'n_{operation_type}_{operation_params}'

    @property
    def acceptable_task_types(self):
        operation_info = self.operations_repo.operation_info_by_id(
            self.operation_type)
        return operation_info.task_type

    @property
    def metadata(self) -> OperationMetaInfo:
        operation_info = self.operations_repo.operation_info_by_id(self.operation_type)
        if not operation_info:
            raise ValueError(f'Operation {self.operation_type} not found')
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

        data = _fill_remaining_gaps(data, self.operation_type)

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

        data = _fill_remaining_gaps(data, self.operation_type)

        prediction = self._eval_strategy.predict(
            trained_operation=fitted_operation,
            predict_data=data,
            is_fit_chain_stage=is_fit_chain_stage)

        return prediction

    def __str__(self):
        return f'{self.operation_type}'


def _eval_strategy_for_task(operation_type: str, current_task_type: TaskTypesEnum,
                            operations_repo):
    """
    The function returns the strategy for the selected operation and task type.
    And if it is necessary, found acceptable strategy for operation

    :param operation_type: name of operation, for example, 'ridge'
    :param current_task_type: task to solve
    :param operations_repo: repository with operations

    :return strategy: EvaluationStrategy class for this operation
    """

    # Get acceptable task types for operation
    operation_info = operations_repo.operation_info_by_id(operation_type)
    acceptable_task_types = operation_info.task_type

    # If the operation can't be used directly for the task type from data
    set_acceptable_types = set(acceptable_task_types)
    if current_task_type not in acceptable_task_types:

        # Search the supplementary task types, that can be included in chain
        # which solves main task
        globally_compatible_task_types = compatible_task_types(current_task_type)
        globally_set = set(globally_compatible_task_types)

        comp_types_acceptable_for_operation = list(set_acceptable_types.intersection(globally_set))
        if len(comp_types_acceptable_for_operation) == 0:
            raise ValueError(f'Operation {operation_type} can not be used as a part of {current_task_type}.')
        current_task_type = comp_types_acceptable_for_operation[0]

    strategy = operations_repo.operation_info_by_id(operation_type).current_strategy(current_task_type)
    return strategy


def _fill_remaining_gaps(data: InputData, operation_type: str):
    """ Function for filling in the nans in the table with features """
    # TODO discuss: move this "filling" to the chain method - we use such method too much here (for all tables)
    #  np.isnan(features).any() and np.isnan(features) doesn't work with non-numeric arrays
    features = data.features
    is_operation_not_for_text = operation_type != 'text_clean'
    if data.data_type == DataTypesEnum.table and is_operation_not_for_text:
        # Got indices of columns with string objects
        categorical_ids, _ = OneHotEncodingImplementation.str_columns_check(features)

        # Apply most_frequent or mean filling strategy
        if len(categorical_ids) == 0:
            data.features = SimpleImputer().fit_transform(features)
        else:
            data.features = SimpleImputer(strategy='most_frequent').fit_transform(features)
    return data
