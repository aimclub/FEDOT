from abc import abstractmethod
from typing import Optional, Union, Dict, Any

from fedot.core.data.data import InputData, OutputData
from fedot.core.log import default_log
from fedot.core.operations.hyperparameters_preprocessing import HyperparametersPreprocessor
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.operation_types_repository import OperationMetaInfo
from fedot.core.repository.tasks import Task, TaskTypesEnum, compatible_task_types
from fedot.core.serializers.serializer import register_serializable


@register_serializable
class Operation:
    """Base class for operations in nodes. Operations could be machine learning
    (or statistical) models or data operations

    Args:
        operation_type: name of the operation
    """

    def __init__(self, operation_type: str, **kwargs):
        self.operation_type = operation_type

        self._eval_strategy = None
        self.operations_repo = None
        self.fitted_operation = None

        self.log = default_log(self)

    def _init(self, task: Task, **kwargs):
        params = kwargs.get('params')
        if not params:
            params = OperationParameters.from_operation_type(self.operation_type)
        if isinstance(params, dict):
            params = OperationParameters.from_operation_type(self.operation_type, **params)
        params_for_fit = HyperparametersPreprocessor(operation_type=self.operation_type,
                                                     n_samples_data=kwargs.get('n_samples_data'))\
            .correct(params.to_dict())
        params_for_fit = OperationParameters.from_operation_type(self.operation_type, **params_for_fit)
        try:
            self._eval_strategy = \
                _eval_strategy_for_task(self.operation_type,
                                        task.task_type,
                                        self.operations_repo)(self.operation_type,
                                                              params_for_fit)
        except Exception as ex:
            self.log.error(f'Can not find evaluation strategy because of {ex}')
            raise ex

        if 'output_mode' in kwargs:
            self._eval_strategy.output_mode = kwargs['output_mode']

    def description(self, operation_params: dict) -> str:
        operation_type = self.operation_type
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
            raise ValueError(f'{self.__class__.__name__} {self.operation_type} not found')
        return operation_info

    def fit(self, params: Optional[Union[OperationParameters, dict]], data: InputData):
        """This method is used for defining and running of the evaluation strategy
        to train the operation with the data provided

        Args:
            params: hyperparameters for operation
            data: data used for operation training

        Returns:
            tuple: trained operation and prediction on train data
        """
        self._init(data.task, params=params, n_samples_data=data.features.shape[0])

        self.fitted_operation = self._eval_strategy.fit(train_data=data)

        predict_train = self.predict_for_fit(self.fitted_operation, data, params)

        return self.fitted_operation, predict_train

    def predict(self, fitted_operation, data: InputData, params: Optional[Union[OperationParameters, dict]] = None,
                output_mode: str = 'default'):
        """This method is used for defining and running of the evaluation strategy
        to predict with the data provided

        Args:
            fitted_operation: trained operation object
            data: data used for prediction
            params: hyperparameters for operation
            output_mode: string with information about output of operation,
            for example, is the operation predict probabilities or class labels
        """
        return self._predict(fitted_operation, data, params, output_mode, is_fit_stage=False)

    def predict_for_fit(self, fitted_operation, data: InputData, params: Optional[OperationParameters] = None,
                        output_mode: str = 'default'):
        """This method is used for defining and running of the evaluation strategy
        to predict with the data provided during fit stage

        Args:
            fitted_operation: trained operation object
            data: data used for prediction
            params: hyperparameters for operation
            output_mode: string with information about output of operation,
                for example, is the operation predict probabilities or class labels
        """
        return self._predict(fitted_operation, data, params, output_mode, is_fit_stage=True)

    def _predict(self, fitted_operation, data: InputData, params: Optional[OperationParameters] = None,
                 output_mode: str = 'default', is_fit_stage: bool = False):

        is_main_target = data.supplementary_data.is_main_target
        data_flow_length = data.supplementary_data.data_flow_length
        self._init(data.task, output_mode=output_mode, params=params, n_samples_data=data.features.shape[0])

        if is_fit_stage:
            prediction = self._eval_strategy.predict_for_fit(
                trained_operation=fitted_operation,
                predict_data=data)
        else:
            prediction = self._eval_strategy.predict(
                trained_operation=fitted_operation,
                predict_data=data)
        prediction = self.assign_tabular_column_types(prediction, output_mode)

        if is_main_target is False:
            prediction.supplementary_data.is_main_target = is_main_target

        prediction.supplementary_data.data_flow_length = data_flow_length
        prediction.supplementary_data.was_preprocessed = True
        return prediction

    @staticmethod
    @abstractmethod
    def assign_tabular_column_types(output_data: OutputData, output_mode: str) -> OutputData:
        """Assign types for columns based on task and output_mode (for classification)\n
        For example, pipeline for solving time series forecasting task contains lagged and ridge operations.
        ``ts_type -> lagged -> tabular type``\n
        So, there is a need to assign column types to new data
        """
        raise NotImplementedError()

    def __str__(self):
        return f'{self.operation_type}'

    def to_json(self) -> Dict[str, Any]:
        """Serializes object and ignores unrelevant fields."""
        return {
            k: v
            for k, v in sorted(vars(self).items())
            if k not in ['log', 'operations_repo', '_eval_strategy', 'fitted_operation']
        }


def _eval_strategy_for_task(operation_type: str, current_task_type: TaskTypesEnum,
                            operations_repo):
    """The function returns the strategy for the selected operation and task type.
    And if it is necessary, found acceptable strategy for operation

    Args:
        operation_type: name of operation, for example, ``'ridge'``
        current_task_type: task to solve
        operations_repo: repository with operations

    Returns:
        EvaluationStrategy: ``EvaluationStrategy`` class for this operation
    """

    # Get acceptable task types for operation
    operation_info = operations_repo.operation_info_by_id(operation_type)

    if operation_info is None:
        raise ValueError(f'{operation_type} is not implemented '
                         f'in {operations_repo.repository_name}')

    acceptable_task_types = operation_info.task_type

    # If the operation can't be used directly for the task type from data
    set_acceptable_types = set(acceptable_task_types)
    if current_task_type not in acceptable_task_types:

        # Search the supplementary task types, that can be included in pipeline
        # which solves main task
        globally_compatible_task_types = compatible_task_types(current_task_type)
        globally_set = set(globally_compatible_task_types)

        comp_types_acceptable_for_operation = list(set_acceptable_types.intersection(globally_set))
        if len(comp_types_acceptable_for_operation) == 0:
            raise ValueError(f'Operation {operation_type} can not be used as a part of {current_task_type}.')
        current_task_type = comp_types_acceptable_for_operation[0]

    strategy = operations_repo.operation_info_by_id(operation_type).current_strategy(current_task_type)
    return strategy
