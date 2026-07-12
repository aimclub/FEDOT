from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, Optional, TYPE_CHECKING, Union, Tuple

from golem.core.log import default_log
from golem.serializers.serializer import register_serializable

if TYPE_CHECKING:
    from fedot.core.caching.predictions_cache import PredictionsCache

from fedot.core.data.input_data.data import InputData, OutputData
from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.operations.evaluation.evaluation_interfaces import EvaluationStrategy
from fedot.core.operations.hyperparameters_preprocessing import HyperparametersPreprocessor
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.operation_types_repository import OperationMetaInfo
from fedot.core.repository.operation_types_repository import OperationTypesRepository
from fedot.core.repository.tasks import Task, TaskTypesEnum, compatible_task_types
from fedot.utilities.custom_errors import AbstractMethodNotImplementError


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
        self.operations_repo: Optional[OperationTypesRepository] = None
        self.fitted_operation = None

        self.log = default_log(self)

    def _init(self, task: Task, **kwargs):
        params = kwargs.get('params')
        if not params:
            params = OperationParameters.from_operation_type(
                self.operation_type)
        if isinstance(params, dict):
            params = OperationParameters.from_operation_type(
                self.operation_type, **params)
        params_for_fit = HyperparametersPreprocessor(operation_type=self.operation_type,
                                                     n_samples_data=kwargs.get('n_samples_data')) \
            .correct(params.to_dict())
        params_for_fit = OperationParameters.from_operation_type(
            self.operation_type, **params_for_fit)
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
        operation_info = self.operations_repo.operation_info_by_id(
            self.operation_type)
        if not operation_info:
            raise ValueError(
                f'{self.__class__.__name__} {self.operation_type} not found')
        return operation_info

    def fit(self,
                       params: Optional[Union[OperationParameters, dict]],
                       data: TensorData,
                       predictions_cache: Optional[PredictionsCache] = None,
                       fold_id: Optional[int] = None,
                       descriptive_id: Optional[str] = None) -> Tuple[Any, TensorData]:
        """Trains the operation on TensorData and returns the node output."""
        self._init(
            data.task,
            params=params,
            n_samples_data=data.features.shape[0],
        )

        self.fitted_operation = self._eval_strategy.fit(train_data=data)

        output = self.predict_for_fit(
            fitted_operation=self.fitted_operation,
            data=data,
            params=params,
            predictions_cache=predictions_cache,
            fold_id=fold_id,
        )
        return self.fitted_operation, output

    def _is_tensor_transform_operation(self) -> bool:
        from fedot.core.operations.data_operation import DataOperation

        return isinstance(self, DataOperation)
    
    def predict(self,
                           fitted_operation,
                           data: TensorData,
                           params: Optional[OperationParameters] = None,
                           output_mode: str = 'default',
                           predictions_cache: Optional[PredictionsCache] = None,
                           fold_id: Optional[int] = None,
                           descriptive_id: Optional[str] = None) -> TensorData:
        return self._predict(
            fitted_operation=fitted_operation,
            data=data, params=params,
            output_mode=output_mode, 
            is_fit_stage=False,
            predictions_cache=predictions_cache,
            fold_id=fold_id,
            descriptive_id=descriptive_id,
        )
    
    def predict_for_fit(self,
                                   fitted_operation,
                                   data: TensorData,
                                   params: Optional[OperationParameters] = None,
                                   output_mode: str = 'default',
                                   predictions_cache: Optional[PredictionsCache] = None,
                                   fold_id: Optional[int] = None,
                                   descriptive_id: Optional[str] = None) -> TensorData:
        return self._predict(
            fitted_operation=fitted_operation,
            data=data,
            params=params,
            output_mode=output_mode,
            is_fit_stage=True,
            predictions_cache=predictions_cache,
            fold_id=fold_id,
            descriptive_id=descriptive_id,
        )

    def _predict(self,
                           fitted_operation,
                           data: TensorData,
                           params: Optional[OperationParameters] = None,
                           output_mode: str = 'default',
                           is_fit_stage: bool = False,
                           predictions_cache: Optional[PredictionsCache] = None,
                           fold_id: Optional[int] = None,
                           descriptive_id: Optional[str] = None) -> TensorData:
        self._init(
            data.task,
            output_mode=output_mode,
            params=params,
            n_samples_data=data.features.shape[0],
        )

        result_data = None

        if predictions_cache is not None:
            result_data = predictions_cache.load_node_prediction(
                descriptive_id, output_mode, fold_id, is_fit=is_fit_stage)
        
        # TODO @romankuklo: change with new cache implementation (if/else)
        if result_data is None:
            if is_fit_stage:
                result_data = self._eval_strategy.predict_for_fit(
                    trained_operation=fitted_operation,
                    predict_data=data)
            else:
                result_data = self._eval_strategy.predict(
                    trained_operation=fitted_operation,
                    predict_data=data)
        
        if predictions_cache is not None:
            predictions_cache.save_node_prediction(
                descriptive_id, output_mode, fold_id, result_data, is_fit=is_fit_stage)

        return result_data

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
                            operations_repo: OperationTypesRepository):
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
        globally_compatible_task_types = compatible_task_types(
            current_task_type)
        globally_set = set(globally_compatible_task_types)

        comp_types_acceptable_for_operation = sorted(
            list(set_acceptable_types.intersection(globally_set)))
        if len(comp_types_acceptable_for_operation) == 0:
            raise ValueError(
                f'Operation {operation_type} can not be used as a part of {current_task_type}.')
        current_task_type = comp_types_acceptable_for_operation[0]

    strategy = operations_repo.operation_info_by_id(
        operation_type).current_strategy(current_task_type)
    return strategy
