from typing import Optional, Union

from fedot.core.operations.evaluation.custom import CustomModelStrategy
from fedot.core.operations.hyperparameters_preprocessing import HyperparametersPreprocessor
from fedot.core.operations.model import Model
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.operation_types_repository import OperationMetaInfo
from fedot.extensions.runtime_rules import (
    build_extension_strategy_params,
    get_extension_acceptable_task_types,
    get_extension_data_types,
)


class ExtensionModel(Model):
    """Runtime adapter for manifest-registered external models.

    It reuses FEDOT's existing custom-model evaluation strategy while keeping
    the external model name as the public operation type.
    """

    def _init(self, task, **kwargs):
        params = kwargs.get('params')
        if not params:
            params = OperationParameters()
        if isinstance(params, dict):
            params = OperationParameters(**params)

        user_params = HyperparametersPreprocessor(
            operation_type='custom',
            n_samples_data=kwargs.get('n_samples_data'),
        ).correct(params.to_dict())
        strategy_params = build_extension_strategy_params(
            operation_name=self.operation_type,
            user_params=user_params,
            output_mode=kwargs.get('output_mode', 'default'),
        )
        params_for_fit = OperationParameters.from_operation_type('custom', **strategy_params)
        self._eval_strategy = CustomModelStrategy('custom', params_for_fit)
        if 'output_mode' in kwargs:
            self._eval_strategy.output_mode = kwargs['output_mode']

    @property
    def acceptable_task_types(self):
        return get_extension_acceptable_task_types(self.operation_type)

    @property
    def metadata(self) -> OperationMetaInfo:
        data_types = list(get_extension_data_types(self.operation_type))
        task_types = list(get_extension_acceptable_task_types(self.operation_type))
        return OperationMetaInfo(
            id=self.operation_type,
            input_types=data_types,
            output_types=data_types,
            task_type=task_types,
            supported_strategies={task_type: CustomModelStrategy for task_type in task_types},
            allowed_positions=['any'],
            tags=['external', 'custom_model'],
            presets=['best_quality', 'fast_train', 'stable', 'gpu', 'automl', 'ts'],
        )
