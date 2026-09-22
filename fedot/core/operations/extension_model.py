from fedot.core.operations.evaluation.extensions import ExtensionModelStrategy
from fedot.core.operations.model import Model
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.operation_types_repository import OperationMetaInfo
from fedot.extensions.data_type_rules import build_extension_data_type_view
from fedot.extensions.runtime_rules import require_extension_spec


class ExtensionOperationMixin:
    """Common wiring; model and transform retain distinct operation base classes."""

    def _init(self, task, **kwargs):
        params = kwargs.get('params')
        if params is None:
            params = OperationParameters()
        elif isinstance(params, dict):
            params = OperationParameters(**params)
        self._eval_strategy = self.strategy_type(self.operation_type, params)
        self._eval_strategy.output_mode = kwargs.get('output_mode', 'default')

    @property
    def acceptable_task_types(self):
        return require_extension_spec(self.operation_type).capabilities.tasks

    @property
    def metadata(self) -> OperationMetaInfo:
        spec = require_extension_spec(self.operation_type)
        caps = spec.capabilities
        input_types = list(build_extension_data_type_view(
            caps.data_types).input_types)
        output_types = ((caps.output_data_type,)
                        if caps.output_data_type is not None else caps.data_types)
        return OperationMetaInfo(
            id=self.operation_type,
            input_types=input_types,
            output_types=list(build_extension_data_type_view(
                output_types).input_types),
            task_type=list(caps.tasks),
            supported_strategies={
                task: self.strategy_type for task in caps.tasks},
            allowed_positions=['any'],
            tags=list(dict.fromkeys(('external',) + caps.tags)),
            presets=[],
        )


class ExtensionModel(ExtensionOperationMixin, Model):
    """TensorData model adapter for the canonical extension registry."""

    strategy_type = ExtensionModelStrategy
