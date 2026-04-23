import numpy as np

from fedot.core.operations.extension_model import ExtensionModel
from fedot.core.operations.factory import OperationFactory
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.extensions.contracts import (
    ExtensionManifest,
    ExternalModelSpec,
    ModelCapabilities,
    ModelHyperparamsSchema,
)
from fedot.extensions.registry import clear_extension_registry, register_extension


class _ExternalEstimator:
    def __init__(self, params=None):
        self.params = params or {}
        self.was_fitted = False

    def fit(self, features, target):
        self.was_fitted = True
        return self

    def predict(self, features):
        return np.ones(features.shape[0])


def _make_manifest():
    return ExtensionManifest(
        name='factory_extension',
        version='1.0.0',
        models=(
            ExternalModelSpec(
                name='external_factory_model',
                factory=lambda params=None: _ExternalEstimator(params),
                capabilities=ModelCapabilities(
                    tasks=(TaskTypesEnum.regression,),
                    data_types=(DataTypesEnum.table,),
                    tags=('external',),
                ),
                hyperparams_schema=ModelHyperparamsSchema(
                    required=(),
                    optional=('beta',),
                    defaults={'beta': 0.5},
                ),
            ),
        ),
    )


def test_operation_factory_returns_extension_model_for_registered_operation():
    clear_extension_registry()
    register_extension(_make_manifest())

    try:
        operation = OperationFactory('external_factory_model').get_operation()

        assert isinstance(operation, ExtensionModel)
        assert OperationFactory('external_factory_model').operation_type_name == 'extension_model'
    finally:
        clear_extension_registry()


def test_extension_model_uses_custom_strategy_adapter_for_runtime_init():
    clear_extension_registry()
    register_extension(_make_manifest())

    try:
        model = ExtensionModel('external_factory_model')
        task = Task(TaskTypesEnum.regression)
        model._init(task, params={'alpha': 2.0}, output_mode='default', n_samples_data=4)

        strategy = model._eval_strategy
        implementation = strategy.fit(type('Data', (), {
            'idx': np.arange(4),
            'features': np.array([[1.0], [2.0], [3.0], [4.0]]),
            'target': np.array([[1.0], [2.0], [3.0], [4.0]]),
            'task': task,
            'data_type': DataTypesEnum.table,
        })())

        metadata = model.metadata

        assert strategy.operation_id == 'custom'
        assert implementation.fitted_model.was_fitted is True
        assert implementation.params.get('alpha') == 2.0
        assert implementation.params.get('beta') == 0.5
        assert metadata.input_types == [DataTypesEnum.table]
        assert metadata.output_types == [DataTypesEnum.table]
    finally:
        clear_extension_registry()
