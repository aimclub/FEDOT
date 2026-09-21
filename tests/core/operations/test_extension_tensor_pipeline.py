from dataclasses import replace
from datetime import timedelta

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import torch

from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.operations.extension_model import ExtensionModel
from fedot.core.operations.extension_transform import ExtensionTransform
from fedot.core.operations.factory import OperationFactory
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.operation_types_repository import OperationTypesRepository
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.extensions import (
    ArrayBackend, ExtensionContractError, ExtensionManifest, ExternalModelSpec,
    ExternalTransformSpec, ModelCapabilities, TransformCapabilities,
    extension_scope,
)


def data():
    features = torch.arange(1, 13, dtype=torch.float64).reshape(6, 2)
    return TensorData(Task(TaskTypesEnum.regression), DataTypesEnum.tabular, features,
                      target=features[:, :1] * 3 + 2, idx=np.array(['a', 'b', 'c', 'd', 'e', 'f']))


def manifest(model_factory=LinearRegression, transform_factory=StandardScaler, backend=ArrayBackend.numpy):
    return ExtensionManifest('tensor_extension', '1', models=(ExternalModelSpec(
        'ext_linear', model_factory, ModelCapabilities(
            (TaskTypesEnum.regression,), (DataTypesEnum.tabular,), tags=('linear',), backend=backend)),),
        transforms=(ExternalTransformSpec('ext_scale', transform_factory, TransformCapabilities(
            (TaskTypesEnum.regression,), (DataTypesEnum.tabular,), DataTypesEnum.tabular,
            tags=('feature_scaling',), backend=backend)),))


@pytest.mark.integration
@pytest.mark.parametrize('time_constraint', [None, timedelta(seconds=10)])
def test_real_cpu_tensor_pipeline_model_and_transform(time_constraint):
    calls = []

    class Model(LinearRegression):
        def fit(self, features, target):
            calls.append('model.fit')
            return super().fit(features, target)

        def predict(self, features):
            calls.append('model.predict')
            return super().predict(features)

    class Transform(StandardScaler):
        def fit(self, features, target=None):
            calls.append('transform.fit')
            return super().fit(features, target)

        def transform(self, features):
            calls.append('transform.transform')
            return super().transform(features)

    def model_factory():
        calls.append('model.factory')
        return Model()

    def transform_factory(*, params):
        calls.append('transform.factory')
        return Transform(**params)

    source = data()
    original = source.features.clone()
    repository = OperationTypesRepository('all')
    builtin_ids = tuple(operation.id for operation in repository.operations)
    with extension_scope(manifest(model_factory, transform_factory)):
        transform = PipelineNode('ext_scale')
        model = PipelineNode('ext_linear', nodes_from=[transform])
        pipeline = Pipeline(model)
        fitted = pipeline.fit(source, time_constraint=time_constraint)
        predicted = pipeline.predict(replace(source, target=None))
        assert isinstance(transform.operation, ExtensionTransform)
        assert isinstance(model.operation, ExtensionModel)
        assert transform.operation._is_tensor_transform_operation()
        assert not model.operation._is_tensor_transform_operation()
        assert isinstance(fitted, TensorData) and isinstance(predicted, TensorData)
        assert fitted.device.type == predicted.device.type == 'cpu'
        np.testing.assert_allclose(fitted.predict.numpy().ravel(), source.target.numpy().ravel(), atol=1e-8)
        np.testing.assert_allclose(predicted.predict.numpy().ravel(), source.target.numpy().ravel(), atol=1e-8)
        np.testing.assert_array_equal(predicted.idx, source.idx)
        assert torch.equal(source.features, original)
        assert source.predict is None
        assert calls == ['transform.factory', 'transform.fit', 'transform.transform',
                         'model.factory', 'model.fit', 'model.predict',
                         'transform.transform', 'model.predict']
        assert tuple(operation.id for operation in repository.operations) == builtin_ids
    assert 'ext_scale' not in OperationTypesRepository('all').suitable_operation()


def test_repository_discovery_separates_kinds_and_respects_scope():
    model_repo = OperationTypesRepository('model')
    transform_repo = OperationTypesRepository('data_operation')
    with extension_scope(manifest()):
        models = model_repo.suitable_operation(task_type=TaskTypesEnum.regression)
        transforms = transform_repo.suitable_operation(task_type=TaskTypesEnum.regression)
        assert 'ext_linear' in models and 'ext_scale' not in models
        assert 'ext_scale' in transforms and 'ext_linear' not in transforms
        assert 'ext_scale' not in transform_repo.suitable_operation(
            tags=['feature_scaling', 'absent'], is_full_match=True)
        assert isinstance(OperationFactory('ext_scale/node1').get_operation(), ExtensionTransform)
    assert 'ext_linear' not in model_repo.suitable_operation()
    assert 'ext_scale' not in transform_repo.suitable_operation()


def test_transform_output_is_features_only_and_preserves_rows_and_target():
    source = data()
    source.predict = torch.ones(6)
    with extension_scope(manifest()):
        operation = OperationFactory('ext_scale').get_operation()
        fitted, result = operation.fit({}, source)
        assert isinstance(result, TensorData)
        assert result.predict is None
        assert torch.equal(result.target, source.target)
        np.testing.assert_array_equal(result.idx, source.idx)
        np.testing.assert_allclose(result.features.mean(0).numpy(), np.zeros(2), atol=1e-10)
        assert source.predict is not None
        assert isinstance(fitted, StandardScaler)


@pytest.mark.parametrize('phase', ['factory', 'fit', 'predict', 'transform'])
def test_internal_type_error_is_single_call_with_original_cause(phase):
    calls = []
    cause = TypeError('user implementation bug')

    def execute(name):
        calls.append(name)
        if phase == name:
            raise cause

    class Implementation:
        def fit(self, features, target=None):
            execute('fit')

        def predict(self, features):
            execute('predict')
            return features[:, :1]

        def transform(self, features):
            execute('transform')
            return features

    def factory(params=None):
        execute('factory')
        return Implementation()

    with extension_scope(manifest(factory, factory)):
        operation = OperationFactory('ext_scale' if phase == 'transform' else 'ext_linear').get_operation()
        with pytest.raises(ExtensionContractError) as error:
            operation.fit({}, data())
        assert error.value.__cause__ is cause
        assert calls.count(phase) == 1


@pytest.mark.parametrize('invalid,code', [('params', 'validation_error'),
                                          ('task', 'unsupported_task'),
                                          ('target', 'target_required'),
                                          ('data_type', 'unsupported_data_type')])
def test_validation_rejects_before_factory(invalid, code):
    calls = []
    source = data()
    params = {}
    if invalid == 'params':
        params = {'undeclared': 1}
    elif invalid == 'task':
        source = replace(source, task=Task(TaskTypesEnum.classification))
    elif invalid == 'target':
        source = replace(source, target=None)
    else:
        source = replace(source, data_type=DataTypesEnum.ts)
    with extension_scope(manifest(lambda: calls.append('factory'))):
        with pytest.raises(ExtensionContractError) as error:
            OperationFactory('ext_linear').get_operation().fit(params, source)
        assert error.value.code == code
        assert calls == []


@pytest.mark.parametrize('output,code', [(None, 'invalid_runtime_output'),
                                         (np.array(['bad'] * 6), 'invalid_runtime_output'),
                                         (np.zeros((2, 1)), 'output_row_mismatch')])
def test_invalid_prediction_is_typed(output, code):
    class Implementation:
        def fit(self, features, target):
            pass

        def predict(self, features):
            return output

    with extension_scope(manifest(Implementation)):
        with pytest.raises(ExtensionContractError) as error:
            OperationFactory('ext_linear').get_operation().fit({}, data())
        assert error.value.code == code


def test_torch_stateless_transform_has_no_fit_call_and_owns_input_buffer():
    class Transform:
        def transform(self, features):
            assert isinstance(features, torch.Tensor)
            features.add_(10)
            return features

    item = manifest(transform_factory=Transform, backend=ArrayBackend.torch)
    transform = item.transforms[0]
    item = replace(item, transforms=(replace(transform, capabilities=replace(
        transform.capabilities, requires_fit=False)),))
    source = data()
    original = source.features.clone()
    with extension_scope(item):
        operation = OperationFactory('ext_scale').get_operation()
        _, result = operation.fit({}, source)
        assert torch.equal(result.features, original + 10)
        assert torch.equal(source.features, original)
