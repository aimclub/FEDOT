from dataclasses import replace
import pickle

import numpy as np
import pytest
import torch
from sklearn.multiclass import OneVsRestClassifier

from fedot.core.backend.array_conversions import as_numpy
from fedot.core.data.input_data.data import InputData, OutputData
from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.operations.implementations.models.estimator_adapters import LDAImplementation
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.operations.strategies.models.adapters import (
    AdapterClassificationStrategy, AdapterRegressionStrategy,
)
from fedot.core.operations.strategies.models.keras import KerasClassificationStrategy
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.operation_types_repository import OperationTypesRepository
from fedot.core.repository.tasks import Task, TaskTypesEnum


def make_data(operation='knn', n_classes=2, device='cpu'):
    features = torch.randn(48, 4, generator=torch.Generator().manual_seed(43))
    regression = operation == 'knnreg'
    target = features[:, 0] + features[:, 1] if regression else torch.arange(48) % n_classes * 3 + 2
    return TensorData(features=features, target=target, idx=torch.arange(48),
                      data_type=DataTypesEnum.tabular,
                      task=Task(TaskTypesEnum.regression if regression else TaskTypesEnum.classification)).to(device)


@pytest.mark.parametrize('operation', ['knn', 'svc', 'lda', 'qda'])
@pytest.mark.parametrize('n_classes', [2, 3])
def test_classification_modes_labels_metadata_and_pickle(operation, n_classes):
    data = make_data(operation, n_classes)
    strategy = AdapterClassificationStrategy(operation, OperationParameters.from_operation_type(operation))
    fitted = pickle.loads(pickle.dumps(strategy.fit(data)))
    inference_data = replace(data, target=None)
    for mode in ['labels', 'probs', 'full_probs', 'default']:
        strategy.output_mode = mode
        output = strategy.predict(fitted, inference_data)
        expected = (fitted.model.predict(data.features.numpy()) if mode == 'labels'
                    else fitted.model.predict_proba(data.features.numpy()))
        if n_classes == 2 and mode in ['probs', 'default']:
            expected = expected[:, 1]
        torch.testing.assert_close(output.predict, torch.as_tensor(expected))
        torch.testing.assert_close(strategy.predict_for_fit(fitted, inference_data).predict, output.predict)
        torch.testing.assert_close(output.idx, data.idx)
        assert output.features is inference_data.features
        assert output.preparation_state is inference_data.preparation_state
        assert output.target is None
        assert inference_data.predict is None
    if operation == 'svc':
        assert isinstance(fitted.model, OneVsRestClassifier)


@pytest.mark.parametrize('operation', ['knn', 'knnreg'])
@pytest.mark.parametrize('n_samples,k,expected_k', [(1, 12.4, 1), (3, 12.4, 2), (3, 2.6, 3), (3, .2, 1)])
def test_knn_preserves_neighbor_rounding_and_small_sample_correction(operation, n_samples, k, expected_k):
    data = make_data(operation)
    data = replace(data, features=data.features[:n_samples], target=data.target[:n_samples], idx=data.idx[:n_samples])
    strategy_class = AdapterRegressionStrategy if operation == 'knnreg' else AdapterClassificationStrategy
    strategy = strategy_class(operation, OperationParameters(n_neighbors=k))
    fitted = strategy.fit(data)
    strategy.output_mode = 'labels'

    assert fitted.model.n_neighbors == expected_k
    assert strategy.predict(fitted, data).predict.shape == (n_samples,)


@pytest.mark.parametrize('target_columns', [0, 1, 2])
def test_knn_regression_preserves_native_target_shape(target_columns):
    data = make_data('knnreg')
    if target_columns:
        data.target = torch.stack([data.target * (i + 1) for i in range(target_columns)], dim=1)
    strategy = AdapterRegressionStrategy('knnreg')
    fitted = strategy.fit(data)
    output = strategy.predict_for_fit(fitted, data)

    assert output.predict.shape == data.target.shape
    torch.testing.assert_close(output.predict, torch.as_tensor(fitted.model.predict(data.features.numpy())))


@pytest.mark.parametrize('operation', ['knn', 'svc'])
def test_multilabel_probabilities_keep_every_target(operation):
    data = make_data(operation)
    data.target = torch.stack((torch.arange(48) % 2, torch.arange(48) // 2 % 2), dim=1)
    strategy = AdapterClassificationStrategy(operation, OperationParameters.from_operation_type(operation))
    fitted = strategy.fit(data)
    native = fitted.model.predict_proba(data.features.numpy())
    expected = (np.stack(native, axis=1) if operation == 'knn'
                else np.stack((1 - native, native), axis=-1))
    strategy.output_mode = 'full_probs'
    full = strategy.predict(fitted, replace(data, target=None))
    strategy.output_mode = 'probs'
    positive = strategy.predict(fitted, data)

    assert full.predict.shape == (48, 2, 2)
    assert positive.predict.shape == (48, 2)
    torch.testing.assert_close(full.predict, torch.as_tensor(expected))
    torch.testing.assert_close(positive.predict, torch.as_tensor(expected[..., 1]))


@pytest.mark.parametrize('solver,expected_shrinkage', [('svd', None), ('lsqr', 'auto')])
def test_lda_preserves_solver_shrinkage_adaptation(solver, expected_shrinkage):
    implementation = LDAImplementation(OperationParameters(solver=solver, shrinkage='auto'))
    implementation.fit(make_data('lda'))

    assert implementation.model.solver == solver
    assert implementation.model.shrinkage == expected_shrinkage
    assert implementation.params.get('shrinkage') == expected_shrinkage


def test_lda_preserves_singular_data_fallback_and_nonfinite_cleanup():
    data = make_data('lda')
    features = torch.tensor([[1.], [0.], [1.], [1.], [1.], [0.]])
    data = replace(data, features=features, target=features.long(), idx=list(range(6)))
    implementation = LDAImplementation()
    implementation.fit(data)

    assert implementation.model.solver == 'lsqr'
    assert implementation.params.get('solver') == 'lsqr'
    assert torch.isfinite(implementation.predict_proba(data)).all()


@pytest.mark.parametrize('operation', ['knn', 'knnreg', 'svc', 'lda', 'qda'])
def test_adapters_fit_in_tensor_pipeline(operation):
    data = make_data(operation)
    pipeline = Pipeline(PipelineNode(operation))
    pipeline.fit(data)
    output = pipeline.predict(replace(data, target=None))

    assert isinstance(output, TensorData)
    assert output.predict.shape == (48,)
    assert torch.isfinite(output.predict).all()


def test_regression_strategy_keeps_implementation_fit_stage_hook():
    data = make_data('knnreg')
    strategy = AdapterRegressionStrategy('knnreg')

    class FitStageModel:
        def predict_for_fit(self, data):
            return torch.ones_like(data.target)

        def predict(self, data):
            return torch.zeros_like(data.target)

    model = FitStageModel()
    assert strategy.predict_for_fit(model, data).predict.sum() == len(data.target)
    assert strategy.predict(model, data).predict.sum() == 0


def test_cnn_keeps_its_legacy_strategy_and_output_contract(monkeypatch):
    class CNNStub:
        classes_ = np.array([0, 1])

        def __init__(self, params):
            pass

        def fit(self, data):
            assert isinstance(data, InputData)

        def predict_proba(self, data):
            return np.tile([.25, .75], (len(data.idx), 1))

    monkeypatch.setitem(KerasClassificationStrategy._operations_by_types, 'cnn', CNNStub)
    info = OperationTypesRepository('model').operation_info_by_id('cnn')
    assert info.supported_strategies is KerasClassificationStrategy
    assert 'tensor-only' not in info.tags
    assert 'cnn' not in AdapterClassificationStrategy._operations_by_types
    data = InputData.from_numpy(np.ones((3, 4)), np.array([0, 1, 0]))
    strategy = KerasClassificationStrategy('cnn')
    strategy.output_mode = 'probs'
    output = strategy.predict(strategy.fit(data), data)

    assert isinstance(output, OutputData)
    np.testing.assert_array_equal(output.predict, np.full(3, .75))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is unavailable')
@pytest.mark.parametrize('operation', ['knn', 'knnreg', 'svc', 'lda', 'qda'])
def test_cpu_adapters_accept_cuda_data_and_return_to_input_device(operation):
    data = make_data(operation, device='cuda')
    strategy_class = AdapterRegressionStrategy if operation == 'knnreg' else AdapterClassificationStrategy
    strategy = strategy_class(operation, OperationParameters.from_operation_type(operation))
    strategy.output_mode = 'full_probs'
    fitted = strategy.fit(data)
    output = strategy.predict(fitted, replace(data, target=None))
    expected = fitted.model.predict_proba(as_numpy(data.features)) if operation != 'knnreg' else (
        fitted.model.predict(as_numpy(data.features)))

    assert output.predict.device == data.device
    torch.testing.assert_close(output.predict.cpu(), torch.as_tensor(expected))
