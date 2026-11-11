from dataclasses import replace
from importlib.machinery import ModuleSpec
import pickle
import sys
from types import ModuleType

import numpy as np
import pytest
import torch

from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.operations.strategies.models import foundation
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum


STRATEGIES = {
    'tabpfn': foundation.TabPFNClassificationStrategy,
    'tabpfnreg': foundation.TabPFNRegressionStrategy,
    'tabicl': foundation.TabICLClassificationStrategy,
    'tabiclreg': foundation.TabICLRegressionStrategy,
}


class RecordingEstimator:
    """Exercise the public array boundary without downloading or loading weights."""

    def __init__(self, **params):
        self.params = params

    def fit(self, X, y):
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        self.features = X.copy()
        self.target = y.copy()
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.full(len(X), self.classes_[-1])

    def predict_proba(self, X):
        probabilities = np.arange(1, len(self.classes_) + 1, dtype=np.float32)
        return np.tile(probabilities / probabilities.sum(), (len(X), 1))


@pytest.fixture(autouse=True)
def external_estimators(monkeypatch):
    for package, class_names in [
        ('tabpfn', ['TabPFNClassifier', 'TabPFNRegressor']),
        ('tabicl', ['TabICLClassifier', 'TabICLRegressor']),
    ]:
        module = ModuleType(package)
        module.__spec__ = ModuleSpec(package, loader=None)
        for name in class_names:
            setattr(module, name, RecordingEstimator)
        monkeypatch.setitem(sys.modules, package, module)


def make_data(regression=False, n_classes=2):
    features = torch.randn(12, 4, generator=torch.Generator().manual_seed(17))
    target = features[:, 0] if regression else torch.arange(12) % n_classes * 3 + 2
    return TensorData(features=features, target=target, idx=list(range(12)),
                      task=Task(TaskTypesEnum.regression if regression else TaskTypesEnum.classification),
                      data_type=DataTypesEnum.tabular, numerical_idx=list(range(4)))


def parameters(operation, **overrides):
    params = OperationParameters.from_operation_type(operation, **overrides)
    if operation.startswith('tabpfn'):
        params.update(model_path=None)
    return params


@pytest.mark.parametrize('operation', ['tabpfn', 'tabicl'])
@pytest.mark.parametrize('n_classes', [2, 3])
def test_classification_modes_preserve_labels_and_metadata(operation, n_classes):
    data = make_data(n_classes=n_classes)
    strategy = STRATEGIES[operation](operation, parameters(operation))
    fitted = pickle.loads(pickle.dumps(strategy.fit(data)))
    inference = replace(data, target=None)
    torch.testing.assert_close(fitted.classes_, torch.unique(data.target))
    for mode in ['labels', 'probs', 'full_probs', 'default']:
        strategy.output_mode = mode
        output = strategy.predict_for_fit(fitted, inference)
        expected = (fitted.model.predict(data.features.numpy()) if mode == 'labels'
                    else fitted.model.predict_proba(data.features.numpy()))
        if n_classes == 2 and mode in ['probs', 'default']:
            expected = expected[:, 1]
        torch.testing.assert_close(output.predict, torch.as_tensor(expected))
        assert output.features is data.features
        assert output.target is None
        assert output.idx == inference.idx
        assert output.preparation_state is data.preparation_state
        assert inference.predict is None


@pytest.mark.parametrize('operation', list(STRATEGIES))
@pytest.mark.parametrize('device', ['cpu', 'cuda:2'])
def test_one_model_id_prepares_device_and_preserves_parameters(operation, device):
    regression = operation.endswith('reg')
    data = make_data(regression)
    data.target = data.target[:, None]
    params = parameters(operation)
    before = params.to_dict()
    strategy = STRATEGIES[operation](operation, params)
    strategy.params_for_fit = strategy.prepare_execution_params(params.copy(), torch.device(device))
    fitted = strategy.fit(data)

    assert fitted.model.params['device'] == device
    assert fitted.model.target.shape == (12,)
    assert 'max_samples' not in fitted.model.params
    assert 'max_features' not in fitted.model.params
    assert params.to_dict() == before
    expected_limit = (10000 if device.startswith('cuda') else 1000) if operation.startswith('tabpfn') else (
        30000 if device.startswith('cuda') else 5000)
    assert strategy.params_for_fit.get('max_samples') == expected_limit
    if device == 'cpu':
        node = PipelineNode(operation)
        node.parameters = before
        pipeline = Pipeline(node)
        pipeline.fit(data)
        output = pipeline.predict(replace(data, target=None))
        assert isinstance(output, TensorData)
        assert output.predict.shape == (12,)
        assert node.fitted_operation.model.params['device'] == 'cpu'


@pytest.mark.parametrize('operation', ['tabpfn', 'tabpfnreg'])
def test_tabpfn_restores_categories_and_recomputes_indices_after_ohe(operation):
    data = make_data(operation.endswith('reg'))
    categories = np.array(['red', 'blue'] * 6, dtype=object)[:, None]
    categories[3, 0] = np.nan
    data = replace(data, numerical_idx=[0, 3], categorical_idx=[1, 2],
                   idx_mapping={0: 0, 1: 1, 2: 1, 3: 2},
                   categorical_features=categories, categorical_source_idx=[1])
    strategy = STRATEGIES[operation](operation, parameters(operation, enable_categorical=True))
    fitted = strategy.fit(data)

    assert fitted.model.categorical_features_indices == [2]
    np.testing.assert_array_equal(fitted.model.features[:, :2].astype(np.float32), data.features[:, [0, 3]])
    assert fitted.model.features[0, 2] == 'red'
    assert np.isnan(fitted.model.features[3, 2])
    assert data.categorical_idx == [1, 2]
    assert data.features.shape == (12, 4)

    inference_categories = categories.copy()
    inference_categories[0, 0] = 'unseen'
    inference = replace(data, target=None, categorical_features=inference_categories)
    prepared = fitted._prepare_features(inference)
    assert prepared[0, 2] == 'unseen'
    assert np.isnan(prepared[3, 2])
    assert fitted.model.categorical_features_indices == [2]


def test_tabpfn_category_recovery_supports_category_only_input():
    data = make_data()
    categories = np.array(['a', 'b'] * 6, dtype=object)[:, None]
    data = replace(data, categorical_idx=[0, 1, 2, 3], numerical_idx=[],
                   idx_mapping={i: 0 for i in range(4)},
                   categorical_features=categories, categorical_source_idx=[0])
    strategy = foundation.TabPFNClassificationStrategy('tabpfn', parameters('tabpfn', enable_categorical=True))
    fitted = strategy.fit(data)

    assert fitted.model.features.shape == (12, 1)
    assert fitted.model.categorical_features_indices == [0]
    np.testing.assert_array_equal(fitted.model.features, categories)


@pytest.mark.parametrize('enable_categorical', [False, True])
def test_numeric_tabpfn_does_not_rebuild_numeric_storage(enable_categorical):
    data = make_data()
    strategy = foundation.TabPFNClassificationStrategy(
        'tabpfn', parameters('tabpfn', enable_categorical=enable_categorical))
    fitted = strategy.fit(data)
    features = fitted._prepare_features(data)

    assert features.dtype == np.float32
    assert np.shares_memory(features, data.features.numpy())


@pytest.mark.parametrize('limit,maximum', [('max_samples', 5), ('max_features', 2)])
def test_existing_size_limits_reject_data_without_truncation(limit, maximum):
    data = make_data()
    strategy = foundation.TabICLClassificationStrategy('tabicl', parameters('tabicl', **{limit: maximum}))
    with pytest.raises(ValueError, match='too many'):
        strategy.fit(data)
    assert data.features.shape == (12, 4)


def test_missing_optional_package_reports_error_before_model_creation(monkeypatch):
    monkeypatch.setattr(foundation, 'is_package_available', lambda name: False)
    strategy = foundation.TabPFNClassificationStrategy('tabpfn', parameters('tabpfn'))
    with pytest.raises(ModuleNotFoundError, match='TabPFN is required'):
        strategy.fit(make_data())
