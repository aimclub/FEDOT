from dataclasses import replace
import pickle

import numpy as np
import pytest
import torch
from sklearn.datasets import make_classification, make_regression

from fedot.api.main import Fedot
from fedot.core.data.input_data.data import InputData
from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.data.tensor_data.tensor_data_creator import TensorDataCreator
from fedot.core.operations.strategies.models.sklearn import SklearnModelStrategy
from fedot.core.operations.implementations.models.estimator_adapters import ExplainableBoostingClassifierAdapter
from fedot.core.operations.strategies.models.sklearn import (
    SklearnClassificationStrategy,
    SklearnRegressionStrategy,
)
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.utilities.random import ImplementationRandomStateHandler


def _fast_tensor_params(operation):
    if operation.startswith('tabm'):
        return {'n_epochs': 1, 'patience': 1, 'tabm_k': 2, 'd_block': 64}
    if operation.startswith('ft_transformer'):
        return {'n_epochs': 1, 'patience': 1, 'd_block': 32, 'n_blocks': 1, 'attention_n_heads': 4}
    if operation.startswith('tab_resnet'):
        return {'n_epochs': 1, 'patience': 1, 'd_block': 32, 'n_blocks': 1}
    return {'n_epochs': 1, 'n_hidden_layers': 2, 'hidden_width': 64}


def _make_tensor_data(features, target, problem):
    task_type = TaskTypesEnum.classification if problem == 'classification' else TaskTypesEnum.regression
    return TensorDataCreator.create(
        features,
        backend_name='cpu',
        target=target,
        task=Task(task_type),
        data_type=DataTypesEnum.tabular,
        use_cache=False,
    )


@pytest.mark.unit
def test_sklearn_strategy_flattens_only_single_column_target():
    single_column_target = np.array([[0], [1], [0]])
    multi_output_target = np.array([[0, 1], [1, 0], [0, 0]])

    assert SklearnModelStrategy._sklearn_compatible_target(single_column_target).shape == (3,)
    assert SklearnModelStrategy._sklearn_compatible_target(multi_output_target).shape == (3, 2)


@pytest.mark.unit
@pytest.mark.parametrize('operation', ['extra_trees', 'hist_gb'])
def test_sklearn_experimental_classifiers_fit_predefined(operation):
    features, target = make_classification(
        n_samples=80, n_features=8, n_informative=4, random_state=42)
    tensor_data = _make_tensor_data(features, target, 'classification')

    model = Fedot(problem='classification', logging_level=50)
    pipeline = model.fit(tensor_data, predefined_model=operation)
    prediction = model.current_pipeline.predict(tensor_data, output_mode='full_probs').predict

    assert pipeline.root_node.operation.operation_type == operation
    assert isinstance(prediction, torch.Tensor)
    assert prediction.device == tensor_data.device
    assert prediction.shape[0] == len(target)
    assert torch.isfinite(prediction).all()


@pytest.mark.unit
@pytest.mark.parametrize('operation', ['hist_gbreg', 'mlpreg'])
def test_sklearn_experimental_regressors_fit_predefined(operation):
    features, target = make_regression(
        n_samples=80, n_features=8, random_state=42)
    tensor_data = _make_tensor_data(features, target, 'regression')

    model = Fedot(problem='regression', logging_level=50)
    pipeline = model.fit(tensor_data, predefined_model=operation)
    prediction = model.current_pipeline.predict(tensor_data).predict

    assert pipeline.root_node.operation.operation_type == operation
    assert isinstance(prediction, torch.Tensor)
    assert prediction.device == tensor_data.device
    assert prediction.shape == (len(target),)
    assert torch.isfinite(prediction).all()


@pytest.mark.unit
@pytest.mark.parametrize('operation', ['ebm', 'ebmreg'])
def test_ebm_fit_predefined(operation):
    pytest.importorskip('interpret')
    if operation == 'ebm':
        features, target = make_classification(
            n_samples=80, n_features=8, n_informative=4, random_state=42)
        tensor_data = _make_tensor_data(features, target, 'classification')
        model = Fedot(problem='classification', logging_level=50)
        node = PipelineNode(operation)
        node.parameters = {'max_rounds': 20, 'outer_bags': 2, 'interactions': 0, 'n_jobs': 1}
        pipeline = model.fit(tensor_data, predefined_model=Pipeline(node))
        prediction = model.current_pipeline.predict(tensor_data, output_mode='full_probs').predict
        assert prediction.shape[0] == len(target)
    else:
        features, target = make_regression(n_samples=80, n_features=8, random_state=42)
        tensor_data = _make_tensor_data(features, target, 'regression')
        model = Fedot(problem='regression', logging_level=50)
        node = PipelineNode(operation)
        node.parameters = {'max_rounds': 20, 'outer_bags': 2, 'interactions': 0, 'n_jobs': 1}
        pipeline = model.fit(tensor_data, predefined_model=Pipeline(node))
        prediction = model.current_pipeline.predict(tensor_data).predict
        assert prediction.shape == (len(target),)

    assert pipeline.root_node.operation.operation_type == operation
    assert isinstance(prediction, torch.Tensor)
    assert prediction.device == tensor_data.device
    assert torch.isfinite(prediction).all()


@pytest.mark.unit
@pytest.mark.parametrize(
    ('operation', 'strategy_class', 'task'),
    [
        ('extra_trees', SklearnClassificationStrategy, 'classification'),
        ('hist_gb', SklearnClassificationStrategy, 'classification'),
        ('ebm', SklearnClassificationStrategy, 'classification'),
        ('hist_gbreg', SklearnRegressionStrategy, 'regression'),
        ('ebmreg', SklearnRegressionStrategy, 'regression'),
        ('mlpreg', SklearnRegressionStrategy, 'regression'),
    ],
)
def test_experimental_sklearn_models_share_the_main_strategies(operation, strategy_class, task):
    strategy = strategy_class(operation)

    assert issubclass(strategy_class, SklearnModelStrategy)
    assert operation in strategy_class._operations_by_types
    assert strategy.implementation_info == str(strategy.operation_impl)


@pytest.mark.unit
@pytest.mark.parametrize('operation', ['hist_gbreg', 'ebmreg', 'mlpreg'])
def test_experimental_sklearn_regressors_preserve_multi_output(operation):
    if operation == 'ebmreg':
        pytest.importorskip('interpret')
        params = {'max_rounds': 2, 'outer_bags': 1, 'interactions': 0, 'n_jobs': 1}
    elif operation == 'mlpreg':
        params = {'hidden_layer_sizes': (8,), 'max_iter': 2}
    else:
        params = {'max_iter': 2}
    features, target = make_regression(n_samples=40, n_features=4, n_targets=2, random_state=42)
    data = _make_tensor_data(features, target, 'regression')
    strategy = SklearnRegressionStrategy(operation, OperationParameters(**params))

    fitted = strategy.fit(data)
    output = strategy.predict(fitted, data)

    assert isinstance(output, TensorData)
    assert output.predict.shape == data.target.shape
    assert output.predict.device == data.device
    assert torch.isfinite(output.predict).all()


@pytest.mark.unit
@pytest.mark.parametrize('operation', ['extra_trees', 'hist_gb', 'ebm'])
def test_experimental_sklearn_classifiers_preserve_output_modes_and_metadata(operation):
    if operation == 'ebm':
        pytest.importorskip('interpret')
        params = {'max_rounds': 2, 'outer_bags': 1, 'interactions': 0, 'n_jobs': 1}
    elif operation == 'hist_gb':
        params = {'max_iter': 2}
    else:
        params = {'n_estimators': 2}
    features, target = make_classification(n_samples=40, n_features=4, n_redundant=0, random_state=42)
    data = _make_tensor_data(features, target, 'classification')
    data.idx = [f'row_{i}' for i in range(len(target))]
    strategy = SklearnClassificationStrategy(operation, OperationParameters(**params))
    fitted = strategy.fit(data)

    for mode in ['labels', 'probs', 'full_probs', 'default']:
        strategy.output_mode = mode
        output = strategy.predict(fitted, data)
        prepared_features = data.features.numpy()
        expected = fitted.predict(prepared_features) if mode == 'labels' else fitted.predict_proba(prepared_features)
        if mode in ['probs', 'default']:
            expected = expected[:, 1]

        torch.testing.assert_close(output.predict, torch.as_tensor(expected))
        assert output.idx == data.idx
        assert output.preparation_state is data.preparation_state
        assert output.task == data.task
        assert output.features is data.features
        assert data.predict is None


@pytest.mark.unit
@pytest.mark.parametrize('class_labels', [[2.5, 10.5], [2.5, 10.5, 40.5]])
def test_ebm_preserves_numeric_class_labels_when_predicting_without_target(class_labels):
    pytest.importorskip('interpret')
    features, target = make_classification(
        n_samples=40, n_features=4, n_redundant=0, n_classes=len(class_labels),
        n_clusters_per_class=1, random_state=42)
    classes = torch.tensor(class_labels)
    target = np.asarray(class_labels)[target]
    data = _make_tensor_data(features, target, 'classification')
    strategy = SklearnClassificationStrategy(
        'ebm', OperationParameters(max_rounds=2, outer_bags=1, interactions=0, n_jobs=1))
    fitted = strategy.fit(data)
    fitted = pickle.loads(pickle.dumps(fitted))
    test_data = replace(data, target=None)

    strategy.output_mode = 'full_probs'
    probabilities = strategy.predict(fitted, test_data)
    strategy.output_mode = 'labels'
    labels = strategy.predict(fitted, test_data)

    assert labels.target is None
    torch.testing.assert_close(labels.predict, classes[probabilities.predict.argmax(dim=1)])


@pytest.mark.unit
def test_ebm_adapter_forwards_and_restores_estimator_random_state():
    pytest.importorskip('interpret')
    adapter = ExplainableBoostingClassifierAdapter(random_state=123)

    with ImplementationRandomStateHandler(seed=17, implementation=adapter):
        assert adapter.implementation.random_state == 17

    assert adapter.random_state == 123
    assert adapter.implementation.random_state == 123


@pytest.mark.unit
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is unavailable')
@pytest.mark.parametrize('operation', ['extra_trees', 'hist_gbreg'])
def test_experimental_sklearn_predictions_return_to_tensor_device(operation):
    problem = 'classification' if operation == 'extra_trees' else 'regression'
    features, target = make_classification(n_samples=40, n_features=4, n_redundant=0, random_state=42)
    data = _make_tensor_data(features, target, problem).to('cuda')
    strategy_class = (SklearnClassificationStrategy if problem == 'classification'
                      else SklearnRegressionStrategy)
    strategy = strategy_class(operation)
    strategy.output_mode = 'full_probs'

    output = strategy.predict(strategy.fit(data), data)

    assert isinstance(output, TensorData)
    assert output.predict.device == data.device
    assert torch.isfinite(output.predict).all()


@pytest.mark.unit
@pytest.mark.parametrize('n_classes', [2, 3])
@pytest.mark.parametrize('operation', ['tabm', 'ft_transformer', 'tab_resnet', 'realmlp'])
def test_tensor_tabular_classifiers_fit_tensordata(operation, n_classes):
    pytest.importorskip('torch')
    if operation == 'tabm':
        pytest.importorskip('tabm')
    if operation in {'ft_transformer', 'tab_resnet'}:
        pytest.importorskip('rtdl_revisiting_models')
    if operation == 'realmlp':
        pytest.importorskip('pytabkit')

    features, target = make_classification(
        n_samples=50, n_features=6, n_informative=3, n_classes=n_classes, random_state=42)
    tensor_data = TensorDataCreator.create(
        features,
        backend_name='cpu',
        target=target,
        task=Task(TaskTypesEnum.classification),
        data_type=DataTypesEnum.tabular,
        use_cache=False,
    )
    node = PipelineNode(operation)
    node.parameters = _fast_tensor_params(operation)
    pipeline = Pipeline(node)

    model = Fedot(problem='classification', logging_level=50)
    fitted = model.fit(tensor_data, predefined_model=pipeline)
    prediction = model.current_pipeline.predict(tensor_data, output_mode='full_probs').predict

    assert fitted.root_node.operation.operation_type == operation
    assert np.asarray(prediction).shape[0] == len(target)
    assert isinstance(prediction, torch.Tensor)
    assert torch.isfinite(prediction).all()
    assert prediction.shape == (len(target), n_classes)

    implementation = fitted.root_node.fitted_operation
    predict_data = replace(tensor_data, target=None)
    probabilities = implementation.predict_proba(predict_data)
    labels = implementation.predict(predict_data)
    assert probabilities.predict.shape == (len(target), n_classes)
    classes = torch.as_tensor(implementation.classes_)
    torch.testing.assert_close(labels.predict, classes[probabilities.predict.argmax(dim=1)])
    for mode in ['labels', 'probs', 'full_probs', 'default', False]:
        output = fitted.predict(predict_data, output_mode=mode)
        expected = labels.predict if mode == 'labels' else probabilities.predict
        if n_classes == 2 and mode not in ['labels', 'full_probs']:
            expected = expected[:, 1]
        torch.testing.assert_close(output.predict, expected)
        assert predict_data.predict is None


@pytest.mark.unit
@pytest.mark.parametrize('operation', ['tabmreg', 'ft_transformerreg', 'tab_resnetreg', 'realmlpreg'])
def test_tensor_tabular_regressors_fit_tensordata(operation):
    pytest.importorskip('torch')
    if operation == 'tabmreg':
        pytest.importorskip('tabm')
    if operation in {'ft_transformerreg', 'tab_resnetreg'}:
        pytest.importorskip('rtdl_revisiting_models')
    if operation == 'realmlpreg':
        pytest.importorskip('pytabkit')

    features, target = make_regression(
        n_samples=50, n_features=6, random_state=42)
    tensor_data = TensorDataCreator.create(
        features,
        backend_name='cpu',
        target=target,
        task=Task(TaskTypesEnum.regression),
        data_type=DataTypesEnum.tabular,
        use_cache=False,
    )
    node = PipelineNode(operation)
    node.parameters = _fast_tensor_params(operation)
    pipeline = Pipeline(node)

    model = Fedot(problem='regression', logging_level=50)
    fitted = model.fit(tensor_data, predefined_model=pipeline)
    prediction = model.current_pipeline.predict(tensor_data).predict

    assert fitted.root_node.operation.operation_type == operation
    assert np.asarray(prediction).shape == (len(target),)
    assert isinstance(prediction, torch.Tensor)
    assert torch.isfinite(prediction).all()
