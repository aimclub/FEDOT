import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression

from fedot.api.main import Fedot
from fedot.core.data.tensor_data.tensor_data_creator import TensorDataCreator
from fedot.core.operations.evaluation.evaluation_interfaces import SkLearnEvaluationStrategy
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum


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

    assert SkLearnEvaluationStrategy._sklearn_compatible_target(single_column_target).shape == (3,)
    assert SkLearnEvaluationStrategy._sklearn_compatible_target(multi_output_target).shape == (3, 2)


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
    assert np.asarray(prediction).shape[0] == len(target)
    assert np.all(np.isfinite(prediction))


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
    assert np.asarray(prediction).shape == (len(target),)
    assert np.all(np.isfinite(prediction))


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
        assert np.asarray(prediction).shape[0] == len(target)
    else:
        features, target = make_regression(n_samples=80, n_features=8, random_state=42)
        tensor_data = _make_tensor_data(features, target, 'regression')
        model = Fedot(problem='regression', logging_level=50)
        node = PipelineNode(operation)
        node.parameters = {'max_rounds': 20, 'outer_bags': 2, 'interactions': 0, 'n_jobs': 1}
        pipeline = model.fit(tensor_data, predefined_model=Pipeline(node))
        prediction = model.current_pipeline.predict(tensor_data).predict
        assert np.asarray(prediction).shape == (len(target),)

    assert pipeline.root_node.operation.operation_type == operation
    assert np.all(np.isfinite(prediction))


@pytest.mark.unit
@pytest.mark.parametrize('operation', ['tabm', 'ft_transformer', 'tab_resnet', 'realmlp'])
def test_tensor_tabular_classifiers_fit_tensordata(operation):
    pytest.importorskip('torch')
    if operation == 'tabm':
        pytest.importorskip('tabm')
    if operation in {'ft_transformer', 'tab_resnet'}:
        pytest.importorskip('rtdl_revisiting_models')
    if operation == 'realmlp':
        pytest.importorskip('pytabkit')

    features, target = make_classification(
        n_samples=50, n_features=6, n_informative=3, random_state=42)
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
    assert np.all(np.isfinite(prediction))


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
    assert np.all(np.isfinite(prediction))
