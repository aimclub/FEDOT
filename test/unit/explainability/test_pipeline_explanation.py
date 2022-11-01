from copy import deepcopy

import numpy as np
import pytest
from sklearn.datasets import load_iris, load_boston

from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.explainability.explainers import explain_pipeline
from fedot.explainability.surrogate_explainer import SurrogateExplainer, get_simple_pipeline

np.random.seed(1)


@pytest.fixture(scope='module', name='classification')
def classification_fixture() -> 'InputData, Pipeline':
    predictors, response = load_iris(return_X_y=True)
    np.random.seed(1)
    np.random.shuffle(predictors)
    np.random.shuffle(response)
    predictors = predictors[:100]
    response = response[:100]
    data = InputData(features=predictors, target=response, idx=np.arange(0, 100),
                     task=Task(TaskTypesEnum.classification),
                     data_type=DataTypesEnum.table)
    return data, get_simple_pipeline('rf')


@pytest.fixture(scope='module', name='regression')
def regression_fixture() -> 'InputData, Pipeline':
    predictors, response = load_boston(return_X_y=True)
    np.random.seed(1)
    np.random.shuffle(predictors)
    np.random.shuffle(response)
    predictors = predictors[:100]
    response = response[:100]
    data = InputData(features=predictors, target=response, idx=np.arange(0, 100),
                     task=Task(TaskTypesEnum.regression),
                     data_type=DataTypesEnum.table)
    return data, get_simple_pipeline('rfr')


@pytest.fixture(scope='module', name="task_type")
def task_type_fixture(request, classification, regression) -> 'InputData, Pipeline':
    task_type = request.param
    if task_type == TaskTypesEnum.classification:
        return classification
    elif task_type == TaskTypesEnum.regression:
        return regression
    else:
        raise ValueError(f'Unsupported task type: {task_type}')


def _successful_output(explainer: 'Explainer'):
    try:
        explainer.visualize()
        return True
    except Exception:
        return False


@pytest.mark.parametrize('method, task_type', [
    ('surrogate_dt', TaskTypesEnum.classification),
    ('surrogate_dt', TaskTypesEnum.regression),
], indirect=['task_type'])
def test_surrogate_explainer(method, task_type, request):
    data, pipeline = request.getfixturevalue('task_type')
    train, _ = train_test_data_setup(data)

    pipeline.fit(input_data=train)

    explainer = explain_pipeline(pipeline, data=train, method=method, visualization=False)

    assert isinstance(explainer, SurrogateExplainer)
    assert isinstance(explainer.surrogate, Pipeline)
    assert explainer.surrogate.is_fitted
    assert isinstance(explainer.score, float) and explainer.score > 0


@pytest.mark.parametrize('method, task_type', [
    ('surrogate_dt', TaskTypesEnum.classification),
], indirect=['task_type'])
def test_pipeline_explain(method, task_type, request):
    data, pipeline = request.getfixturevalue('task_type')
    train, _ = train_test_data_setup(data)

    pipeline.fit(input_data=train)
    old_pipeline = deepcopy(pipeline)

    explainer = explain_pipeline(pipeline, data=train, method=method, visualization=False)

    assert pipeline == old_pipeline
    assert _successful_output(explainer)
