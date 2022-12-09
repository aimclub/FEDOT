from copy import deepcopy

import numpy as np
import pytest
from sklearn.datasets import load_iris, load_diabetes

from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.explainability.explainers import explain_pipeline
from fedot.explainability.surrogate_explainer import SurrogateExplainer, get_simple_pipeline

np.random.seed(1)


@pytest.fixture(scope='module')
def data_for_task_type(request) -> (InputData, Pipeline):
    task_type = request.param
    if task_type == TaskTypesEnum.classification:
        load_func = load_iris
        pipeline = get_simple_pipeline('rf')
    elif task_type == TaskTypesEnum.regression:
        load_func = load_diabetes
        pipeline = get_simple_pipeline('rfr')
    else:
        raise ValueError(f'Unsupported task type: {task_type}')
    predictors, response = load_func(return_X_y=True)
    np.random.seed(1)
    np.random.shuffle(predictors)
    np.random.shuffle(response)
    predictors = predictors[:100]
    response = response[:100]
    data = InputData(features=predictors, target=response, idx=np.arange(0, 100),
                     task=Task(task_type),
                     data_type=DataTypesEnum.table)
    return data, pipeline


def _successful_output(explainer):
    try:
        explainer.visualize()
        return True
    except Exception as e:
        raise e


@pytest.mark.parametrize(
    'data_for_task_type, method',
    [
        (TaskTypesEnum.classification, 'surrogate_dt'),
        (TaskTypesEnum.regression, 'surrogate_dt'),
    ],
    indirect=['data_for_task_type'])
def test_surrogate_explainer(data_for_task_type, method):
    data, pipeline = data_for_task_type
    train, _ = train_test_data_setup(data)
    pipeline.fit(input_data=train)

    explainer = explain_pipeline(pipeline, data=train, method=method, visualization=False)

    assert isinstance(explainer, SurrogateExplainer)
    assert isinstance(explainer.surrogate, Pipeline)
    assert explainer.surrogate.is_fitted
    assert isinstance(explainer.score, float) and explainer.score > 0


@pytest.mark.parametrize(
    'data_for_task_type, method',
    [
        (TaskTypesEnum.classification, 'surrogate_dt'),
        (TaskTypesEnum.regression, 'surrogate_dt'),
    ],
    indirect=['data_for_task_type'])
def test_pipeline_explain(data_for_task_type, method):
    data, pipeline = data_for_task_type
    train, _ = train_test_data_setup(data)

    pipeline.fit(input_data=train)
    old_pipeline = deepcopy(pipeline)

    explainer = explain_pipeline(pipeline, data=train, method=method, visualization=False)

    assert pipeline == old_pipeline
    assert _successful_output(explainer)
