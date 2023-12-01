import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error as mse

from fedot import Fedot
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from test.unit.common_tests import is_predict_ignores_target
from test.unit.composer.test_metrics import data_setup  # noqa


def check_predict_correct(pipeline, input_data):
    return is_predict_ignores_target(
        predict_func=pipeline.predict,
        data_arg_name='input_data',
        input_data=input_data,
    )


def get_simple_composer_params() -> dict:
    params = {'max_depth': 2,
              'max_arity': 3,
              'pop_size': 2,
              'num_of_generations': 2,
              'with_tuning': True,
              'preset': 'fast_train',
              'show_progress': False}
    return params


def generate_pipeline() -> Pipeline:
    node_scaling = PipelineNode('scaling')
    node_lasso = PipelineNode('lasso', nodes_from=[node_scaling])
    node_ridge = PipelineNode('ridge', nodes_from=[node_scaling])
    node_root = PipelineNode('linear', nodes_from=[node_lasso, node_ridge])
    pipeline = Pipeline(node_root)
    return pipeline


def get_synthetic_regression_data(n_samples=1000, n_features=10, random_state=None) -> InputData:
    synthetic_data = make_regression(n_samples=n_samples, n_features=n_features, random_state=random_state)
    input_data = InputData(idx=np.arange(0, len(synthetic_data[1])),
                           features=synthetic_data[0],
                           target=synthetic_data[1].reshape((-1, 1)),
                           task=Task(TaskTypesEnum.regression),
                           data_type=DataTypesEnum.table)
    return input_data


def get_rmse_value(pipeline: Pipeline, train_data: InputData, test_data: InputData) -> (float, float):
    train_pred = pipeline.predict(input_data=train_data)
    test_pred = pipeline.predict(input_data=test_data)
    rmse_value_test = mse(y_true=test_data.target, y_pred=test_pred.predict, squared=False)
    rmse_value_train = mse(y_true=train_data.target, y_pred=train_pred.predict, squared=False)

    return rmse_value_train, rmse_value_test


def test_regression_pipeline_fit_predict_correct():
    data = get_synthetic_regression_data()

    pipeline = generate_pipeline()
    train_data, test_data = train_test_data_setup(data)

    pipeline.fit(input_data=train_data)
    _, rmse_on_test = get_rmse_value(pipeline, train_data, test_data)

    rmse_threshold = np.std(data.target) * 0.05
    assert rmse_on_test < rmse_threshold
    assert check_predict_correct(pipeline, train_data)


def test_regression_pipeline_with_data_operation_fit_predict_correct():
    data = get_synthetic_regression_data()
    train_data, test_data = train_test_data_setup(data)

    #           linear
    #       /           \
    #     ridge          |
    #       |            |
    # ransac_lin_reg   lasso
    #        \         /
    #          scaling
    node_scaling = PipelineNode('scaling')
    node_ransac = PipelineNode('ransac_lin_reg', nodes_from=[node_scaling])
    node_lasso = PipelineNode('lasso', nodes_from=[node_scaling])
    node_ridge = PipelineNode('ridge', nodes_from=[node_ransac])
    node_root = PipelineNode('linear', nodes_from=[node_lasso, node_ridge])
    pipeline = Pipeline(node_root)

    pipeline.fit(train_data)
    results = pipeline.predict(test_data)

    assert results.predict.shape == test_data.target.shape
    assert check_predict_correct(pipeline, train_data)


@pytest.mark.parametrize('data_setup', ['multitarget'], indirect=True)
def test_multi_target_regression_composing_correct(data_setup):
    # Load simple dataset for multi-target
    train, test, _ = data_setup

    problem = 'regression'
    timeout = 0.1
    simple_composer_params = get_simple_composer_params()

    automl_model = Fedot(problem=problem, timeout=timeout, **simple_composer_params)
    automl_model.fit(train)
    predicted_array = automl_model.predict(test)
    assert predicted_array is not None
