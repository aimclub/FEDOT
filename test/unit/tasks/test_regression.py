import numpy as np
from sklearn.metrics import mean_squared_error as mse

from fedot.api.main import Fedot
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from data.data_manager import multi_target_data_setup, get_synthetic_regression_data
from data.pipeline_manager import generate_pipeline
from data.utils import get_simple_composer_params


def get_rmse_value(pipeline: Pipeline, train_data: InputData, test_data: InputData) -> (float, float):
    train_pred = pipeline.predict(input_data=train_data)
    test_pred = pipeline.predict(input_data=test_data)
    rmse_value_test = mse(y_true=test_data.target, y_pred=test_pred.predict, squared=False)
    rmse_value_train = mse(y_true=train_data.target, y_pred=train_pred.predict, squared=False)

    return rmse_value_train, rmse_value_test


def test_regression_pipeline_fit_correct():
    data = get_synthetic_regression_data()

    pipeline = generate_pipeline()
    train_data, test_data = train_test_data_setup(data)

    pipeline.fit(input_data=train_data)
    _, rmse_on_test = get_rmse_value(pipeline, train_data, test_data)

    rmse_threshold = np.std(data.target) * 0.05
    assert rmse_on_test < rmse_threshold


def test_regression_pipeline_with_data_operation_fit_correct():
    data = get_synthetic_regression_data()
    train_data, test_data = train_test_data_setup(data)

    #           linear
    #       /           \
    #     ridge          |
    #       |            |
    # ransac_lin_reg   lasso
    #        \         /
    #          scaling
    node_scaling = PrimaryNode('scaling')
    node_ransac = SecondaryNode('ransac_lin_reg', nodes_from=[node_scaling])
    node_lasso = SecondaryNode('lasso', nodes_from=[node_scaling])
    node_ridge = SecondaryNode('ridge', nodes_from=[node_ransac])
    node_root = SecondaryNode('linear', nodes_from=[node_lasso, node_ridge])
    pipeline = Pipeline(node_root)

    pipeline.fit(train_data)
    results = pipeline.predict(test_data)

    assert results.predict.shape == test_data.target.shape


def test_multi_target_regression_composing_correct():
    # Load simple dataset for multi-target
    train, test = multi_target_data_setup()

    problem = 'regression'
    simple_composer_params = get_simple_composer_params()

    automl_model = Fedot(problem=problem, composer_params=simple_composer_params)
    automl_model.fit(features=train)
    predicted_array = automl_model.predict(features=test)
    assert predicted_array is not None
