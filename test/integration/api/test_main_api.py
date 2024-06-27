import os
import shutil
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from golem.core.dag.graph_utils import graph_structure
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder

from examples.simple.time_series_forecasting.ts_pipelines import ts_complex_ridge_smoothing_pipeline
from fedot import Fedot
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.tasks import TsForecastingParams
from test.data.datasets import get_dataset, get_multimodal_ts_data, load_categorical_unimodal, \
    load_categorical_multidata
from test.unit.common_tests import is_predict_ignores_target
from test.unit.tasks.test_multi_ts_forecast import get_multi_ts_data

TESTS_MAIN_API_DEFAULT_PARAMS = {
    'timeout': 0.5,
    'preset': 'fast_train',
    'max_depth': 1,
    'max_arity': 2,
}


@pytest.mark.parametrize('task_type, metric_name', [
    ('classification', 'f1'),
    ('classification', 'neg_log_loss'),
    ('regression', 'rmse')
])
def test_api_predict_correct(task_type, metric_name):
    train_data, test_data, _ = get_dataset(task_type)
    changed_api_params = {
        **TESTS_MAIN_API_DEFAULT_PARAMS,
        'timeout': 1,
        'preset': 'fast_train'
    }
    model = Fedot(problem=task_type, metric=metric_name, **changed_api_params)
    fedot_model = model.fit(features=train_data)
    prediction = model.predict(features=test_data)
    metric = model.get_metrics(metric_names=metric_name, rounding_order=5)
    assert isinstance(fedot_model, Pipeline)
    assert len(prediction) == len(test_data.target)
    assert all(value >= 0 for value in metric.values())
    # composing and tuning was applied
    assert model.history is not None
    assert model.history.tuning_result is not None
    assert is_predict_ignores_target(model.predict, model.train_data, 'features')


@pytest.mark.parametrize('task_type, metric_name, pred_model', [
    ('classification', 'f1', 'dt'),
    ('regression', 'rmse', 'dtreg'),
    ('ts_forecasting', 'rmse', 'glm')
])
def test_api_tune_correct(task_type, metric_name, pred_model):
    tuning_timeout = 0.2

    if task_type == 'ts_forecasting':
        forecast_length = 1
        train_data, test_data, _ = get_dataset(task_type, validation_blocks=1,
                                               forecast_length=forecast_length)
        model = Fedot(
            problem=task_type,
            task_params=TsForecastingParams(forecast_length=forecast_length))
    else:
        train_data, test_data, _ = get_dataset(task_type, n_samples=100, n_features=10, iris_dataset=False)
        model = Fedot(problem=task_type)

    base_pipeline = deepcopy(model.fit(features=train_data, predefined_model=pred_model))
    pred_before = model.predict(features=test_data)

    tuned_pipeline = deepcopy(model.tune(timeout=tuning_timeout, n_jobs=1))
    pred_after = model.predict(features=test_data)

    assert isinstance(tuned_pipeline, Pipeline)
    assert graph_structure(base_pipeline) != graph_structure(tuned_pipeline)
    assert model.api_composer.was_tuned
    assert not model.api_composer.was_optimised
    assert len(test_data.target) == len(pred_before) == len(pred_after)


def test_api_simple_ts_predict_correct(task_type: str = 'ts_forecasting'):
    # The forecast length must be equal to 5
    forecast_length = 5
    train_data, test_data, _ = get_dataset(task_type, validation_blocks=1)
    model = Fedot(problem='ts_forecasting', **TESTS_MAIN_API_DEFAULT_PARAMS,
                  task_params=TsForecastingParams(forecast_length=forecast_length))

    model.fit(features=train_data, predefined_model='auto')
    ts_forecast = model.predict(features=test_data)
    _ = model.get_metrics(target=test_data.target, metric_names='rmse')

    assert len(ts_forecast) == forecast_length


@pytest.mark.parametrize('validation_blocks', [None, 2, 3])
def test_api_in_sample_ts_predict_correct(validation_blocks, task_type: str = 'ts_forecasting'):
    # The forecast length must be equal to 5
    forecast_length = 5
    train_data, test_data, _ = get_dataset(task_type, validation_blocks=validation_blocks)
    model = Fedot(problem='ts_forecasting', **TESTS_MAIN_API_DEFAULT_PARAMS,
                  task_params=TsForecastingParams(forecast_length=forecast_length))

    model.fit(features=train_data, predefined_model='auto')
    ts_forecast = model.predict(features=test_data, validation_blocks=validation_blocks)
    _ = model.get_metrics(target=test_data.target, metric_names='rmse', validation_blocks=validation_blocks)

    assert len(ts_forecast) == forecast_length * validation_blocks if validation_blocks else forecast_length * 2


@pytest.mark.parametrize('validation_blocks', [None, 2, 3])
def test_api_in_sample_multi_ts_predict_correct(validation_blocks, task_type: str = 'ts_forecasting'):
    forecast_length = 2
    train_data, test_data = get_multi_ts_data(forecast_length=forecast_length, validation_blocks=validation_blocks)
    model = Fedot(problem='ts_forecasting', **TESTS_MAIN_API_DEFAULT_PARAMS,
                  task_params=TsForecastingParams(forecast_length=forecast_length),
                  available_operations=['lagged', 'smoothing', 'diff_filter', 'gaussian_filter',
                                        'ridge', 'lasso', 'linear', 'cut'])

    model.fit(features=train_data, predefined_model=ts_complex_ridge_smoothing_pipeline())
    ts_forecast = model.predict(features=test_data, validation_blocks=validation_blocks)
    _ = model.get_metrics(target=test_data.target, metric_names='rmse', validation_blocks=validation_blocks)

    assert len(ts_forecast) == forecast_length * validation_blocks if validation_blocks else forecast_length * 254


@pytest.mark.parametrize('validation_blocks', [None, 2, 3])
def test_api_in_sample_multimodal_ts_predict_correct(validation_blocks):
    forecast_length = 5
    historical_data, target = get_multimodal_ts_data()

    model = Fedot(problem='ts_forecasting', **TESTS_MAIN_API_DEFAULT_PARAMS,
                  task_params=TsForecastingParams(forecast_length=forecast_length))
    model.fit(features=historical_data, target=target, predefined_model='auto')
    ts_forecast = model.predict(historical_data, validation_blocks=validation_blocks)
    assert len(ts_forecast) == forecast_length * validation_blocks if validation_blocks else forecast_length


def test_api_forecast_numpy_input_with_static_model_correct(task_type: str = 'ts_forecasting'):
    forecast_length = 2
    train_data, test_data, _ = get_dataset(task_type, validation_blocks=1)
    model = Fedot(problem='ts_forecasting',
                  task_params=TsForecastingParams(forecast_length=forecast_length))

    # Define pipeline for prediction
    node_lagged = PipelineNode('lagged')
    pipeline = Pipeline(PipelineNode('linear', nodes_from=[node_lagged]))

    model.fit(features=train_data.features,
              target=train_data.target,
              predefined_model=pipeline)
    ts_forecast = model.predict(features=test_data, in_sample=False)
    metric = model.get_metrics(target=test_data.target, metric_names='rmse')

    assert len(ts_forecast) == forecast_length
    assert all(value > 0 for value in metric.values())


def test_pandas_input_for_api():
    train_data, test_data, threshold = get_dataset('classification')

    train_features = pd.DataFrame(train_data.features)
    train_target = pd.Series(train_data.target.reshape(-1))

    test_features = pd.DataFrame(test_data.features)
    test_target = pd.Series(test_data.target.reshape(-1))

    # task selection, initialisation of the framework
    baseline_model = Fedot(problem='classification')

    # fit model without optimisation - single XGBoost node is used
    baseline_model.fit(features=train_features, target=train_target, predefined_model='xgboost')

    # evaluate the prediction with test data
    prediction = baseline_model.predict(features=test_features)

    assert len(prediction) == len(test_target)

    # evaluate quality metric for the test sample
    baseline_metrics = baseline_model.get_metrics(metric_names='f1', target=test_target)

    assert baseline_metrics['f1'] > 0


def test_multiobj_for_api():
    train_data, test_data, _ = get_dataset('classification')

    params = {
        **TESTS_MAIN_API_DEFAULT_PARAMS,
        'metric': ['f1', 'node_number']
    }

    model = Fedot(problem='classification', **params)
    model.fit(features=train_data)
    prediction = model.predict(features=test_data)
    metric = model.get_metrics()

    assert len(prediction) == len(test_data.target)
    assert metric['f1'] > 0
    assert model.best_models is not None and len(model.best_models) > 0


def test_categorical_preprocessing_unidata():
    train_data, test_data = load_categorical_unimodal()

    auto_model = Fedot(problem='classification', **TESTS_MAIN_API_DEFAULT_PARAMS)
    auto_model.fit(features=train_data)
    prediction = auto_model.predict(features=test_data)
    prediction_proba = auto_model.predict_proba(features=test_data)

    assert np.issubdtype(prediction.dtype, np.number)
    assert np.isnan(prediction).sum() == 0
    assert np.issubdtype(prediction_proba.dtype, np.number)
    assert np.isnan(prediction_proba).sum() == 0


def test_categorical_preprocessing_unidata_predefined_linear():
    train_data, test_data = load_categorical_unimodal()

    pipeline = Pipeline(nodes=PipelineNode('logit'))
    pipeline.fit(train_data)
    prediction = pipeline.predict(test_data)

    for i in range(prediction.features.shape[1]):
        assert all(list(map(lambda x: isinstance(x, (int, float)), prediction.features[:, i])))


def test_fill_nan_without_categorical():
    train_data, test_data = load_categorical_unimodal()
    train_data.features = np.hstack((train_data.features[:, :2], train_data.features[:, 4:]))
    test_data.features = np.hstack((test_data.features[:, :2], test_data.features[:, 4:]))

    pipeline = Pipeline(nodes=PipelineNode('logit'))
    pipeline.fit(train_data)
    prediction = pipeline.predict(test_data)
    prediction_train = pipeline.predict(train_data)

    assert pd.isna(prediction.features).sum() == 0
    assert pd.isna(prediction_train.features).sum() == 0


def test_dict_multimodal_input_for_api():
    data, target = load_categorical_multidata()

    model = Fedot(problem='classification', metric=['f1'], **TESTS_MAIN_API_DEFAULT_PARAMS)

    model.fit(features=data, target=target)

    prediction = model.predict(features=data)

    assert len(prediction) == len(target)

    metrics = model.get_metrics(metric_names='f1')

    assert metrics['f1'] > 0


def test_unshuffled_data():
    target_column = 'species'
    df_el, y = load_iris(return_X_y=True, as_frame=True)
    df_el[target_column] = LabelEncoder().fit_transform(y)

    features, target = df_el.drop(target_column, axis=1).values, df_el[target_column].values

    problem = 'classification'
    params = {
        **TESTS_MAIN_API_DEFAULT_PARAMS,
        'metric': 'f1'}

    auto_model = Fedot(problem=problem, seed=42, **params)
    pipeline = auto_model.fit(features=features, target=target)
    assert pipeline is not None


def test_custom_history_dir_define_correct():
    train_data, test_data, _ = get_dataset('ts_forecasting')

    custom_path = os.path.join(os.path.abspath(os.getcwd()), 'history_dir')

    params = {
        **TESTS_MAIN_API_DEFAULT_PARAMS,
        'history_dir': custom_path,
        'timeout': None,
        'num_of_generations': 1,
        'pop_size': 3}

    model = Fedot(problem='ts_forecasting', **params,
                  task_params=TsForecastingParams(forecast_length=5))

    model.fit(features=train_data)

    assert len(os.listdir(custom_path)) != 0
    shutil.rmtree(custom_path)


@pytest.mark.parametrize('horizon', [1, 2, 3, 4])
def test_forecast_with_different_horizons(horizon):
    forecast_length = 2
    train_data, test_data, _ = get_dataset('ts_forecasting')
    model = Fedot(problem='ts_forecasting', **TESTS_MAIN_API_DEFAULT_PARAMS,
                  task_params=TsForecastingParams(forecast_length=forecast_length))
    model.fit(train_data, predefined_model='auto')
    forecast = model.forecast(pre_history=test_data, horizon=horizon)
    assert len(forecast) == horizon
    assert np.array_equal(model.test_data.idx, test_data.idx)


def test_forecast_with_unfitted_model():
    forecast_length = 2
    model = Fedot(problem='ts_forecasting', **TESTS_MAIN_API_DEFAULT_PARAMS,
                  task_params=TsForecastingParams(forecast_length=forecast_length))
    with pytest.raises(ValueError):
        model.forecast()


def test_forecast_with_not_ts_problem():
    model = Fedot(problem='classification', **TESTS_MAIN_API_DEFAULT_PARAMS)
    train_data, test_data, _ = get_dataset('classification')
    model.fit(train_data, predefined_model='auto')
    with pytest.raises(ValueError):
        model.forecast(pre_history=test_data)
