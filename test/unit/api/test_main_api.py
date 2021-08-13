import pandas as pd

from fedot.api.main import Fedot, _define_data
from fedot.core.data.data import InputData
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from test.data_manager import import_metoocean_data, get_dataset, get_split_data, get_split_data_paths


composer_params = {'max_depth': 1,
                   'max_arity': 2,
                   'timeout': 0.0001,
                   'preset': 'ultra_light'}


def test_api_predict_correct(task_type: str = 'classification'):
    train_data, test_data, _ = get_dataset(task_type)
    model = Fedot(problem=task_type,
                  composer_params=composer_params)
    fedot_model = model.fit(features=train_data)
    prediction = model.predict(features=test_data)
    metric = model.get_metrics()

    assert isinstance(fedot_model, Pipeline)
    assert len(prediction) == len(test_data.target)
    assert metric['f1'] > 0


def test_api_forecast_correct(task_type: str = 'ts_forecasting'):
    # The forecast length must be equal to 5
    forecast_length = 5
    train_data, test_data, _ = get_dataset(task_type)
    model = Fedot(problem='ts_forecasting', composer_params=composer_params,
                  task_params=TsForecastingParams(forecast_length=forecast_length))

    model.fit(features=train_data)
    ts_forecast = model.predict(features=train_data)
    metric = model.get_metrics(target=test_data.target, metric_names='rmse')

    assert len(ts_forecast) == forecast_length
    assert metric['rmse'] >= 0


def test_api_forecast_numpy_input_with_static_model_correct(task_type: str = 'ts_forecasting'):
    forecast_length = 5
    train_data, test_data, _ = get_dataset(task_type)
    model = Fedot(problem='ts_forecasting',
                  task_params=TsForecastingParams(forecast_length=forecast_length))

    # Define pipeline for prediction
    node_lagged = PrimaryNode('lagged')
    pipeline = Pipeline(SecondaryNode('linear', nodes_from=[node_lagged]))

    model.fit(features=train_data.features,
              target=train_data.target,
              predefined_model=pipeline)
    ts_forecast = model.predict(features=train_data)
    metric = model.get_metrics(target=test_data.target, metric_names='rmse')

    assert len(ts_forecast) == forecast_length
    assert metric['rmse'] >= 0


def test_api_check_data_correct():
    task_type, x_train, x_test, y_train, y_test = get_split_data()
    path_to_train, path_to_test = get_split_data_paths()
    train_data, test_data, threshold = get_dataset(task_type)
    string_data_input = _define_data(ml_task=Task(TaskTypesEnum.regression),
                                     features=path_to_train)
    array_data_input = _define_data(ml_task=Task(TaskTypesEnum.regression),
                                    features=x_train)
    fedot_data_input = _define_data(ml_task=Task(TaskTypesEnum.regression),
                                    features=train_data)
    assert (not type(string_data_input) == InputData
            or type(array_data_input) == InputData
            or type(fedot_data_input) == InputData)


def test_baseline_with_api():
    train_data, test_data, threshold = get_dataset('classification')

    # task selection, initialisation of the framework
    baseline_model = Fedot(problem='classification')

    # fit model without optimisation - single XGBoost node is used
    baseline_model.fit(features=train_data, target='target', predefined_model='xgboost')

    # evaluate the prediction with test data
    prediction = baseline_model.predict_proba(features=test_data)

    assert len(prediction) == len(test_data.target)

    # evaluate quality metric for the test sample
    baseline_metrics = baseline_model.get_metrics(metric_names='f1')

    assert baseline_metrics['f1'] > 0


def test_pandas_input_for_api():
    train_data, test_data, threshold = get_dataset('classification')

    train_features = pd.DataFrame(train_data.features)
    train_target = pd.Series(train_data.target)

    test_features = pd.DataFrame(test_data.features)
    test_target = pd.Series(test_data.target)

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
    composer_params['composer_metric'] = ['f1', 'node_num']

    model = Fedot(problem='classification',
                  composer_params=composer_params)
    model.fit(features=train_data)
    prediction = model.predict(features=test_data)
    metric = model.get_metrics()

    assert len(prediction) == len(test_data.target)
    assert metric['f1'] > 0
    assert model.best_models is not None


def test_multivariate_ts():
    forecast_length = 1

    target_history, add_history, _ = import_metoocean_data()

    historical_data = {
        'ws': add_history,  # additional variable
        'ssh': target_history,  # target variable
    }

    fedot = Fedot(problem='ts_forecasting', composer_params=composer_params,
                  task_params=TsForecastingParams(forecast_length=forecast_length))
    fedot.fit(features=historical_data, target=target_history)
    forecast = fedot.forecast(historical_data, forecast_length=forecast_length)
    assert forecast is not None
