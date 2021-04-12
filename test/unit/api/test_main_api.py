import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from fedot.api.main import Fedot, _define_data
from fedot.core.chains.chain import Chain
from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.data.data import InputData, train_test_data_setup
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.core.utils import project_root
from test.unit.models.test_split_train_test import get_synthetic_input_data
from test.unit.tasks.test_classification import get_iris_data
from test.unit.tasks.test_forecasting import get_synthetic_ts_data_period
from test.unit.tasks.test_regression import get_synthetic_regression_data

composer_params = {'max_depth': 1,
                   'max_arity': 2,
                   'learning_time': 0.0001,
                   'preset': 'ultra_light'}


def get_split_data_paths():
    file_path_train = 'test/data/simple_regression_train.csv'
    file_path_test = 'test/data/simple_regression_test.csv'
    full_path_train = os.path.join(str(project_root()), file_path_train)
    full_path_test = os.path.join(str(project_root()), file_path_test)

    return full_path_train, full_path_test


def get_split_data():
    task_type = 'regression'
    train_full, test = get_split_data_paths()
    train_file = pd.read_csv(train_full)
    x, y = train_file.loc[:, ~train_file.columns.isin(['target'])].values, train_file['target'].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=24)
    return task_type, x_train, x_test, y_train, y_test


def get_dataset(task_type: str):
    if task_type == 'regression':
        data = get_synthetic_regression_data()
        train_data, test_data = train_test_data_setup(data)
        threshold = np.std(test_data.target) * 0.05
    elif task_type == 'classification':
        data = get_iris_data()
        train_data, test_data = train_test_data_setup(data, shuffle_flag=True)
        threshold = 0.95
    elif task_type == 'clustering':
        data = get_synthetic_input_data(n_samples=10000)
        train_data, test_data = train_test_data_setup(data)
        threshold = 0.5
    elif task_type == 'ts_forecasting':
        train_data, test_data = get_synthetic_ts_data_period(forecast_length=12)
        threshold = np.str(test_data.target)
    else:
        raise ValueError('Incorrect type of machine learning task')
    return train_data, test_data, threshold


def test_api_predict_correct(task_type: str = 'classification'):
    train_data, test_data, _ = get_dataset(task_type)
    model = Fedot(problem=task_type,
                  composer_params=composer_params)
    fedot_model = model.fit(features=train_data)
    prediction = model.predict(features=test_data)
    metric = model.get_metrics()

    assert isinstance(fedot_model, Chain)
    assert len(prediction) == len(test_data.target)
    assert metric['f1'] > 0


def test_api_forecast_correct(task_type: str = 'ts_forecasting'):
    # The forecast length must be equal to 12
    forecast_length = 12
    train_data, test_data, _ = get_dataset(task_type)
    model = Fedot(problem='ts_forecasting', composer_params=composer_params,
                  task_params=TsForecastingParams(forecast_length=forecast_length))

    model.fit(features=train_data)
    ts_forecast = model.predict(features=train_data)
    metric = model.get_metrics(target=test_data.target, metric_names='rmse')

    assert len(ts_forecast) == forecast_length
    assert metric['rmse'] >= 0


def test_api_forecast_numpy_input_with_static_model_correct(task_type: str = 'ts_forecasting'):
    forecast_length = 10
    train_data, test_data, _ = get_dataset(task_type)
    model = Fedot(problem='ts_forecasting',
                  task_params=TsForecastingParams(forecast_length=forecast_length))

    # Define chain for prediction
    node_lagged = PrimaryNode('lagged')
    chain = Chain(SecondaryNode('linear', nodes_from=[node_lagged]))

    model.fit(features=train_data.features,
              target=train_data.target,
              predefined_model=chain)
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
