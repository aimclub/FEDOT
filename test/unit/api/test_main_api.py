import numpy as np
import pandas as pd
import pytest

from fedot import Fedot
from fedot.api.api_utils.api_data import ApiDataProcessor
from fedot.core.data.data import InputData
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.core.utils import fedot_project_root
from test.data.datasets import data_with_binary_features_and_categorical_target, get_dataset, \
    load_categorical_multidata, get_split_data_paths, get_split_data, get_multimodal_ts_data, load_categorical_unimodal

TESTS_MAIN_API_DEFAULT_PARAMS = {
    'timeout': 0.5,
    'preset': 'fast_train',
    'max_depth': 1,
    'max_arity': 2,
}


def test_pipeline_preprocessing_through_api_correctly():
    """ Preprocessing applying in two modules (places): API and pipeline.
    In API preprocessing there is an obligatory preparation for data.
    After API finish processing it returns pipeline which preprocessing module
    must be identical to preprocessing in api.
    """
    data = data_with_binary_features_and_categorical_target()

    fedot_model = Fedot(problem='classification')
    # Using API preprocessing and train pipeline to give forecasts
    pipeline = fedot_model.fit(data, predefined_model='dt')
    # Stand-alone pipeline with it's own preprocessing
    predicted = pipeline.predict(data, output_mode='labels')

    # check whether NaN-field was correctly predicted
    assert predicted.predict[3] == 'red-blue'


def test_data_from_csv_load_correctly():
    """
    Check if data obtained from csv files processed correctly for fit and
    predict stages when for predict stage there is no target column in csv file
    """
    task = Task(TaskTypesEnum.regression)
    project_root = fedot_project_root()
    path_train = 'test/data/empty_target_tables/train.csv'
    path_test = 'test/data/empty_target_tables/test.csv'
    full_path_train = project_root.joinpath(path_train)
    full_path_test = project_root.joinpath(path_test)

    data_loader = ApiDataProcessor(task)
    train_input = data_loader.define_data(features=full_path_train, target='class')
    test_input = data_loader.define_data(features=full_path_test, is_predict=True)

    assert train_input.target.shape == (14, 1)
    assert test_input.target is None


def test_unknown_param_raises_error():
    api_params = {'problem': 'classification', 'unknown': 2}
    try:
        _ = Fedot(**api_params)
    except KeyError as e:
        assert str(e) == '"Invalid key parameters {\'unknown\'}"'


def test_api_check_data_correct():
    """ Check that data preparing correctly using API methods
    Attention! During test execution the following warning arises
    "Columns number and types numbers do not match."

    This happens because the data are prepared for the predict stage
     without going through the fitting stage
    """
    task = Task(TaskTypesEnum.regression)

    # Get data
    task_type, x_train, x_test, y_train, y_test = get_split_data()
    path_to_train, path_to_test = get_split_data_paths()
    train_data, test_data, threshold = get_dataset(task_type)

    string_data_input = ApiDataProcessor(task).define_data(features=path_to_train, target='target')
    array_data_input = ApiDataProcessor(task).define_data(features=x_train, target=x_test)
    fedot_data_input = ApiDataProcessor(task).define_data(features=train_data)
    assert (not type(string_data_input) == InputData or
            type(array_data_input) == InputData or
            type(fedot_data_input) == InputData)


def test_api_check_multimodal_data_correct():
    """ Check that DataDefiner works correctly with multimodal data """
    task = Task(TaskTypesEnum.classification)

    # Get data
    array_data, target = load_categorical_multidata()

    array_data_input = ApiDataProcessor(task).define_data(features=array_data, target=target)

    assert isinstance(array_data_input, MultiModalData)
    for data_source in array_data_input:
        assert isinstance(array_data_input[data_source], InputData)


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


def test_forecast_with_multivariate_ts():
    forecast_length = 2

    historical_data, target = get_multimodal_ts_data()

    model = Fedot(problem='ts_forecasting', **TESTS_MAIN_API_DEFAULT_PARAMS,
                  task_params=TsForecastingParams(forecast_length=forecast_length))
    model.fit(features=historical_data, target=target, predefined_model='auto')
    forecast = model.forecast()
    assert len(forecast) == forecast_length
    forecast = model.forecast(horizon=forecast_length - 1)
    assert len(forecast) == forecast_length - 1
    with pytest.raises(ValueError):
        model.forecast(horizon=forecast_length + 1)


def test_ts_from_array():
    df = pd.read_csv(fedot_project_root().joinpath('test/data/simple_sea_level.csv'))
    train_array = np.array(df['Level'])

    task = Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(forecast_length=250))
    data = ApiDataProcessor(task).define_data(features=train_array, target='target')
    assert np.array_equal(data.target, data.features)



def test_default_forecast():
    forecast_length = 2
    train_data, test_data, _ = get_dataset('ts_forecasting')
    model = Fedot(problem='ts_forecasting', **TESTS_MAIN_API_DEFAULT_PARAMS,
                  task_params=TsForecastingParams(forecast_length=forecast_length))
    model.fit(train_data, predefined_model='auto')
    forecast = model.forecast()

    assert len(forecast) == forecast_length
    assert np.array_equal(model.test_data.idx, train_data.idx)

    metrics = model.get_metrics(metric_names=['rmse', 'mae', 'mape'],
                                validation_blocks=1, target=test_data.target)

    assert len(metrics) == 3
    assert all([m > 0 for m in metrics.values()])

    in_sample_forecast = model.predict(test_data, validation_blocks=1)
    metrics = model.get_metrics(metric_names=['mase', 'mae', 'mape'],
                                validation_blocks=1)
    assert in_sample_forecast is not None
    assert all([m > 0 for m in metrics.values()])


def test_categorical_preprocessing_unidata_predefined():
    train_data, test_data = load_categorical_unimodal()

    auto_model = Fedot(problem='classification', **TESTS_MAIN_API_DEFAULT_PARAMS)
    auto_model.fit(features=train_data, predefined_model='rf')
    prediction = auto_model.predict(features=test_data)
    prediction_proba = auto_model.predict_proba(features=test_data)

    assert np.issubdtype(prediction.dtype, np.number)
    assert np.isnan(prediction).sum() == 0
    assert np.issubdtype(prediction_proba.dtype, np.number)
    assert np.isnan(prediction_proba).sum() == 0
