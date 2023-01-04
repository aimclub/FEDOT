import os
import shutil
from typing import Optional

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from cases.metocean_forecasting_problem import prepare_input_data
from examples.simple.time_series_forecasting.ts_pipelines import ts_complex_ridge_smoothing_pipeline
from fedot.api.api_utils.api_composer import _divide_parameters
from fedot.api.api_utils.api_data import ApiDataProcessor
from fedot.api.main import Fedot
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.data.supplementary_data import SupplementaryData
from fedot.core.optimisers.gp_comp.operators.inheritance import GeneticSchemeTypesEnum
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.quality_metrics_repository import RegressionMetricsEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.core.utils import fedot_project_root
from test.unit.common_tests import is_predict_ignores_target
from test.unit.models.test_split_train_test import get_synthetic_input_data
from test.unit.tasks.test_classification import get_iris_data
from test.unit.tasks.test_forecasting import get_ts_data
from test.unit.tasks.test_multi_ts_forecast import get_multi_ts_data
from test.unit.tasks.test_regression import get_synthetic_regression_data

default_params = {
    'timeout': 0.1,
    'preset': 'fast_train',
    'max_depth': 1,
    'max_arity': 2,
}


def get_split_data_paths():
    file_path_train = 'test/data/simple_regression_train.csv'
    file_path_test = 'test/data/simple_regression_test.csv'
    full_path_train = os.path.join(str(fedot_project_root()), file_path_train)
    full_path_test = os.path.join(str(fedot_project_root()), file_path_test)

    return full_path_train, full_path_test


def get_split_data():
    task_type = 'regression'
    train_full, test = get_split_data_paths()
    train_file = pd.read_csv(train_full)
    x, y = train_file.loc[:, ~train_file.columns.isin(['target'])].values, train_file['target'].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=24)
    return task_type, x_train, x_test, y_train, y_test


def get_cholesterol_dataset():
    data_path = f'{fedot_project_root()}/cases/data/cholesterol/cholesterol.csv'
    data = InputData.from_csv(data_path, task=Task(TaskTypesEnum.regression))
    train, test = train_test_data_setup(data)
    return train, test


def get_dataset(task_type: str, validation_blocks: Optional[int] = None):
    if task_type == 'regression':
        data = get_synthetic_regression_data(n_samples=50, n_features=5)
        train_data, test_data = train_test_data_setup(data)
        threshold = np.std(test_data.target) * 0.05
    elif task_type == 'classification':
        data = get_iris_data()
        train_data, test_data = train_test_data_setup(data, shuffle_flag=True)
        threshold = 0.95
    elif task_type == 'clustering':
        data = get_synthetic_input_data(n_samples=100)
        train_data, test_data = train_test_data_setup(data)
        threshold = 0.5
    elif task_type == 'ts_forecasting':
        train_data, test_data = get_ts_data(forecast_length=5, validation_blocks=validation_blocks)
        threshold = np.std(test_data.target)
    else:
        raise ValueError('Incorrect type of machine learning task')
    return train_data, test_data, threshold


def get_multimodal_ts_data(size=500):
    file_path_train = 'cases/data/metocean/metocean_data_train.csv'
    full_path_train = os.path.join(str(fedot_project_root()), file_path_train)

    # a dataset for a final validation of the composed model
    file_path_test = 'cases/data/metocean/metocean_data_test.csv'
    full_path_test = os.path.join(str(fedot_project_root()), file_path_test)

    target_history, add_history, _ = prepare_input_data(full_path_train, full_path_test,
                                                        history_size=size)
    historical_data = {
        'ws': add_history,  # additional variable
        'ssh': target_history,  # target variable
    }
    return historical_data, target_history


def load_categorical_unimodal():
    dataset_path = 'test/data/classification_with_categorical.csv'
    full_path = os.path.join(str(fedot_project_root()), dataset_path)
    data = InputData.from_csv(full_path)
    train_data, test_data = train_test_data_setup(data, shuffle_flag=True)

    return train_data, test_data


def load_categorical_multidata():
    # Create features table
    features_first = np.array([[0, '  a'], [1, ' a '], [2, '  b'], [3, np.nan], [4, '  a'],
                               [5, '  b'], [6, 'b  '], [7, '  c'], [8, ' c ']], dtype=object)
    features_second = np.array([[10, '  a'], [11, ' a '], [12, '  b'], [13, ' a '], [14, '  a'],
                                [15, '  b'], [16, 'b  '], [17, '  c'], [18, ' c ']], dtype=object)
    # TODO @andreygetmanov (fails if target = ['true', 'false', ...])
    target = np.array([1, 0, 1, 0, 0, 0, 0, 1, 1])

    fit_data = {'first': features_first,
                'second': features_second}

    return fit_data, target


def data_with_binary_features_and_categorical_target():
    """
    A dataset is generated where features and target require transformations.
    The categorical binary features and categorical target must be converted to int
    """
    task = Task(TaskTypesEnum.classification)
    features = np.array([['red', 'blue'],
                         [np.nan, 'blue'],
                         ['green', 'blue'],
                         ['green', 'orange']])
    target = np.array(['red-blue', 'red-blue', 'green-blue', 'green-orange'])
    train_input = InputData(idx=[0, 1, 2, 3], features=features, target=target,
                            task=task, data_type=DataTypesEnum.table,
                            supplementary_data=SupplementaryData(was_preprocessed=False))

    return train_input


@pytest.mark.parametrize('task_type, predefined_model, metric_name', [
    (TaskTypesEnum.classification, 'dt', 'f1'),
    (TaskTypesEnum.regression, 'dtreg', 'rmse'),
])
def test_api_predict_correct(task_type, predefined_model, metric_name):
    task_type = task_type.value

    train_data, test_data, _ = get_dataset(task_type)
    model = Fedot(problem=task_type, **default_params)
    fedot_model = model.fit(features=train_data, predefined_model=predefined_model)
    prediction = model.predict(features=test_data)
    metric = model.get_metrics()
    assert isinstance(fedot_model, Pipeline)
    assert len(prediction) == len(test_data.target)
    assert metric[metric_name] > 0
    assert is_predict_ignores_target(model.predict, train_data, 'features')


def test_api_simple_ts_predict_correct(task_type: str = 'ts_forecasting'):
    # The forecast length must be equal to 5
    forecast_length = 5
    train_data, test_data, _ = get_dataset(task_type, validation_blocks=1)
    model = Fedot(problem='ts_forecasting', **default_params,
                  task_params=TsForecastingParams(forecast_length=forecast_length),
                  validation_blocks=1)

    model.fit(features=train_data, predefined_model='auto')
    ts_forecast = model.predict(features=test_data)
    _ = model.get_metrics(target=test_data.target, metric_names='rmse')

    assert len(ts_forecast) == forecast_length


@pytest.mark.parametrize('validation_blocks', [None, 2, 3])
def test_api_in_sample_ts_predict_correct(validation_blocks, task_type: str = 'ts_forecasting'):
    # The forecast length must be equal to 5
    forecast_length = 5
    train_data, test_data, _ = get_dataset(task_type, validation_blocks=validation_blocks)
    model = Fedot(problem='ts_forecasting', **default_params,
                  task_params=TsForecastingParams(forecast_length=forecast_length),
                  validation_blocks=validation_blocks)

    model.fit(features=train_data, predefined_model='auto')
    ts_forecast = model.predict(features=test_data, validation_blocks=validation_blocks)
    _ = model.get_metrics(target=test_data.target, metric_names='rmse', validation_blocks=validation_blocks)

    assert len(ts_forecast) == forecast_length * validation_blocks if validation_blocks else forecast_length * 2


@pytest.mark.parametrize('validation_blocks', [None, 2, 3])
def test_api_in_sample_multi_ts_predict_correct(validation_blocks, task_type: str = 'ts_forecasting'):
    forecast_length = 2
    train_data, test_data = get_multi_ts_data(forecast_length=forecast_length, validation_blocks=validation_blocks)
    model = Fedot(problem='ts_forecasting', **default_params,
                  task_params=TsForecastingParams(forecast_length=forecast_length),
                  validation_blocks=validation_blocks,
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

    model = Fedot(problem='ts_forecasting', **default_params,
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
    node_lagged = PrimaryNode('lagged')
    pipeline = Pipeline(SecondaryNode('linear', nodes_from=[node_lagged]))

    model.fit(features=train_data.features,
              target=train_data.target,
              predefined_model=pipeline)
    ts_forecast = model.predict(features=test_data, in_sample=False)
    metric = model.get_metrics(target=test_data.target, metric_names='rmse')

    assert len(ts_forecast) == forecast_length
    assert metric['rmse'] >= 0


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

    params = {
        **default_params,
        'metric': ['f1', 'node_num']
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

    auto_model = Fedot(problem='classification', **default_params)
    auto_model.fit(features=train_data)
    prediction = auto_model.predict(features=test_data)
    prediction_proba = auto_model.predict_proba(features=test_data)

    assert np.issubdtype(prediction.dtype, np.number)
    assert np.isnan(prediction).sum() == 0
    assert np.issubdtype(prediction_proba.dtype, np.number)
    assert np.isnan(prediction_proba).sum() == 0


def test_categorical_preprocessing_unidata_predefined():
    train_data, test_data = load_categorical_unimodal()

    auto_model = Fedot(problem='classification', **default_params)
    auto_model.fit(features=train_data, predefined_model='rf')
    prediction = auto_model.predict(features=test_data)
    prediction_proba = auto_model.predict_proba(features=test_data)

    assert np.issubdtype(prediction.dtype, np.number)
    assert np.isnan(prediction).sum() == 0
    assert np.issubdtype(prediction_proba.dtype, np.number)
    assert np.isnan(prediction_proba).sum() == 0


def test_categorical_preprocessing_unidata_predefined_linear():
    train_data, test_data = load_categorical_unimodal()

    pipeline = Pipeline(nodes=PrimaryNode('logit'))
    pipeline.fit(train_data)
    prediction = pipeline.predict(test_data)

    for i in range(prediction.features.shape[1]):
        assert all(list(map(lambda x: isinstance(x, (int, float)), prediction.features[:, i])))


def test_fill_nan_without_categorical():
    train_data, test_data = load_categorical_unimodal()
    train_data.features = np.hstack((train_data.features[:, :2], train_data.features[:, 4:]))
    test_data.features = np.hstack((test_data.features[:, :2], test_data.features[:, 4:]))

    pipeline = Pipeline(nodes=PrimaryNode('logit'))
    pipeline.fit(train_data)
    prediction = pipeline.predict(test_data)
    prediction_train = pipeline.predict(train_data)

    assert pd.isna(prediction.features).sum() == 0
    assert pd.isna(prediction_train.features).sum() == 0


def test_dict_multimodal_input_for_api():
    data, target = load_categorical_multidata()

    model = Fedot(problem='classification', **default_params)

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
        **default_params,
        'metric': 'f1'}

    auto_model = Fedot(problem=problem, seed=42, **params)
    pipeline = auto_model.fit(features=features, target=target)
    assert pipeline is not None


def test_custom_history_folder_define_correct():
    train_data, test_data, _ = get_dataset('ts_forecasting')

    custom_path = os.path.join(os.path.abspath(os.getcwd()), 'history_folder')

    params = {
        **default_params,
        'history_folder': custom_path,
        'timeout': None,
        'num_of_generations': 1,
        'pop_size': 3}

    model = Fedot(problem='ts_forecasting', **params,
                  task_params=TsForecastingParams(forecast_length=5))

    model.fit(features=train_data)

    assert len(os.listdir(custom_path)) != 0
    shutil.rmtree(custom_path)


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

    assert predicted.predict[-1] == 'green-orange'


def test_data_from_csv_load_correctly():
    """
    Check if data obtained from csv files processed correctly for fit and
    predict stages when for predict stage there is no target column in csv file
    """
    task = Task(TaskTypesEnum.regression)
    path_train = 'test/data/empty_target_tables/train.csv'
    path_test = 'test/data/empty_target_tables/test.csv'
    full_path_train = os.path.join(str(fedot_project_root()), path_train)
    full_path_test = os.path.join(str(fedot_project_root()), path_test)

    data_loader = ApiDataProcessor(task)
    train_input = data_loader.define_data(features=full_path_train, target='class')
    test_input = data_loader.define_data(features=full_path_test, is_predict=True)

    assert train_input.target.shape == (14, 1)
    assert test_input.target is None


def test_api_params():
    """ Test checks if all params from api are processed and divided correctly"""
    default_int_value = 2
    api_params = {'problem': 'ts_forecasting', 'timeout': default_int_value,
                  'task_params': TsForecastingParams(forecast_length=default_int_value), 'seed': default_int_value,
                  'logging_level': default_int_value, 'safe_mode': False, 'n_jobs': default_int_value,
                  'max_depth': default_int_value, 'max_arity': default_int_value,
                  'early_stopping_iterations': default_int_value, 'early_stopping_timeout': default_int_value,
                  'pop_size': default_int_value, 'num_of_generations': default_int_value,
                  'keep_n_best': default_int_value, 'available_operations': ['lagged', 'ridge'],
                  'with_tuning': True, 'cv_folds': default_int_value, 'max_pipeline_fit_time': default_int_value,
                  'initial_assumption': PipelineBuilder().add_node('lagged').add_node('ridge').to_pipeline(),
                  'genetic_scheme': GeneticSchemeTypesEnum.steady_state, 'history_folder': 'history',
                  'metric': RegressionMetricsEnum.SMAPE,
                  'collect_intermediate_metric': True, 'preset': 'fast_train',
                  'optimizer_external_params': {'path': default_int_value}}

    correct_api_params = {'n_jobs': default_int_value,
                          'task': Task(task_type=TaskTypesEnum.ts_forecasting,
                                       task_params=TsForecastingParams(forecast_length=2)),
                          'timeout': default_int_value,
                          'train_data': None}
    correct_composer_params = {'available_operations': ['lagged', 'ridge'],
                               'collect_intermediate_metric': True,
                               'metric': RegressionMetricsEnum.SMAPE,
                               'cv_folds': default_int_value,
                               'genetic_scheme': GeneticSchemeTypesEnum.steady_state,
                               'history_folder': 'history',
                               'initial_assumption': PipelineBuilder().add_node('lagged').add_node(
                                   'ridge').to_pipeline(),
                               'max_arity': default_int_value,
                               'max_depth': default_int_value,
                               'max_pipeline_fit_time': default_int_value,
                               'num_of_generations': default_int_value,
                               'keep_n_best': default_int_value,
                               'optimizer': None,
                               'pop_size': default_int_value,
                               'preset': 'fast_train',
                               'early_stopping_iterations': default_int_value,
                               'early_stopping_timeout': default_int_value,
                               'validation_blocks': default_int_value,
                               'optimizer_external_params': {'path': default_int_value},
                               'use_pipelines_cache': True, 'use_preprocessing_cache': True, 'cache_folder': None}
    correct_tuner_params = {'with_tuning': True}

    model = Fedot(**api_params)
    api_params, composer_params, tuner_params = _divide_parameters(model.params.api_params)

    for key in correct_api_params.keys():
        assert correct_api_params[key] == api_params[key]

    for key in correct_composer_params.keys():
        assert correct_composer_params[key] == composer_params[key]

    for key in correct_tuner_params.keys():
        assert correct_tuner_params[key] == tuner_params[key]


def test_unknown_param_raises_error():
    api_params = {'problem': 'classification', 'unknown': 2}
    try:
        model = Fedot(**api_params)
        _divide_parameters(model.params.api_params)
    except KeyError as e:
        assert str(e) == "'Invalid key parameter unknown'"


def test_default_forecast():
    forecast_length = 2
    train_data, test_data, _ = get_dataset('ts_forecasting')
    model = Fedot(problem='ts_forecasting', **default_params,
                  task_params=TsForecastingParams(forecast_length=forecast_length))
    model.fit(train_data, predefined_model='auto')
    forecast = model.forecast()
    assert len(forecast) == forecast_length
    assert np.array_equal(model.test_data.idx, train_data.idx)


@pytest.mark.parametrize('horizon', [1, 2, 3, 4])
def test_forecast_with_different_horizons(horizon):
    forecast_length = 2
    train_data, test_data, _ = get_dataset('ts_forecasting')
    model = Fedot(problem='ts_forecasting', **default_params,
                  task_params=TsForecastingParams(forecast_length=forecast_length))
    model.fit(train_data, predefined_model='auto')
    forecast = model.forecast(pre_history=test_data, horizon=horizon)
    assert len(forecast) == horizon
    assert np.array_equal(model.test_data.idx, test_data.idx)


def test_forecast_with_unfitted_model():
    forecast_length = 2
    model = Fedot(problem='ts_forecasting', **default_params,
                  task_params=TsForecastingParams(forecast_length=forecast_length))
    with pytest.raises(ValueError):
        model.forecast()


def test_forecast_with_not_ts_problem():
    model = Fedot(problem='classification', **default_params)
    train_data, test_data, _ = get_dataset('classification')
    model.fit(train_data, predefined_model='auto')
    with pytest.raises(ValueError):
        model.forecast(pre_history=test_data)


def test_forecast_with_multivariate_ts():
    forecast_length = 2

    historical_data, target = get_multimodal_ts_data()

    model = Fedot(problem='ts_forecasting', **default_params,
                  task_params=TsForecastingParams(forecast_length=forecast_length))
    model.fit(features=historical_data, target=target, predefined_model='auto')
    forecast = model.forecast()
    assert len(forecast) == forecast_length
    forecast = model.forecast(horizon=forecast_length - 1)
    assert len(forecast) == forecast_length - 1
    with pytest.raises(ValueError):
        model.forecast(horizon=forecast_length + 1)
