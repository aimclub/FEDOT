import numpy as np

from fedot.core.data.data import InputData
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.log import default_log
from fedot.core.operations.evaluation.operation_implementations.data_operations.ts_transformations import (
    _sparse_matrix,
    prepare_target,
    ts_to_table
)
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams

window_size = 4
forecast_length = 4
log = default_log(prefix=__name__)


def synthetic_univariate_ts():
    """ Method returns InputData for classical time series forecasting task """
    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=forecast_length))
    # Simple time series to process
    ts_train = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130])
    ts_test = np.array([140, 150, 160, 170])

    # Prepare train data
    train_input = InputData(idx=np.arange(0, len(ts_train)),
                            features=ts_train,
                            target=ts_train,
                            task=task,
                            data_type=DataTypesEnum.ts)

    start_forecast = len(ts_train)
    end_forecast = start_forecast + forecast_length
    predict_input = InputData(idx=np.arange(start_forecast, end_forecast),
                              features=ts_train,
                              target=None,
                              task=task,
                              data_type=DataTypesEnum.ts)
    return train_input, predict_input, ts_test


def synthetic_with_exogenous_ts():
    """ Method returns InputData for time series forecasting task with
    exogenous variable """
    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=forecast_length))

    # Time series with exogenous variable
    ts_train = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130])
    ts_exog = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23])

    ts_test = np.array([140, 150, 160, 170])
    ts_test_exog = np.array([24, 25, 26, 27])

    # Indices for forecast
    start_forecast = len(ts_train)
    end_forecast = start_forecast + forecast_length

    # Input for source time series
    train_source_ts = InputData(idx=np.arange(0, len(ts_train)),
                                features=ts_train, target=ts_train,
                                task=task, data_type=DataTypesEnum.ts)
    predict_source_ts = InputData(idx=np.arange(start_forecast, end_forecast),
                                  features=ts_train, target=None,
                                  task=task, data_type=DataTypesEnum.ts)

    # Input for exogenous variable
    train_exog_ts = InputData(idx=np.arange(0, len(ts_train)),
                              features=ts_exog, target=ts_train,
                              task=task, data_type=DataTypesEnum.ts)
    predict_exog_ts = InputData(idx=np.arange(start_forecast, end_forecast),
                                features=ts_test_exog, target=None,
                                task=task, data_type=DataTypesEnum.ts)
    return train_source_ts, predict_source_ts, train_exog_ts, predict_exog_ts, ts_test


def test_ts_to_lagged_table():
    # Check first step - lagged transformation of features
    train_input, _, _ = synthetic_univariate_ts()
    new_idx, lagged_table = ts_to_table(idx=train_input.idx,
                                        time_series=train_input.features,
                                        window_size=window_size,
                                        is_lag=True)

    correct_lagged_table = ((0., 10., 20., 30.),
                            (10., 20., 30., 40.),
                            (20., 30., 40., 50.),
                            (30., 40., 50., 60.),
                            (40., 50., 60., 70.),
                            (50., 60., 70., 80.),
                            (60., 70., 80., 90.),
                            (70., 80., 90., 100.),
                            (80., 90., 100., 110.),
                            (90., 100., 110., 120.),
                            (100., 110., 120., 130.))

    correct_new_idx = (4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 13)

    # Convert into tuple for comparison
    new_idx_as_tuple = tuple(new_idx)
    lagged_table_as_tuple = tuple(map(tuple, lagged_table))
    assert lagged_table_as_tuple == correct_lagged_table
    assert new_idx_as_tuple == correct_new_idx

    # Second step - processing for correct the target
    final_idx, features_columns, final_target = prepare_target(all_idx=train_input.idx,
                                                               idx=new_idx,
                                                               features_columns=lagged_table,
                                                               target=train_input.target,
                                                               forecast_length=forecast_length)
    correct_final_idx = (4, 5, 6, 7, 8, 9, 10)
    correct_features_columns = ((0., 10., 20., 30.),
                                (10., 20., 30., 40.),
                                (20., 30., 40., 50.),
                                (30., 40., 50., 60.),
                                (40., 50., 60., 70.),
                                (50., 60., 70., 80.),
                                (60., 70., 80., 90.))

    correct_final_target = ((40., 50., 60., 70.),
                            (50., 60., 70., 80.),
                            (60., 70., 80., 90.),
                            (70., 80., 90., 100.),
                            (80., 90., 100., 110.),
                            (90., 100., 110., 120.),
                            (100., 110., 120., 130.))

    # Convert into tuple for comparison
    final_idx_as_tuple = tuple(final_idx)
    features_columns_as_tuple = tuple(map(tuple, features_columns))
    final_target_as_tuple = tuple(map(tuple, final_target))

    assert final_idx_as_tuple == correct_final_idx
    assert features_columns_as_tuple == correct_features_columns
    assert final_target_as_tuple == correct_final_target


def test_sparse_matrix():
    # Create lagged matrix for sparse
    train_input, _, _ = synthetic_univariate_ts()
    _, lagged_table = ts_to_table(idx=train_input.idx,
                                  time_series=train_input.features,
                                  window_size=window_size)
    features_columns = _sparse_matrix(log, lagged_table)

    # assert if sparse matrix features less than half or less than another dimension
    assert features_columns.shape[0] == lagged_table.shape[0]
    assert features_columns.shape[1] <= lagged_table.shape[1] / 2 or features_columns.shape[1] < lagged_table.shape[0]


def test_forecast_with_sparse_lagged():
    train_source_ts, predict_source_ts, train_exog_ts, predict_exog_ts, ts_test = synthetic_with_exogenous_ts()

    node_lagged = PrimaryNode('sparse_lagged')
    # Set window size for lagged transformation
    node_lagged.custom_params = {'window_size': window_size}

    node_final = SecondaryNode('linear', nodes_from=[node_lagged])
    pipeline = Pipeline(node_final)

    pipeline.fit(input_data=MultiModalData({'sparse_lagged': train_source_ts}))

    pipeline.predict(input_data=MultiModalData({'sparse_lagged': predict_source_ts}))
    is_forecasted = True

    assert is_forecasted


def test_forecast_with_exog():
    train_source_ts, predict_source_ts, train_exog_ts, predict_exog_ts, ts_test = synthetic_with_exogenous_ts()

    # Source data for lagged node
    node_lagged = PrimaryNode('lagged')
    # Set window size for lagged transformation
    node_lagged.custom_params = {'window_size': window_size}
    # Exogenous variable for exog node
    node_exog = PrimaryNode('exog_ts')

    node_final = SecondaryNode('linear', nodes_from=[node_lagged, node_exog])
    pipeline = Pipeline(node_final)

    pipeline.fit(input_data=MultiModalData({'exog_ts': train_exog_ts,
                                            'lagged': train_source_ts}))

    forecast = pipeline.predict(input_data=MultiModalData({'exog_ts': predict_exog_ts,
                                                           'lagged': predict_source_ts}))
    prediction = np.ravel(np.array(forecast.predict))

    assert tuple(prediction) == tuple(ts_test)
