import numpy as np

from fedot.core.data.multi_modal import MultiModalData
from fedot.core.operations.evaluation.operation_implementations.data_operations.ts_transformations import \
    _prepare_target, _ts_to_table, _sparse_matrix
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.log import default_log
from data.data_manager import synthetic_univariate_ts, synthetic_with_exogenous_ts

WINDOW_SIZE = 4
FORECAST_LENGTH = 4
log = default_log(__name__)


def test_ts_to_lagged_table():
    # Check first step - lagged transformation of features
    train_input, _, _ = synthetic_univariate_ts(FORECAST_LENGTH)

    new_idx, lagged_table = _ts_to_table(idx=train_input.idx,
                                         time_series=train_input.features,
                                         window_size=WINDOW_SIZE)

    correct_lagged_table = ((0., 10., 20., 30.),
                            (10., 20., 30., 40.),
                            (20., 30., 40., 50.),
                            (30., 40., 50., 60.),
                            (40., 50., 60., 70.),
                            (50., 60., 70., 80.),
                            (60., 70., 80., 90.),
                            (70., 80., 90., 100.),
                            (80., 90., 100., 110.),
                            (90., 100., 110., 120.))

    correct_new_idx = (4, 5, 6, 7, 8, 9, 10, 11, 12, 13)

    # Convert into tuple for comparison
    new_idx_as_tuple = tuple(new_idx)
    lagged_table_as_tuple = tuple(map(tuple, lagged_table))
    assert lagged_table_as_tuple == correct_lagged_table
    assert new_idx_as_tuple == correct_new_idx

    # Second step - processing for correct the target
    final_idx, features_columns, final_target = _prepare_target(idx=new_idx,
                                                                features_columns=lagged_table,
                                                                target=train_input.target,
                                                                forecast_length=FORECAST_LENGTH)
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
    train_input, _, _ = synthetic_univariate_ts(FORECAST_LENGTH)
    _, lagged_table = _ts_to_table(idx=train_input.idx,
                                   time_series=train_input.features,
                                   window_size=WINDOW_SIZE)
    features_columns = _sparse_matrix(log, lagged_table)

    # assert if sparse matrix features less than half or less than another dimension
    assert features_columns.shape[0] == lagged_table.shape[0]
    assert features_columns.shape[1] <= lagged_table.shape[1]/2 or features_columns.shape[1] < lagged_table.shape[0]


def test_forecast_with_sparse_lagged():
    train_source_ts, predict_source_ts, train_exog_ts, predict_exog_ts, ts_test =\
        synthetic_with_exogenous_ts(FORECAST_LENGTH)

    node_lagged = PrimaryNode('sparse_lagged')
    # Set window size for lagged transformation
    node_lagged.custom_params = {'window_size': WINDOW_SIZE}

    node_final = SecondaryNode('linear', nodes_from=[node_lagged])
    pipeline = Pipeline(node_final)

    pipeline.fit(input_data=MultiModalData({'sparse_lagged': train_source_ts}))

    forecast = pipeline.predict(input_data=MultiModalData({'sparse_lagged': predict_source_ts}))
    is_forecasted = True

    assert is_forecasted


def test_forecast_with_exog():
    train_source_ts, predict_source_ts, train_exog_ts, predict_exog_ts, ts_test =\
        synthetic_with_exogenous_ts(FORECAST_LENGTH)

    # Source data for lagged node
    node_lagged = PrimaryNode('lagged')
    # Set window size for lagged transformation
    node_lagged.custom_params = {'window_size': WINDOW_SIZE}
    # Exogenous variable for exog node
    node_exog = PrimaryNode('exog_ts_data_source')

    node_final = SecondaryNode('linear', nodes_from=[node_lagged, node_exog])
    pipeline = Pipeline(node_final)

    pipeline.fit(input_data=MultiModalData({'exog_ts_data_source': train_exog_ts,
                                            'lagged': train_source_ts}))

    forecast = pipeline.predict(input_data=MultiModalData({'exog_ts_data_source': predict_exog_ts,
                                                           'lagged': predict_source_ts}))
    prediction = np.ravel(np.array(forecast.predict))

    assert tuple(prediction) == tuple(ts_test)
