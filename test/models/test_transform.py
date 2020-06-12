import numpy as np
from numpy import nan

from core.models.data import InputData
from core.models.transformation import direct, \
    ts_to_lagged_3d, ts_to_lagged_table
from core.repository.dataset_types import DataTypesEnum
from core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams

max_window_size = 4
forecast_length = 2


def synthetic_forecasting_problem(forecast_length: int, max_window_size: int):
    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=forecast_length,
                                    max_window_size=max_window_size))
    ts_len = 10
    ts = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])

    exog_variable = np.asarray([10.0, 11.0, 12.0, 13.0, 14.0,
                                15.0, 16.0, 17.0, 18.0, 19.0]).reshape(-1, 1)

    ts_data = InputData(idx=range(len(ts)), features=exog_variable, target=ts,
                        task=task, data_type=DataTypesEnum.table)

    # shape is (10, 4, 1)
    exog_variable_3d = np.asarray([[[nan], [nan], [nan], [nan]],
                                   [[nan], [nan], [nan], [10.0]],
                                   [[nan], [nan], [10.0], [11.0]],
                                   [[nan], [10.0], [11.0], [12.0]],
                                   [[10.0], [11.0], [12.0], [13.0]],
                                   [[11.0], [12.0], [13.0], [14.0]],
                                   [[12.0], [13.0], [14.0], [15.0]],
                                   [[13.0], [14.0], [15.0], [16.0]],
                                   [[14.0], [15.0], [16.0], [17.0]],
                                   [[15.0], [16.0], [17.0], [18.0]]])
    # [[16.0], [17.0], [18.0], [19.0]],
    # [[17.0], [18.0], [19.0], [nan]],
    # [[18.0], [19.0], [nan], [nan]],
    # [[19.0], [nan], [nan], [nan]],
    # [[nan], [nan], [nan], [nan]]])

    lagged_target_3d_as_feature = np.asarray([[[nan], [nan], [nan], [nan]],
                                              [[nan], [nan], [nan], [0.0]],
                                              [[nan], [nan], [0.0], [1.0]],
                                              [[nan], [0.0], [1.0], [2.0]],
                                              [[0.0], [1.0], [2.0], [3.0]],
                                              [[1.0], [2.0], [3.0], [4.0]],
                                              [[2.0], [3.0], [4.0], [5.0]],
                                              [[3.0], [4.0], [5.0], [6.0]],
                                              [[4.0], [5.0], [6.0], [7.0]],
                                              [[5.0], [6.0], [7.0], [8.0]]])
    # [[6.0], [7.0], [8.0], [9.0]]
    # [[7.0], [8.0], [9.0], [nan]],
    # [[8.0], [9.0], [nan], [nan]],
    # [[9.0], [nan], [nan], [nan]],
    # [[nan], [nan], [nan], [nan]]])

    # now we concat exog lagged variables as well as target lagged
    # so features now is (15, 4, 2), i.e.
    # (n+max_window_size+forecast_length, max_window_size, amount_exog_features+target_shape)
    feature_3d = np.concatenate((exog_variable_3d, lagged_target_3d_as_feature), axis=2)

    # target is (10, 4, 1)
    # (n, max_window_size, target_shape)
    # So lstm returns predictions with same shape
    # To get only forecast values do next:
    # pred_3d[:, -forecast_length:, :]
    # i.e. we take values only from last `forecast_length` timestamps
    target_3d_full = np.asarray([[[0.0], [1.0], [2.0], [3.0]],
                                 [[1.0], [2.0], [3.0], [4.0]],
                                 [[2.0], [3.0], [4.0], [5.0]],
                                 [[3.0], [4.0], [5.0], [6.0]],
                                 [[4.0], [5.0], [6.0], [7.0]],
                                 [[5.0], [6.0], [7.0], [8.0]],
                                 [[6.0], [7.0], [8.0], [9.0]],
                                 [[7.0], [8.0], [9.0], [nan]],
                                 [[8.0], [9.0], [nan], [nan]],
                                 [[9.0], [nan], [nan], [nan]]])

    # lagged format contains only the values to forecast (future values) in the target
    # this format convinient to use with classic regression modules
    # shape is (5, 2), i.e. (n-max_window_size-forecast_length+1, forecast_length * target_shape)
    target_lagged = np.asarray([[0.0, 1.0],
                                [1.0, 2.0],
                                [2.0, 3.0],
                                [3.0, 4.0],
                                [4.0, 5.0],
                                [5.0, 6.0],
                                [6.0, 7.0],
                                [7.0, 8.0],
                                [8.0, 9.0],
                                [9.0, nan]])

    target_3d = np.asarray([[[0.0], [1.0]],
                            [[1.0], [2.0]],
                            [[2.0], [3.0]],
                            [[3.0], [4.0]],
                            [[4.0], [5.0]],
                            [[5.0], [6.0]],
                            [[6.0], [7.0]],
                            [[7.0], [8.0]],
                            [[8.0], [9.0]],
                            [[9.0], [nan]]])

    ts_data_3d = InputData(idx=range(len(ts)), features=feature_3d, target=target_3d,
                           task=task, data_type=DataTypesEnum.ts_lagged_3d)

    # in lagged format all features are as feature_3d, but in 2d format
    # with shape (n, max_window_size * (amount_exog_features + target_shape))
    features_lagged = feature_3d.reshape(feature_3d.shape[0], -1)

    ts_data_lagged = InputData(idx=range(len(ts)), features=features_lagged, target=target_lagged,
                               task=task, data_type=DataTypesEnum.ts_lagged_table)

    return task, ts_len, ts_data, ts_data_3d, ts_data_lagged


def test_ts_to_lagged_3d():
    task, ts_len, ts_data, ts_data_3d, ts_data_lagged = \
        synthetic_forecasting_problem(forecast_length=forecast_length,
                                      max_window_size=max_window_size)

    transformed_data = ts_to_lagged_3d(ts_data)

    assert np.allclose(transformed_data.features.shape,
                       [ts_len,
                        max_window_size, 2])  # target_shape + amount_exog_features
    assert np.allclose(transformed_data.features, ts_data_3d.features, equal_nan=True)

    assert np.equal(transformed_data.target.shape,
                    [ts_len, forecast_length, 1]).all()
    assert np.allclose(transformed_data.target, ts_data_3d.target, equal_nan=True)


def test_ts_to_lagged_table():
    task, ts_len, ts_data, ts_data_3d, ts_data_lagged = \
        synthetic_forecasting_problem(forecast_length=forecast_length,
                                      max_window_size=max_window_size)

    transformed_data = ts_to_lagged_table(ts_data)

    assert np.equal(transformed_data.target.shape,
                    [ts_len,
                     forecast_length]).all()
    assert np.allclose(transformed_data.target, ts_data_lagged.target, equal_nan=True)

    # 2 is equal to amount_exog_features + target_shape
    assert np.equal(transformed_data.features.shape,
                    [ts_len,
                     max_window_size * 2]).all()

    assert np.allclose(transformed_data.features,
                       ts_data_lagged.features, equal_nan=True)


def test_direct():
    task, ts_len, ts_data, ts_data_3d, ts_data_lagged = \
        synthetic_forecasting_problem(forecast_length=forecast_length,
                                      max_window_size=max_window_size)

    transformed_data = direct(ts_data)

    print(transformed_data.target.shape)
    assert np.equal(transformed_data.target.shape,
                    [ts_len]).all()
    assert np.equal(transformed_data.target, ts_data.target).all()
    print(transformed_data.features.shape)
