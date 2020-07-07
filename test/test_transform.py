import numpy as np
import pytest

from core.models.data import InputData
from core.models.transformation import direct, ts_lagged_3d_to_ts, \
    ts_to_lagged_3d, ts_to_lagged_table, ts_lagged_to_ts, \
    ts_lagged_table_to_3d, ts_lagged_3d_to_lagged_table
from core.repository.dataset_types import DataTypesEnum
from core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams

max_window_size = 4
forecast_length = 2


def synthetic_forecasting_problem(forecast_length: int, max_window_size: int):
    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=forecast_length,
                                    max_window_size=max_window_size))
    ts_len = 10
    ts = np.asarray(range(ts_len))

    exog_variable = 10 + np.asarray(range(ts_len)).reshape(-1, 1)

    ts_data = InputData(idx=range(len(ts)), features=exog_variable, target=ts,
                        task=task, data_type=DataTypesEnum.table)




    # shape is (5, 4, 1)
    exog_variable_3d = np.asarray([[[10], [11], [12], [13]],
                                   [[11], [12], [13], [14]],
                                   [[12], [13], [14], [15]],
                                   [[13], [14], [15], [16]],
                                   [[14], [15], [16], [17]]])
    
    lagged_target_3d_as_feature = np.asarray([[[0], [1], [2], [3]],
                                              [[1], [2], [3], [4]],
                                              [[2], [3], [4], [5]],
                                              [[3], [4], [5], [6]],
                                              [[4], [5], [6], [7]]])

    # now we concat exog lagged variables as well as target lagged
    # so features now is (5, 4, 2), i.e.
    # (n-max_window_size-forecast_length+1, max_window_size, amount_exog_features+target_shape)
    feature_3d = np.concatenate((exog_variable_3d, lagged_target_3d_as_feature), axis=2)

    # target is (5, 4, 1)
    # (n-max_window_size-forecast_length+1, max_window_size, target_shape)
    # So lstm returns predictions with same shape
    # To get only forecast values do next:
    # pred_3d[:, -forecast_length:, :]
    # i.e. we take values only from last `forecast_length` timestamps
    target_3d = np.asarray([[[2], [3], [4], [5]],
                            [[3], [4], [5], [6]],
                            [[4], [5], [6], [7]],
                            [[5], [6], [7], [8]],
                            [[6], [7], [8], [9]]])

    ts_data_3d = InputData(idx=range(len(ts)), features=feature_3d, target=target_3d,
                        task=task, data_type=DataTypesEnum.ts_lagged_3d)





    # lagged format contains only the values to forecast (future values) in the target
    # this format convinient to use with classic regression modules
    # shape is (5, 2), i.e. (n-max_window_size-forecast_length+1, forecast_length * target_shape)
    target_lagged = np.asarray([[4, 5],
                                [5, 6],
                                [6, 7],
                                [7, 8],
                                [8, 9]])

    # in lagged format all features are as feature_3d, but in 2d format 
    # with shape (n-max_window_size-forecast_length+1, max_window_size * (amount_exog_features + target_shape))
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
                       [ts_len - task.task_params.max_window_size - forecast_length + 1,
                        max_window_size,
                        2]) # amount_exog_features + target_shape
    assert np.allclose(transformed_data.features, ts_data_3d.features)


    assert np.equal(transformed_data.target.shape,
                    [ts_len - task.task_params.max_window_size - forecast_length + 1,
                     max_window_size,
                     1]).all()
    assert np.equal(transformed_data.target, ts_data_3d.target).all()
    



def test_lagged_3d_to_ts():
    task, ts_len, ts_data, ts_data_3d, ts_data_lagged = \
        synthetic_forecasting_problem(forecast_length=forecast_length,
                                      max_window_size=max_window_size)

    transformed_data = ts_lagged_3d_to_ts(ts_data_3d)

    assert np.equal(transformed_data.target.shape,
                    [ts_len]).all()
    assert np.equal(transformed_data.target, ts_data.target).all()
    
    assert np.allclose(transformed_data.features.shape,
                       [ts_len, 1])
    
    # when ts data was transformed to lagged_3d, 
    # last `prediction_len` features is lost, so it is not considered in comparing
    assert np.allclose(transformed_data.features[:-forecast_length],
                       ts_data.features[:-forecast_length])



def test_ts_to_lagged_table():
    task, ts_len, ts_data, ts_data_3d, ts_data_lagged = \
        synthetic_forecasting_problem(forecast_length=forecast_length,
                                      max_window_size=max_window_size)

    transformed_data = ts_to_lagged_table(ts_data)

    
    assert np.equal(transformed_data.target.shape,
                    [ts_len - task.task_params.max_window_size - forecast_length + 1, 
                    forecast_length]).all()
    assert np.equal(transformed_data.target, ts_data_lagged.target).all()
    
    # 2 is equal to amount_exog_features + target_shape
    assert np.allclose(transformed_data.features.shape,
                       [ts_len - task.task_params.max_window_size - forecast_length + 1, 
                        max_window_size * 2])
    
    assert np.allclose(transformed_data.features,
                       ts_data_lagged.features)


def test_lagged_table_to_ts():
    task, ts_len, ts_data, ts_data_3d, ts_data_lagged = \
        synthetic_forecasting_problem(forecast_length=forecast_length,
                                      max_window_size=max_window_size)

    transformed_data = ts_lagged_to_ts(ts_data_lagged)

    assert np.allclose(transformed_data.target.shape,
                       [ts_len])

    assert np.allclose(transformed_data.target,
                       ts_data.target)


    assert np.allclose(transformed_data.features.shape,
                       [ts_len, 1])
    
    # when ts data was transformed to lagged_format, 
    # last `prediction_len` features is lost, so it is not considered in comparing
    assert np.allclose(transformed_data.features[:-forecast_length],
                       ts_data.features[:-forecast_length])


def test_lagged_3d_to_lagged_ts():
    task, ts_len, ts_data, ts_data_3d, ts_data_lagged = \
        synthetic_forecasting_problem(forecast_length=forecast_length,
                                      max_window_size=max_window_size)

    transformed_data = ts_lagged_3d_to_lagged_table(ts_data_3d)

    assert np.allclose(transformed_data.target.shape,
                       [ts_len - task.task_params.max_window_size - forecast_length + 1, 
                       2])

    assert np.allclose(transformed_data.target,
                       ts_data_lagged.target)

    # 2 is equal to amount_exog_features + target_shape
    assert np.allclose(transformed_data.features.shape,
                       [ts_len - task.task_params.max_window_size - forecast_length + 1, 
                       2 * max_window_size])
    
    # when ts data was transformed to lagged_format, 
    # last `prediction_len` features is lost, so it is not considered in comparing
    assert np.allclose(transformed_data.features,
                       ts_data_lagged.features)


    

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
