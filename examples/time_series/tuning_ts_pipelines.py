import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

from examples.time_series.composing_ts_pipelines import display_metric, visualise
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from examples.time_series.pipelines import *

datasets = {
    'australia': '../data/ts/australia.csv',
    'beer': '../data/ts/beer.csv',
    'salaries': '../data/ts/salaries.csv',
    'stackoverflow': '../data/ts/stackoverflow.csv',
}


def run_experiment(dataset, pipeline, len_forecast=250, tuning=True):
    """ Function with example of ts forecasting with different models (with optional tuning)
    :param dataset: name of dataset
    :param pipeline: pipeline to use
    :param len_forecast: forecast length
    :param tuning: is tuning needed
    """
    time_series = pd.read_csv(datasets[dataset])
    # Let's divide our data on train and test samples
    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=len_forecast))
    if dataset not in ['australia']:
        idx = pd.to_datetime(time_series['idx'].values)
    else:
        # non datetime indexes
        idx = time_series['idx'].values
    time_series = time_series['value'].values
    train_input = InputData(idx=idx,
                            features=time_series,
                            target=time_series,
                            task=task,
                            data_type=DataTypesEnum.ts)
    train_data, test_data = train_test_data_setup(train_input)
    test_target = np.ravel(test_data.target)

    pipeline.fit(train_data)

    prediction = pipeline.predict(test_data)
    predict = np.ravel(np.array(prediction.predict))

    plot_info = []
    metrics_info = []
    plot_info.append({'idx': idx,
                      'series': time_series,
                      'label': 'Actual time series'
                      })

    rmse = mean_squared_error(test_target, predict, squared=False)
    mae = mean_absolute_error(test_target, predict)

    metrics_info.append(f'RMSE without tuning - {rmse:.4f}')
    metrics_info.append(f'MAE without tuning - {mae:.4f}')
    plot_info.append({'idx': prediction.idx,
                      'series': predict,
                      'label': 'Forecast without tuning'
                      })
    plot_info.append({'idx': [prediction.idx[0], prediction.idx[0]],
                      'series': [
                          min(np.concatenate([np.ravel(time_series), predict])),
                          max(np.concatenate([np.ravel(time_series), predict]))
                      ],
                      'label': 'Border line',
                      'color': 'black'
                      })

    if tuning:
        pipeline = pipeline.fine_tune_all_nodes(input_data=train_input,
                                                loss_function=mean_squared_error,
                                                loss_params={'squared': False},
                                                iterations=100
                                                )

        prediction_after = pipeline.predict(test_data)
        predict_after = np.ravel(np.array(prediction_after.predict))

        rmse = mean_squared_error(test_target, predict_after, squared=False)
        mae = mean_absolute_error(test_target, predict_after)

        metrics_info.append(f'RMSE after tuning - {rmse:.4f}')
        metrics_info.append(f'MAE after tuning - {mae:.4f}')
        plot_info.append({'idx': prediction_after.idx,
                          'series': predict_after,
                          'label': 'Forecast after tuning'
                          })

    display_metric(metrics_info)
    pipeline.print_structure()
    visualise(plot_info)


if __name__ == '__main__':
    run_experiment('salaries', clstm_pipeline(), len_forecast=30, tuning=False)
