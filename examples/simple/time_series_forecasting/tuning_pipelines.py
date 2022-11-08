import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

from examples.advanced.time_series_forecasting.composing_pipelines import visualise, get_border_line_info
from examples.simple.time_series_forecasting.ts_pipelines import ts_locf_ridge_pipeline
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder
from fedot.core.pipelines.tuning.unified import PipelineTuner
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.quality_metrics_repository import RegressionMetricsEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.core.utils import fedot_project_root

datasets = {
    'australia': f'{fedot_project_root()}/examples/data/ts/australia.csv',
    'beer': f'{fedot_project_root()}/examples/data/ts/beer.csv',
    'salaries': f'{fedot_project_root()}/examples/data/ts/salaries.csv',
    'stackoverflow': f'{fedot_project_root()}/examples/data/ts/stackoverflow.csv'}


def run_experiment(dataset: str, pipeline: Pipeline, len_forecast=250, tuning=True, visualisalion=False):
    """ Example of ts forecasting using custom pipelines with optional tuning
    :param dataset: name of dataset
    :param pipeline: pipeline to use
    :param len_forecast: forecast length
    :param tuning: is tuning needed
    """
    # show initial pipeline
    pipeline.print_structure()

    time_series = pd.read_csv(datasets[dataset])

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
    metrics_info = {}
    plot_info.append({'idx': idx,
                      'series': time_series,
                      'label': 'Actual time series'})

    rmse = mean_squared_error(test_target, predict, squared=False)
    mae = mean_absolute_error(test_target, predict)

    metrics_info['Metrics without tuning'] = {'RMSE': round(rmse, 3),
                                              'MAE': round(mae, 3)}
    plot_info.append({'idx': prediction.idx,
                      'series': predict,
                      'label': 'Forecast without tuning'})
    plot_info.append(get_border_line_info(prediction.idx[0], predict, time_series, 'Border line'))

    if tuning:
        tuner = TunerBuilder(task)\
            .with_tuner(PipelineTuner)\
            .with_metric(RegressionMetricsEnum.MSE)\
            .with_iterations(100) \
            .build(train_data)
        pipeline = tuner.tune(pipeline)
        pipeline.fit(train_data)
        prediction_after = pipeline.predict(test_data)
        predict_after = np.ravel(np.array(prediction_after.predict))

        rmse = mean_squared_error(test_target, predict_after, squared=False)
        mae = mean_absolute_error(test_target, predict_after)

        metrics_info['Metrics after tuning'] = {'RMSE': round(rmse, 3),
                                                'MAE': round(mae, 3)}
        plot_info.append({'idx': prediction_after.idx,
                          'series': predict_after,
                          'label': 'Forecast after tuning'})

    print(metrics_info)
    # plot lines
    if visualisalion:
        visualise(plot_info)
        pipeline.print_structure()


if __name__ == '__main__':
    run_experiment('australia', ts_locf_ridge_pipeline(), len_forecast=50, tuning=True, visualisalion=True)
