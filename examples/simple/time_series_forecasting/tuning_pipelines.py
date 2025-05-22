import numpy as np
from golem.core.tuning.simultaneous import SimultaneousTuner
from sklearn.metrics import mean_absolute_error

from examples.advanced.time_series_forecasting.composing_pipelines import visualise, get_border_line_info
from examples.simple.time_series_forecasting.api_forecasting import get_ts_data
from examples.simple.time_series_forecasting.ts_pipelines import ts_locf_ridge_pipeline
from fedot.core.composer.metrics import root_mean_squared_error
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder
from fedot.core.repository.metrics_repository import RegressionMetricsEnum


def run_experiment(dataset: str, pipeline: Pipeline, len_forecast=250, tuning=True, visualisalion=False):
    """ Example of ts forecasting using custom pipelines with optional tuning
    :param dataset: name of dataset
    :param pipeline: pipeline to use
    :param len_forecast: forecast length
    :param tuning: is tuning needed
    """
    # show initial pipeline
    pipeline.print_structure()

    train_data, test_data, label = get_ts_data(dataset, len_forecast)
    test_target = np.ravel(test_data.target)

    pipeline.fit(train_data)

    prediction = pipeline.predict(test_data)
    predict = np.ravel(np.array(prediction.predict))

    plot_info = []
    metrics_info = {}
    plot_info.append({'idx': np.concatenate([train_data.idx, test_data.idx]),
                      'series': np.concatenate([test_data.features, test_data.target]),
                      'label': 'Actual time series'})

    rmse = root_mean_squared_error(test_target, predict)
    mae = mean_absolute_error(test_target, predict)

    metrics_info['Metrics without tuning'] = {'RMSE': round(rmse, 3),
                                              'MAE': round(mae, 3)}
    plot_info.append({'idx': prediction.idx,
                      'series': predict,
                      'label': 'Forecast without tuning'})
    plot_info.append(get_border_line_info(prediction.idx[0], predict, train_data.features, 'Border line'))

    if tuning:
        tuner = TunerBuilder(train_data.task) \
            .with_tuner(SimultaneousTuner) \
            .with_metric(RegressionMetricsEnum.MSE) \
            .with_iterations(300) \
            .build(train_data)
        pipeline = tuner.tune(pipeline)
        pipeline.fit(train_data)
        prediction_after = pipeline.predict(test_data)
        predict_after = np.ravel(np.array(prediction_after.predict))

        rmse = root_mean_squared_error(test_target, predict_after)
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
    run_experiment('m4_monthly', ts_locf_ridge_pipeline(), len_forecast=10, tuning=True, visualisalion=True)
