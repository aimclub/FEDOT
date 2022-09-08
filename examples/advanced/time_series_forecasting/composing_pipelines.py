import datetime
from typing import Any, List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

from examples.simple.time_series_forecasting.ts_pipelines import ts_complex_ridge_pipeline
from fedot.core.composer.composer_builder import ComposerBuilder
from fedot.core.composer.gp_composer.specific_operators import parameter_change_mutation
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.optimisers.gp_comp.gp_params import GPGraphOptimizerParameters
from fedot.core.optimisers.gp_comp.operators.mutation import MutationTypesEnum
from fedot.core.optimisers.gp_comp.pipeline_composer_requirements import PipelineComposerRequirements
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.quality_metrics_repository import \
    RegressionMetricsEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.core.utils import fedot_project_root


def get_available_operations():
    """ Function returns available operations for primary and secondary nodes """
    primary_operations = ['lagged', 'smoothing', 'gaussian_filter', 'ar']
    secondary_operations = ['lagged', 'ridge', 'lasso', 'knnreg', 'linear',
                            'scaling', 'ransac_lin_reg', 'rfe_lin_reg']
    return primary_operations, secondary_operations


datasets = {
    'australia': f'{fedot_project_root()}/examples/data/ts/australia.csv',
    'beer': f'{fedot_project_root()}/examples/data/ts/beer.csv',
    'salaries': f'{fedot_project_root()}/examples/data/ts/salaries.csv',
    'stackoverflow': f'{fedot_project_root()}/examples/data/ts/stackoverflow.csv'}


def run_composing(dataset: str, pipeline: Pipeline, len_forecast=250):
    """ Example of ts forecasting using custom pipelines with composing
    :param dataset: name of dataset
    :param pipeline: pipeline to use
    :param len_forecast: forecast length
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

    metrics_info['Metrics without composing'] = {'RMSE': round(rmse, 3),
                                                 'MAE': round(mae, 3)}
    plot_info.append({'idx': prediction.idx,
                      'series': predict,
                      'label': 'Forecast without composing'})
    plot_info.append(get_border_line_info(prediction.idx[0], predict, time_series, 'Border line'))

    # Get available_operations type
    primary_operations, secondary_operations = get_available_operations()

    # Composer parameters
    composer_requirements = PipelineComposerRequirements(
        primary=primary_operations,
        secondary=secondary_operations,
        max_arity=3, max_depth=8,
        num_of_generations=10,
        timeout=datetime.timedelta(minutes=10),
        cv_folds=2,
        validation_blocks=2
    )
    optimizer_parameters = GPGraphOptimizerParameters(
        pop_size=10,
        crossover_prob=0.8, mutation_prob=0.8,
        mutation_types=[parameter_change_mutation,
                        MutationTypesEnum.growth,
                        MutationTypesEnum.reduce,
                        MutationTypesEnum.simple]
    )
    composer = ComposerBuilder(task). \
        with_requirements(composer_requirements). \
        with_optimizer_params(optimizer_parameters). \
        with_metrics(RegressionMetricsEnum.RMSE). \
        with_initial_pipelines([pipeline]). \
        build()

    obtained_pipeline = composer.compose_pipeline(data=train_data)

    obtained_pipeline.fit_from_scratch(train_data)
    prediction_after = obtained_pipeline.predict(test_data)
    predict_after = np.ravel(np.array(prediction_after.predict))

    rmse = mean_squared_error(test_target, predict_after, squared=False)
    mae = mean_absolute_error(test_target, predict_after)

    metrics_info['Metrics after composing'] = {'RMSE': round(rmse, 3),
                                               'MAE': round(mae, 3)}
    plot_info.append({'idx': prediction_after.idx,
                      'series': predict_after,
                      'label': 'Forecast after composing'})
    print(metrics_info)

    visualise(plot_info)
    # structure of obtained pipeline
    obtained_pipeline.print_structure()
    obtained_pipeline.show()


def visualise(plot_info: List[dict]):
    """
    Creates a plot based on plot_info

    :param plot_info: list of parameters for plot:
    The possible parameters are:
            'idx' - idx (or x axis data)
            'series' - data to plot (or y axis data)
            'label' - label for legend
            'color' - (optional) color of line
    """
    plt.figure()
    for p in plot_info:
        color = p.get('color')
        plt.plot(p['idx'], p['series'], label=p['label'], color=color)
    plt.legend()
    plt.grid()
    plt.show()


def get_border_line_info(idx: Any, predict: np.array, time_series: np.array, label: str, color: str = 'black') -> dict:
    """
    Return plot_info for border vertical line that divides train and test part of data

    :param idx: idx for vertical line
    :param predict: predictions
    :param time_series: full time series with test_data
    :param label: label for a legend
    :parma color: color of a line
    """
    return {'idx': [idx, idx],
            'series': [min(np.concatenate([np.ravel(time_series), predict])),
                       max(np.concatenate([np.ravel(time_series), predict]))],
            'label': label,
            'color': color}


if __name__ == '__main__':
    run_composing('australia', ts_complex_ridge_pipeline(), len_forecast=10)
