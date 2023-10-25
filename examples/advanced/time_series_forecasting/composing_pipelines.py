import datetime

import numpy as np
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.mutation import MutationTypesEnum
from sklearn.metrics import mean_squared_error, mean_absolute_error

from examples.advanced.time_series_forecasting.multistep import get_border_line_info, visualise
from examples.simple.time_series_forecasting.api_forecasting import get_ts_data
from examples.simple.time_series_forecasting.ts_pipelines import ts_complex_ridge_pipeline
from fedot.core.composer.composer_builder import ComposerBuilder
from fedot.core.composer.gp_composer.specific_operators import parameter_change_mutation
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.pipeline_composer_requirements import PipelineComposerRequirements
from fedot.core.repository.quality_metrics_repository import \
    RegressionMetricsEnum


def get_available_operations():
    """ Function returns available operations for primary and secondary nodes """
    primary_operations = ['lagged', 'smoothing', 'gaussian_filter', 'ar']
    secondary_operations = ['lagged', 'ridge', 'lasso', 'knnreg', 'linear',
                            'scaling', 'ransac_lin_reg', 'rfe_lin_reg']
    return primary_operations, secondary_operations


def run_composing(dataset: str, pipeline: Pipeline, len_forecast=250):
    """ Example of ts forecasting using custom pipelines with composing
    :param dataset: name of dataset
    :param pipeline: pipeline to use
    :param len_forecast: forecast length
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

    rmse = mean_squared_error(test_target, predict, squared=False)
    mae = mean_absolute_error(test_target, predict)

    metrics_info['Metrics without composing'] = {'RMSE': round(rmse, 3),
                                                 'MAE': round(mae, 3)}
    plot_info.append({'idx': prediction.idx,
                      'series': predict,
                      'label': 'Forecast without composing'})
    plot_info.append(get_border_line_info(prediction.idx[0], predict, train_data.features, 'Border line'))

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
    )
    optimizer_parameters = GPAlgorithmParameters(
        pop_size=10,
        crossover_prob=0.8, mutation_prob=0.8,
        mutation_types=[parameter_change_mutation,
                        MutationTypesEnum.growth,
                        MutationTypesEnum.reduce,
                        MutationTypesEnum.simple]
    )
    composer = ComposerBuilder(train_data.task). \
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


if __name__ == '__main__':
    run_composing('m4_monthly', ts_complex_ridge_pipeline(), len_forecast=10)
