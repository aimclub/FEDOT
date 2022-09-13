import datetime
from copy import deepcopy

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

from cases.multi_ts_level_forecasting import prepare_data
from examples.simple.time_series_forecasting.ts_pipelines import ts_complex_ridge_smoothing_pipeline
from fedot.core.composer.composer_builder import ComposerBuilder
from fedot.core.composer.gp_composer.specific_operators import parameter_change_mutation
from fedot.core.optimisers.gp_comp.gp_params import GPGraphOptimizerParameters
from fedot.core.optimisers.gp_comp.operators.mutation import MutationTypesEnum
from fedot.core.optimisers.gp_comp.pipeline_composer_requirements import PipelineComposerRequirements
from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder
from fedot.core.pipelines.tuning.unified import PipelineTuner
from fedot.core.repository.quality_metrics_repository import \
    RegressionMetricsEnum


def calculate_metrics(target, predicted):
    rmse = mean_squared_error(target, predicted, squared=True)
    mae = mean_absolute_error(target, predicted)
    return rmse, mae


def get_available_operations():
    """ Function returns available operations for primary and secondary nodes """
    primary_operations = ['lagged', 'smoothing', 'diff_filter', 'gaussian_filter']
    secondary_operations = ['lagged', 'ridge', 'lasso', 'linear']
    return primary_operations, secondary_operations


def compose_pipeline(pipeline, train_data, task):
    # pipeline structure optimization
    primary_operations, secondary_operations = get_available_operations()
    composer_requirements = PipelineComposerRequirements(
        primary=primary_operations,
        secondary=secondary_operations,
        max_arity=3, max_depth=5,
        num_of_generations=30,
        timeout=datetime.timedelta(minutes=10))
    optimizer_parameters = GPGraphOptimizerParameters(
        pop_size=15,
        mutation_prob=0.8, crossover_prob=0.8,
        mutation_types=[parameter_change_mutation,
                        MutationTypesEnum.single_change,
                        MutationTypesEnum.single_drop,
                        MutationTypesEnum.single_add]
    )
    composer = ComposerBuilder(task=task). \
        with_optimizer_params(optimizer_parameters). \
        with_requirements(composer_requirements). \
        with_metrics(RegressionMetricsEnum.MAE). \
        with_initial_pipelines([pipeline]). \
        build()
    obtained_pipeline = composer.compose_pipeline(data=train_data)
    obtained_pipeline.show()
    return obtained_pipeline


def run_multiple_ts_forecasting(forecast_length, is_multi_ts):
    # separate data on test/train
    train_data, test_data, task = prepare_data(forecast_length, is_multi_ts=is_multi_ts)
    # pipeline initialization
    pipeline = ts_complex_ridge_smoothing_pipeline()
    # pipeline fit and predict
    pipeline.fit(train_data)
    prediction_before = np.ravel(np.array(pipeline.predict(test_data).predict))
    # metrics evaluation
    rmse, mae = calculate_metrics(np.ravel(test_data.target), prediction_before)

    # compose pipeline with initial approximation
    obtained_pipeline = compose_pipeline(pipeline, train_data, task)
    # composed pipeline fit and predict
    obtained_pipeline_copy = deepcopy(obtained_pipeline)
    obtained_pipeline.fit_from_scratch(train_data)
    prediction_after = obtained_pipeline.predict(test_data)
    predict_after = np.ravel(np.array(prediction_after.predict))
    # metrics evaluation
    rmse_composing, mae_composing = calculate_metrics(np.ravel(test_data.target), predict_after)

    # tuning composed pipeline
    tuner = TunerBuilder(task)\
        .with_tuner(PipelineTuner)\
        .with_metric(RegressionMetricsEnum.MAE)\
        .with_iterations(50)\
        .build(train_data)
    obtained_pipeline_copy = tuner.tune(obtained_pipeline_copy)
    obtained_pipeline_copy.print_structure()
    # tuned pipeline fit and predict
    obtained_pipeline_copy.fit_from_scratch(train_data)
    prediction_after_tuning = obtained_pipeline_copy.predict(test_data)
    predict_after_tuning = np.ravel(np.array(prediction_after_tuning.predict))
    # metrics evaluation
    rmse_tuning, mae_tuning = calculate_metrics(np.ravel(test_data.target), predict_after_tuning)

    # visualization of results
    if is_multi_ts:
        history = np.ravel(train_data.target[:, 0])
    else:
        history = np.ravel(train_data.target)
    plt.plot(np.ravel(test_data.idx), np.ravel(test_data.target), label='test')
    plt.plot(np.ravel(train_data.idx), history, label='history')
    plt.plot(np.ravel(test_data.idx), prediction_before, label='prediction')
    plt.plot(np.ravel(test_data.idx), predict_after, label='prediction_after_composing')
    plt.plot(np.ravel(test_data.idx), predict_after_tuning, label='prediction_after_tuning')
    plt.xlabel('Time step')
    plt.ylabel('Sea level')
    plt.legend()
    plt.show()

    print(f'RMSE: {round(rmse, 3)}')
    print(f'MAE: {round(mae, 3)}')
    print(f'RMSE after composing: {round(rmse_composing, 3)}')
    print(f'MAE after composing: {round(mae_composing, 3)}')
    print(f'RMSE after tuning: {round(rmse_tuning, 3)}')
    print(f'MAE after tuning: {round(mae_tuning, 3)}')


if __name__ == '__main__':
    run_multiple_ts_forecasting(forecast_length=60, is_multi_ts=True)
