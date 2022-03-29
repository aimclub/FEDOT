import datetime

import numpy as np
import os
import pandas as pd
from hyperopt import hp
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error

from fedot.core.composer.composer_builder import ComposerBuilder
from fedot.core.composer.gp_composer.gp_composer import PipelineComposerRequirements
from fedot.core.composer.gp_composer.specific_operators import parameter_change_mutation
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.optimisers.gp_comp.gp_optimiser import GPGraphOptimiserParameters
from fedot.core.optimisers.gp_comp.operators.mutation import MutationTypesEnum
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.tuning.search_space import SearchSpace
from fedot.core.pipelines.tuning.unified import PipelineTuner
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.quality_metrics_repository import RegressionMetricsEnum, MetricsRepository
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.core.utils import fedot_project_root
from examples.advanced.time_series_forecasting.multi_ts_arctic_forecasting import calculate_metrics
from fedot.core.repository.default_params_repository import DefaultOperationParamsRepository


def nemo_domain_model(fitted_model: any, idx: np.array, predict_data: np.array, params: dict):
    norm = params.get('norm')
    project_root_path = str(fedot_project_root())
    df = pd.read_csv(os.path.join(project_root_path, 'cases/data/arctic/nemo_multi_ts.csv'))
    time_series = df['61_91'].to_numpy()
    result = time_series * norm
    result = result[:predict_data.shape[0]]
    return result, 'ts'


def prepare_time_series(time_series, forecast_length):
    task = Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(forecast_length=forecast_length))
    data = InputData(idx=range(len(time_series)),
                     features=time_series,
                     target=time_series,
                     task=task,
                     data_type=DataTypesEnum.ts)
    train_input, predict_input = train_test_data_setup(data)
    return train_input, predict_input, task


def get_initial_pipeline():
    """
        custom -> lagged -> ridge\
                                    ridge
                  lagged -> ridge/
    """
    lagged_node1 = PrimaryNode('lagged')
    ridge_node1 = SecondaryNode('ridge', nodes_from=[lagged_node1])

    custom_node = PrimaryNode('custom')
    custom_node.custom_params = {"norm": 0.3, 'model_predict': nemo_domain_model}

    lagged_node2 = SecondaryNode('lagged', nodes_from=[custom_node])
    ridge_node2 = SecondaryNode('ridge', nodes_from=[lagged_node2])

    node_final = SecondaryNode('ridge', nodes_from=[ridge_node1, ridge_node2])

    pipeline = Pipeline(node_final)
    return pipeline


def get_available_operations():
    """ Function returns available operations for primary and secondary nodes """
    primary_operations = ['lagged', 'smoothing', 'diff_filter', 'gaussian_filter', 'custom']
    secondary_operations = ['lagged', 'ridge', 'lasso', 'linear']
    return primary_operations, secondary_operations


def compose_pipeline(pipeline, train_data, task, custom_params):
    # pipeline structure optimization
    primary_operations, secondary_operations = get_available_operations()
    composer_requirements = PipelineComposerRequirements(
        primary=primary_operations,
        secondary=secondary_operations, max_arity=3,
        max_depth=5, pop_size=10, num_of_generations=30,
        crossover_prob=0.8, mutation_prob=0.8,
        timeout=datetime.timedelta(minutes=10))
    mutation_types = [parameter_change_mutation,
                      MutationTypesEnum.single_change,
                      MutationTypesEnum.single_drop,
                      MutationTypesEnum.single_add]
    optimiser_parameters = GPGraphOptimiserParameters(mutation_types=mutation_types)
    metric_function = MetricsRepository().metric_by_id(RegressionMetricsEnum.MAE)
    builder = ComposerBuilder(task=task). \
        with_optimiser(parameters=optimiser_parameters). \
        with_requirements(composer_requirements). \
        with_metrics(metric_function).with_initial_pipelines([pipeline])
    composer = builder.build()

    DefaultOperationParamsRepository.add_model_to_repository({'custom': custom_params})

    obtained_pipeline = composer.compose_pipeline(data=train_data)
    obtained_pipeline.show()
    return obtained_pipeline


def tune_pipeline(obtained_pipeline, train_data, task):
    custom_search_space = {'custom': {'norm': (hp.uniform, [-1, 1]),
                                      'model_predict': (hp.choice, [[nemo_domain_model]])}}
    pipeline_tuner = PipelineTuner(pipeline=obtained_pipeline,
                                   task=task,
                                   iterations=500,
                                   search_space=SearchSpace(custom_search_space=custom_search_space,
                                                            replace_default_search_space=True))
    # Tuning pipeline
    pipeline = pipeline_tuner.tune_pipeline(input_data=train_data,
                                            loss_function=mean_absolute_error,
                                            cv_folds=2)
    pipeline.print_structure()
    return pipeline


def run_hybrid_modeling(forecast_length):
    project_root_path = str(fedot_project_root())
    df = pd.read_csv(os.path.join(project_root_path, 'cases/data/arctic/topaz_multi_ts.csv'))
    time_series = df['61_91'].to_numpy()
    train_input, predict_input, task = prepare_time_series(time_series, forecast_length)
    initial_pipeline = get_initial_pipeline()

    initial_pipeline.fit(train_input)
    prediction_before = np.ravel(np.array(initial_pipeline.predict(predict_input).predict))
    rmse, mae = calculate_metrics(np.ravel(predict_input.target), prediction_before)

    custom_params = {"norm": 0.3, 'model_predict': nemo_domain_model}
    pipeline = compose_pipeline(initial_pipeline, train_input, task, custom_params)
    pipeline.fit_from_scratch(train_input)
    prediction_after = pipeline.predict(predict_input)
    predict_after = np.ravel(np.array(prediction_after.predict))
    rmse_composing, mae_composing = calculate_metrics(np.ravel(predict_input.target), predict_after)

    tuned_pipeline = tune_pipeline(pipeline, train_input, task)
    tuned_pipeline.fit_from_scratch(train_input)
    prediction_after_tuning = tuned_pipeline.predict(predict_input)
    predict_after_tuning = np.ravel(np.array(prediction_after_tuning.predict))
    rmse_tuning, mae_tuning = calculate_metrics(np.ravel(predict_input.target), predict_after_tuning)

    plt.plot(np.ravel(predict_input.idx), np.ravel(predict_input.target), label='test')
    plt.plot(np.ravel(train_input.idx), np.ravel(train_input.target), label='history')
    plt.plot(np.ravel(predict_input.idx), prediction_before, label='prediction')
    plt.plot(np.ravel(predict_input.idx), predict_after, label='prediction_after_composing')
    plt.plot(np.ravel(predict_input.idx), predict_after_tuning, label='prediction_after_tuning')
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
    run_hybrid_modeling(forecast_length=60)
