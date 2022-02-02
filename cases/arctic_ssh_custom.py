import os
import datetime
from typing import Any, List

from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

from examples.simple.time_series_forecasting.ts_pipelines import *
from fedot.core.composer.composer_builder import ComposerBuilder
from fedot.core.composer.gp_composer.gp_composer import PipelineComposerRequirements
from fedot.core.composer.gp_composer.specific_operators import parameter_change_mutation
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.optimisers.gp_comp.gp_optimiser import GPGraphOptimiserParameters
from fedot.core.optimisers.gp_comp.operators.mutation import MutationTypesEnum
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.quality_metrics_repository import \
    MetricsRepository, RegressionMetricsEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.core.pipelines.tuning.unified import PipelineTuner
from fedot.core.utils import fedot_project_root


def ts_complex_ridge_pipeline():
    """
    Return pipeline with the following structure:
    lagged - ridge \
                    -> ridge -> final forecast
    lagged - ridge /
    """
    node_lagged_1 = PrimaryNode("lagged")
    node_lagged_2 = PrimaryNode("lagged")

    node_ridge_1 = SecondaryNode("ridge", nodes_from=[node_lagged_1])
    node_ridge_2 = SecondaryNode("ridge", nodes_from=[node_lagged_2])

    node_final = SecondaryNode("ridge", nodes_from=[node_ridge_1, node_ridge_2])
    pipeline = Pipeline(node_final)

    return pipeline


def get_available_operations():
    """ Function returns available operations for primary and secondary nodes """
    primary_operations = ['lagged', 'smoothing', 'gaussian_filter', 'ar']
    secondary_operations = ['lagged', 'ridge', 'lasso', 'knnreg', 'linear',
                            'scaling', 'rfe_lin_reg']
    return primary_operations, secondary_operations

pipeline = ts_complex_ridge_pipeline()

time_series = pd.read_csv('data/arctic/56_56_topaz.csv')
#time_series = pd.read_csv(f'{fedot_project_root()}/examples/data/ts/australia.csv')
task = Task(TaskTypesEnum.ts_forecasting,
            TsForecastingParams(forecast_length=90))

time_series = time_series['ssh'].values
#time_series = time_series['value'].values
idx = np.arange(len(time_series))
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

rmse = mean_squared_error(test_target, predict, squared=False)
mae = mean_absolute_error(test_target, predict)

metrics_info['Metrics without composing'] = {'RMSE': round(rmse, 3),
                                             'MAE': round(mae, 3)}

primary_operations, secondary_operations = get_available_operations()
composer_requirements = PipelineComposerRequirements(
    primary=primary_operations,
    secondary=secondary_operations, max_arity=3,
    max_depth=8, pop_size=10, num_of_generations=30,
    crossover_prob=0.8, mutation_prob=0.8,
    timeout=datetime.timedelta(minutes=10),
    validation_blocks=3)

mutation_types = [parameter_change_mutation, MutationTypesEnum.growth, MutationTypesEnum.reduce,
                  MutationTypesEnum.simple]
optimiser_parameters = GPGraphOptimiserParameters(mutation_types=mutation_types)

metric_function = MetricsRepository().metric_by_id(RegressionMetricsEnum.RMSE)
builder = ComposerBuilder(task=task). \
    with_optimiser(parameters=optimiser_parameters). \
    with_requirements(composer_requirements). \
    with_metrics(metric_function).with_initial_pipelines([pipeline])
composer = builder.build()

obtained_pipeline = composer.compose_pipeline(data=train_data, is_visualise=True)
obtained_pipeline.fit_from_scratch(train_data)
prediction_after = obtained_pipeline.predict(test_data)
predict_after = np.ravel(np.array(prediction_after.predict))

rmse = mean_squared_error(test_target, predict_after, squared=False)
mae = mean_absolute_error(test_target, predict_after)

metrics_info['Metrics after composing'] = {'RMSE': round(rmse, 3),
                                           'MAE': round(mae, 3)}

print(metrics_info)

# structure of obtained pipeline
obtained_pipeline.print_structure()
obtained_pipeline.show()

pipeline_tuner = PipelineTuner(pipeline=obtained_pipeline,
                               task=train_data.task,
                               iterations=100)
# Tuning pipeline
obtained_pipeline = pipeline_tuner.tune_pipeline(input_data=train_data,
                                                 loss_function=mean_squared_error,
                                                 loss_params={'squared': False},
                                                 cv_folds=3,
                                                 validation_blocks=3)
# Fit pipeline on the entire train data
obtained_pipeline.fit_from_scratch(train_data)
# Predict tuned pipeline
predicted_values = obtained_pipeline.predict(test_data).predict
obtained_pipeline.print_structure()

rmse = mean_squared_error(test_target, predicted_values[0], squared=False)
mae = mean_absolute_error(test_target, predicted_values[0])
print(metrics_info)
print(f'\nRMSE: {rmse}')
print(f'MAE: {mae}')


plt.plot(prediction_after.idx, predict, label='Before composing')
plt.plot(prediction_after.idx, predict_after, label='Before tuning')
plt.plot(prediction_after.idx, predicted_values[0], label='After tuning')
plt.plot(prediction_after.idx, test_data.target, label='Real')
plt.legend()
plt.show()

#obtained_pipeline.save('')
