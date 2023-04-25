import datetime
import numpy as np
import pandas as pd
from fedot.core.composer.composer_builder import ComposerBuilder
from fedot.core.composer.gp_composer.specific_operators import parameter_change_mutation
from fedot.core.pipelines.pipeline_composer_requirements import PipelineComposerRequirements
from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder
from fedot.core.repository.quality_metrics_repository import RegressionMetricsEnum
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.base_mutations import MutationTypesEnum
from golem.core.tuning.simultaneous import SimultaneousTuner
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TsForecastingParams, TaskTypesEnum


def initial_pipeline():

    lag1 = PipelineNode('lagged')
    lag1.parameters = {'window_size': 360}
    lag2 = PipelineNode('lagged')
    r1 = PipelineNode('ridge', nodes_from=[lag1])
    r2 = PipelineNode('ridge', nodes_from=[lag2])
    r3 = PipelineNode('ridge', nodes_from=[r1, r2])

    crop_node1 = PipelineNode('crop_range', nodes_from=[r3])

    node_final = PipelineNode('ridge', nodes_from=[crop_node1])
    pipeline = Pipeline(crop_node1)
    pipeline.show()
    pipeline.print_structure()
    return pipeline


def calculate_metrics(target, predicted):
    rmse = mean_squared_error(target, predicted, squared=True)
    mae = mean_absolute_error(target, predicted)
    return rmse, mae


def compose_pipeline(pipeline, train_data, task):
    # pipeline structure optimization
    composer_requirements = PipelineComposerRequirements(
        max_arity=10, max_depth=10,
        num_of_generations=30,
        timeout=datetime.timedelta(minutes=10))
    optimizer_parameters = GPAlgorithmParameters(
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


def tune_pipeline(pipeline, train_data, task):
    tuner = TunerBuilder(task) \
        .with_tuner(SimultaneousTuner) \
        .with_metric(RegressionMetricsEnum.MAE) \
        .with_iterations(100) \
        .build(train_data)
    tuned_pipeline = tuner.tune(pipeline)
    tuned_pipeline.print_structure()
    return tuned_pipeline


df = pd.read_csv('../../data/ts/osisaf_ice_conc.csv')
len_forecast = 1095
time_series = np.array(df['50_40'])

task = Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(forecast_length=len_forecast))
train_input, predict_input = train_test_data_setup(InputData(idx=range(len(time_series)),
                                                             features=time_series,
                                                             target=time_series,
                                                             task=task,
                                                             data_type=DataTypesEnum.ts))

pipeline = initial_pipeline()
composed_pipeline = compose_pipeline(pipeline, train_input, task)
tuned_pipeline = tune_pipeline(composed_pipeline, train_input, task)

pipeline.fit_from_scratch(train_input)
prediction = pipeline.predict(predict_input)
prediction_values = np.ravel(np.array(prediction.predict))

rmse_tuning, mae_tuning = calculate_metrics(np.ravel(predict_input.target), prediction_values)
plt.plot(np.ravel(predict_input.idx), np.ravel(predict_input.target), label='test')
plt.plot(np.ravel(train_input.idx)[-1300:], np.ravel(train_input.target)[-1300:], label='history')
plt.plot(np.ravel(predict_input.idx), prediction_values, label='prediction_after_tuning')
plt.xlabel('Time step')
plt.ylabel('Ice conc')
plt.legend()
plt.show()

print(f'RMSE after tuning: {round(rmse_tuning, 3)}')
print(f'MAE after tuning: {round(mae_tuning, 3)}')
