import datetime
import sys

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy import stats
from sklearn.cluster import KMeans

from fedot.core.composer.composer_builder import ComposerBuilder
from fedot.core.composer.gp_composer.specific_operators import parameter_change_mutation
from fedot.core.composer.metrics import QualityMetric
from fedot.core.pipelines.pipeline_composer_requirements import PipelineComposerRequirements
from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder
from fedot.core.repository.quality_metrics_repository import RegressionMetricsEnum
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.base_mutations import MutationTypesEnum
from golem.core.tuning.simultaneous import SimultaneousTuner
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from fedot.core.data.data import InputData, OutputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TsForecastingParams, TaskTypesEnum
import warnings
warnings.filterwarnings("ignore")


def calc_ice_phases(ts, dates, seasons_threshold):
    ice_start_days = []
    ice_end_days = []
    flag = 'winter'
    rolling_pre_window = 5
    rolling_for_window = 5
    for i in range(len(ts) - rolling_pre_window):
        interval_pre_values = ts[i:i + rolling_pre_window]
        interval_for_values = ts[i + rolling_pre_window:i + rolling_pre_window + rolling_for_window]
        pre_mean = np.mean(interval_pre_values)
        for_mean = np.mean(interval_for_values)
        if pre_mean > seasons_threshold and for_mean <= seasons_threshold:
            if flag != 'summer':
                ice_start_days.append(dates[i])
                flag = 'summer'
        if for_mean > seasons_threshold and pre_mean <= seasons_threshold:
            if flag != 'winter':
                ice_end_days.append(dates[i])
                flag = 'winter'
    return ice_start_days, ice_end_days


def calc_threshold_value_for_point(ts):
    X = np.array(ts).reshape(-1, 1)
    kmeans = KMeans(n_clusters=2)
    color = kmeans.fit_predict(X)

    ts1 = np.array(ts)[color == 0]
    ts2 = np.array(ts)[color == 1]

    density = stats.gaussian_kde(ts1)
    x = np.linspace(ts1.min(), ts1.max(), 200)
    y = density.evaluate(x)
    indices = np.where(y == y.max())
    val1 = x[indices]

    density = stats.gaussian_kde(ts2)
    x = np.linspace(ts2.min(), ts2.max(), 200)
    y = density.evaluate(x)
    indices = np.where(y == y.max())
    val2 = x[indices]

    max_value_for_point = max(val1, val2)
    min_value_for_point = min(val1, val2)
    threshold = (max_value_for_point + min_value_for_point) / 2

    return threshold


def initial_pipeline():
    lag1 = PipelineNode('lagged')
    lag1.parameters = {'window_size': 360}
    lag2 = PipelineNode('lagged')
    r1 = PipelineNode('ridge', nodes_from=[lag1])
    r2 = PipelineNode('ridge', nodes_from=[lag2])
    r3 = PipelineNode('ridge', nodes_from=[r1, r2])
    crop_node1 = PipelineNode('crop_range', nodes_from=[r3])
    crop_node1.parameters = {'min_value': 0, 'max_value': 1}
    pipeline = Pipeline(crop_node1)
    pipeline.show()
    return pipeline


def calculate_metrics(target, predicted):
    rmse = mean_squared_error(target, predicted, squared=True)
    mae = mean_absolute_error(target, predicted)
    return rmse, mae


class PhasesErrorMetric(QualityMetric):
    default_value = sys.maxsize

    @staticmethod
    def metric(reference: InputData, predicted: OutputData) -> float:
        dates = np.arange(reference.target.shape[0])
        target_ice_start, target_ice_end = calc_ice_phases(reference.target,
                                                           dates,
                                                           calc_threshold_value_for_point(reference.target))
        predict_ice_start, predict_ice_end = calc_ice_phases(predicted.predict,
                                                            dates,
                                                            calc_threshold_value_for_point(predicted.predict))
        start_phase_error = mean_absolute_error(y_true=target_ice_start, y_pred=predict_ice_start)
        end_phase_error = mean_absolute_error(y_true=target_ice_end, y_pred=predict_ice_end)
        return abs(start_phase_error)+abs(end_phase_error)


def compose_pipeline(pipeline, train_data, task):
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
        with_metrics(PhasesErrorMetric.get_value). \
        with_initial_pipelines([pipeline]). \
        build()
    obtained_pipeline = composer.compose_pipeline(data=train_data)
    for pipeline in obtained_pipeline:
        pipeline.show()
    return obtained_pipeline


def tune_pipeline(pipeline, train_data, task):
    tuner = TunerBuilder(task) \
        .with_tuner(SimultaneousTuner) \
        .with_metric(PhasesErrorMetric.get_value) \
        .with_iterations(50) \
        .build(train_data)
    tuned_pipeline = tuner.tune(pipeline)
    tuned_pipeline.print_structure()
    return tuned_pipeline


df = pd.read_csv('../../data/ts/osisaf_ice_conc.csv')
df['date'] = pd.to_datetime(df['date'])
len_forecast = 1250
point = '110_70'
time_series = np.array(df[point])

smoothed_time_series = uniform_filter1d(time_series, size=30)
plt.plot(df['date'][-2000:], time_series[-2000:], label='real ts')
plt.plot(df['date'][-2000:], smoothed_time_series[-2000:], label='smoothed ts')
plt.legend()
plt.title(point)
plt.show()

task = Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(forecast_length=len_forecast))
train_input, predict_input = train_test_data_setup(InputData(idx=np.array(range(len(time_series))),
                                                             features=smoothed_time_series,
                                                             target=smoothed_time_series,
                                                             task=task,
                                                             data_type=DataTypesEnum.ts))

pipeline = initial_pipeline()
composed_pipelines = compose_pipeline(pipeline, train_input, task)
#composed_pipeline = pipeline
for composed_pipeline in composed_pipelines:
    tuned_pipeline = tune_pipeline(composed_pipeline, train_input, task)

    tuned_pipeline.fit_from_scratch(train_input)
    prediction = tuned_pipeline.predict(predict_input)
    prediction_values = np.ravel(np.array(prediction.predict))

    rmse_tuning, mae_tuning = calculate_metrics(np.ravel(predict_input.target), prediction_values)

    ts_smoothed = uniform_filter1d(np.ravel(predict_input.target), size=30)
    dates = np.array(df['date'])[-len_forecast:]
    threshold = calc_threshold_value_for_point(ts_smoothed)
    ice_start, ice_end = calc_ice_phases(ts_smoothed, dates, threshold)

    # predicted_smoothed = uniform_filter1d(prediction_values, size=30)
    threshold_pr = calc_threshold_value_for_point(prediction_values)
    ice_start_pr, ice_end_pr = calc_ice_phases(prediction_values, dates, threshold_pr)

    for s in ice_start:
        plt.axvline(s, c='r')
    for e in ice_end:
        plt.axvline(e, c='black')

    for s in ice_start_pr:
        plt.axvline(s, c='r')
    for e in ice_end_pr:
        plt.axvline(e, c='black')

    plt.plot(np.array(df['date'])[-len_forecast:], np.ravel(predict_input.target), label='test')
    plt.plot(np.array(df['date'])[-900 - len_forecast:-len_forecast],
             np.array(df[point])[-900 - len_forecast:-len_forecast], label='history')
    plt.plot(np.array(df['date'])[-len_forecast:], prediction_values, label='prediction_after_tuning')
    plt.xlabel('Time step')
    plt.ylabel('Ice conc')
    plt.title(point)
    plt.legend()
    plt.show()

    print(ice_start_pr)
    print(ice_start)
    print('__________')
    print(ice_end_pr)
    print(ice_end)

    print(f'RMSE after tuning: {round(rmse_tuning, 3)}')
    print(f'MAE after tuning: {round(mae_tuning, 3)}')
