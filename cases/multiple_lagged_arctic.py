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


def create_complex_train(points_list, forecast_length):
    complex_train = pd.DataFrame()
    for name in points_list:
        ts = pd.read_csv(f'data/arctic/{name}_topaz.csv')['ssh']
        train_ts = ts[:-forecast_length]
        complex_train[name] = train_ts
    complex_train = complex_train.to_numpy()
    return complex_train


points = ['61_91', '56_86', '61_86', '66_86', '66_91', '66_96']

forecast_length = 60
# target point
time_series = pd.read_csv('data/arctic/61_91_topaz.csv')['ssh'].values
x_test = time_series[:-forecast_length]
y_test = time_series[-forecast_length:]
y_train = time_series[:-forecast_length]

x_train = create_complex_train(points, forecast_length)

inds = np.arange(len(time_series))
idx_train = inds[:-forecast_length]
idx_train = np.tile(idx_train, (6, 1)).T
idx_test = inds[-forecast_length:]
idx_test = np.tile(idx_test, (6, 1)).T

# Prepare data to train the operation

task = Task(TaskTypesEnum.ts_forecasting,
            TsForecastingParams(forecast_length=forecast_length))
train_data = InputData(idx=idx_train, features=x_train, target=x_train,
                       task=task, data_type=DataTypesEnum.multi_ts)
test_data = InputData(idx=idx_test, features=x_test, target=y_test,
                      task=task, data_type=DataTypesEnum.multi_ts)

node_lagged_1 = PrimaryNode("lagged")
node_final = SecondaryNode("ridge", nodes_from=[node_lagged_1])
pipeline = Pipeline(node_final)

pipeline.fit(train_data)

prediction = pipeline.predict(test_data)
predict = np.ravel(np.array(prediction.predict))

plt.plot(np.ravel(test_data.idx[:, 0]), test_data.target, label='test')
plt.plot(np.ravel(train_data.idx[:, 0]), np.ravel(train_data.target[:, 0]), label='history')
plt.plot(np.ravel(test_data.idx[:, 0]), predict, label='prediction')
plt.legend()
plt.show()
