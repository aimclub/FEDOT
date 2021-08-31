import sys
import traceback

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris

from fedot.api.main import Fedot
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.core.pipelines.ts_wrappers import in_sample_ts_forecast
from fedot.core.repository.tasks import TsForecastingParams
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from sklearn.linear_model import Lasso, Ridge


def api_example():
    df_el, _ = load_iris(return_X_y=True, as_frame=True)
    features = np.ravel(np.array(df_el['sepal length (cm)']))
    target = features

    auto_model = Fedot(problem='ts_forecasting',
                       timeout=0.05,
                       task_params=TsForecastingParams(forecast_length=14),
                       composer_params={'metric': 'rmse'},
                       verbose_level=0)
    pipeline = auto_model.fit(features=features, target=target)
    pipeline.print_structure()
    max_lag = round(
        max([np.ceil(n.custom_params['window_size']) for n in pipeline.nodes if str(n.operation) == 'lagged']))
    print(max_lag)


if __name__ == '__main__':
    api_example()
