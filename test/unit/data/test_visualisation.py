from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np

from fedot.core.data.data import InputData, OutputData
from fedot.core.data.visualisation import plot_forecast
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams


def test_single_step_forecast_is_visible():
    task = Task(
        TaskTypesEnum.ts_forecasting,
        TsForecastingParams(forecast_length=1),
    )
    data = InputData(
        idx=np.arange(4),
        features=np.array([1.0, 2.0, 3.0, 4.0]),
        target=None,
        task=task,
        data_type=DataTypesEnum.ts,
    )
    prediction = OutputData(
        idx=np.array([4]),
        predict=np.array([5.0]),
        task=task,
        data_type=DataTypesEnum.ts,
    )

    with patch("matplotlib.pyplot.show"):
        plot_forecast(data, prediction)

    predicted_line = next(
        line for line in plt.gca().get_lines() if line.get_label() == "Predicted"
    )
    assert predicted_line.get_marker() == "o"
    plt.close()
