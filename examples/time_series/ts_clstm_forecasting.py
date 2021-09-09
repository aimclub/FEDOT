import os

from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.ts_wrappers import out_of_sample_ts_forecast
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.core.utils import fedot_project_root

import pandas as pd
import numpy as np

def get_ts_data_long(n_steps=80, forecast_length=5):
    """ Prepare data from csv file with time series and take needed number of
    elements

    :param n_steps: number of elements in time series to take
    :param forecast_length: the length of forecast
    """
    project_root_path = str(fedot_project_root())
    file_path = os.path.join(project_root_path, 'examples/data/ts_long.csv')
    df = pd.read_csv(file_path)
    df = df[df["series_id"] == "traffic_volume"]
    time_series = np.array(df['value'])[:n_steps]
    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=forecast_length))

    data = InputData(idx=np.arange(0, len(time_series)),
                     features=time_series,
                     target=time_series,
                     task=task,
                     data_type=DataTypesEnum.ts)
    return train_test_data_setup(data)



def clstm_forecasting():
    horizon = 12
    window_size = 20
    n_steps = 225
    train_data, test_data = get_ts_data_long(n_steps=n_steps, forecast_length=1)
    train_data2, test_data2 = get_ts_data_long(n_steps=n_steps-horizon, forecast_length=12)

    node_lagged = PrimaryNode('lagged')
    node_lagged.custom_params = {'window_size': window_size}
    node_scaler = SecondaryNode("scaling", nodes_from=[node_lagged])
    node_root = SecondaryNode("clstm", nodes_from=[node_scaler])
    node_root.custom_params = {
        "input_size": 1,
        "forecast_length": 1,
        "hidden_size": 200,
        "learning_rate": 0.005,
        "cnn1_kernel_size": 5,
        "cnn1_output_size": 8,
        "cnn2_kernel_size": 3,
        "cnn2_output_size": 16,
        "num_epochs": 700
    }

    print(train_data.features.shape)
    pipeline = Pipeline(node_root)
    pipeline.fit(train_data)

    predicted = out_of_sample_ts_forecast(
        pipeline=pipeline,
        input_data=test_data,
        horizon=12
    )
    print(predicted)
    print(test_data2.target)
    import matplotlib.pyplot as plt
    plt.plot(predicted, label="predicted")
    plt.plot(test_data2.target, label="observed")
    plt.legend()
    plt.show()

    assert len(predicted) == horizon





if __name__ == '__main__':
    clstm_forecasting()