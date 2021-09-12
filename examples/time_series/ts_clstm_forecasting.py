import json
import os

from sklearn.metrics import mean_absolute_error

from examples.pipeline_import_export import create_correct_path, get_pipeline
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.ts_wrappers import out_of_sample_ts_forecast
from fedot.core.pipelines.tuning.unified import PipelineTuner
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
    return train_test_data_setup(data), task


def clstm_forecasting():
    horizon = 24*2
    window_size = 24*7
    n_steps = 200
    (train_data, test_data), task = get_ts_data_long(n_steps=n_steps+1, forecast_length=1)
    (train_data2, test_data2), _ = get_ts_data_long(n_steps=n_steps+horizon, forecast_length=horizon)

    node_root = PrimaryNode("clstm")
    node_root.custom_params = {
        'input_size': 1,
        'forecast_length': 1,
        'window_size': 29.3418382487487,
        'hidden_size': 135.66211398383396,
        'learning_rate': 0.004641403016307329,
        'cnn1_kernel_size': 5,
        'cnn1_output_size': 32,
        'cnn2_kernel_size': 4,
        'cnn2_output_size': 32,
        'batch_size': 64,
        'num_epochs': 50
    }

    pipeline = Pipeline(node_root)

    # pipeline_tuner = PipelineTuner(pipeline=pipeline, task=task,
    #                                iterations=30)
    #
    # pipeline = pipeline_tuner.tune_pipeline(input_data=train_data,
    #                                         loss_function=mean_absolute_error,
    #                                         cv_folds=3,
    #                                         validation_blocks=2)
    # pipeline.print_structure()

    pipeline.fit(train_data)

    prediction_before_export = out_of_sample_ts_forecast(
        pipeline=pipeline,
        input_data=test_data,
        horizon=horizon
    )

    print(f'Before export {prediction_before_export[:4]}')

    path = "import_export"
    # Export it
    pipeline.save(path=path)

    # Import pipeline
    json_path_load = create_correct_path(path)
    print(json_path_load)
    new_pipeline = Pipeline()
    new_pipeline.load(json_path_load)

    predicted_output_after_export = out_of_sample_ts_forecast(
        pipeline=new_pipeline,
        input_data=test_data,
        horizon=horizon
    )

    print(f'After import {predicted_output_after_export[:4]}')

    dict_pipeline, dict_fitted_operations = pipeline.save()
    dict_pipeline = json.loads(dict_pipeline)
    pipeline_from_dict = Pipeline()
    pipeline_from_dict.load(dict_pipeline, dict_fitted_operations)

    predicted_output_from_dict = out_of_sample_ts_forecast(
        pipeline=pipeline_from_dict,
        input_data=test_data,
        horizon=horizon
    )
    print(f'Prediction from pipeline loaded from dict {predicted_output_from_dict[:4]}')

    import matplotlib.pyplot as plt
    plt.plot(np.arange(n_steps, n_steps+horizon), predicted_output_from_dict, label="predicted")
    plt.plot(np.arange(n_steps, n_steps+horizon), test_data2.target, label="observed")
    plt.plot(np.arange(n_steps-window_size, n_steps - 1), test_data.features[n_steps-window_size:n_steps - 1], label="seen")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    clstm_forecasting()
