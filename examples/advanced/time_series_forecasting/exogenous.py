import os
import warnings

import numpy as np
import pandas as pd

from fedot.api.main import Fedot
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.core.utils import fedot_project_root

warnings.filterwarnings('ignore')
np.random.seed(2020)


def run_exogenous_experiment(path_to_file, len_forecast=250, with_exog=True, visualization=False) -> np.array:
    """ Function with example how time series forecasting can be made with using
    exogenous features

    :param path_to_file: path to the csv file with dataframe
    :param len_forecast: forecast length
    :param with_exog: is it needed to make prediction with exogenous time series
    :param visualization: is it needed to make visualizations
    """

    # Read the file
    df = pd.read_csv(path_to_file)
    time_series = np.array(df['Level'])
    exog_variable = np.array(df['Neighboring level'])

    task = Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(forecast_length=len_forecast))
    validation_blocks = 2

    # Target time series for lagged transformation
    train_lagged, predict_lagged = train_test_data_setup(InputData(idx=np.arange(len(time_series)),
                                                                   features=time_series,
                                                                   target=time_series,
                                                                   task=task,
                                                                   data_type=DataTypesEnum.ts),
                                                         validation_blocks=validation_blocks)

    # Exogenous time series
    train_exog, predict_exog = train_test_data_setup(InputData(idx=np.arange(len(exog_variable)),
                                                               features=exog_variable,
                                                               target=time_series,
                                                               task=task,
                                                               data_type=DataTypesEnum.ts),
                                                     validation_blocks=validation_blocks)

    if with_exog:
        train_dataset = MultiModalData({
            'lagged': train_lagged,
            'exog_ts': train_exog
        })

        predict_dataset = MultiModalData({
            'lagged': predict_lagged,
            'exog_ts': predict_exog
        })

        # Create a pipeline with different data sources in th nodes
        pipeline = PipelineBuilder().add_node('lagged', 0).add_node('exog_ts', 1).join_branches('ridge').build()
    else:
        train_dataset = train_lagged
        predict_dataset = predict_lagged

        # Simple example without exogenous time series
        pipeline = PipelineBuilder().add_sequence('lagged', 'ridge').build()

    # Fit it
    fedot = Fedot(problem='ts_forecasting',
                  task_params=task.task_params,
                  timeout=10,
                  initial_assumption=pipeline,
                  available_operations=['lagged', 'ridge', 'exog_ts'],
                  max_pipeline_fit_time=2,
                  n_jobs=-1)
    fedot.fit(train_dataset)

    # Predict
    predicted = fedot.predict(predict_dataset, validation_blocks=validation_blocks)
    print(fedot.get_metrics(metric_names='mae', validation_blocks=validation_blocks))

    if visualization:
        fedot.current_pipeline.show()
        # Plot predictions and true values
        fedot.plot_prediction(target='lagged')

    return predicted


if __name__ == '__main__':
    data_path = os.path.join(f'{fedot_project_root()}', 'examples/data/ts', 'ts_sea_level.csv')
    run_exogenous_experiment(path_to_file=data_path, len_forecast=250, with_exog=True, visualization=True)
