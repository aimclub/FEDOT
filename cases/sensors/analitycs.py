import os
import pandas as pd
import numpy as np
from typing import List
from sklearn.metrics import r2_score, mean_absolute_percentage_error, \
    mean_absolute_error

from cases.sensors.launch_tools import lstm_ref, lstm_non_ref, arima_ref, \
    arima_non_ref, ridge_ref, ridge_non_ref
from fedot.core.data.data import InputData
from fedot.core.pipelines.ts_wrappers import fitted_values, in_sample_fitted_values
import matplotlib.pyplot as plt

from fedot.core.repository.tasks import TaskTypesEnum, Task, TsForecastingParams


def vis_predictions(path, dataset_number: int = 0, horizon: int = 10):
    """ Function plot predictions """

    # Get folders
    folders = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    for folder in folders:
        print(f'Processing folder {folder}, dataset number {dataset_number}')
        current_folder = os.path.join(path, folder, str(dataset_number))

        # Find appropriate files
        files = os.listdir(current_folder)
        for file in files:
            if ''.join((str(horizon), '_')) in file:
                name = os.path.join(current_folder, file)

                splitted = file.split('_')
                model_name = splitted[3]
                df = pd.read_csv(name)

                thr = 350 + 3 * horizon
                plt.plot(df['decomposed'][-thr:], label='With decomposition')
                plt.plot(df['non_decompose'][-thr:], label='Without decomposition')
                plt.plot(df['actual'][-thr:], label='Actual time series')
                for i in [1, 2, 3]:
                    forecast_point = len(df) - (i * horizon) - 1
                    plt.plot([forecast_point, forecast_point], [min(df['actual']), max(df['actual'])],
                             c='black', alpha=0.5)
                plt.grid()
                plt.title(f'Model - {model_name}')
                plt.legend()
                plt.show()


def calculate_metrics(path, dataset_numbers: List[int], horizons: List[int],
                      save_file: str = None):
    """ Function calculate metrics on dataset """
    # Get folders
    folders = [d for d in os.listdir(path) if
               os.path.isdir(os.path.join(path, d))]

    r2s_decomposed = []
    r2s_non_decomposed = []
    mapes_decomposed = []
    mapes_non_decomposed = []
    store_horizons = []
    models = []
    for folder in ['results']:
        for dataset_number in dataset_numbers:
            current_folder = os.path.join(path, folder, str(dataset_number))

            # Find appropriate files
            files = os.listdir(current_folder)
            for file in files:
                for horizon in horizons:
                    if ''.join((str(horizon), '_')) in file:
                        name = os.path.join(current_folder, file)

                        splitted = file.split('_')
                        model_name = splitted[3]
                        if model_name == 'ridge':
                            # Read file
                            df = pd.read_csv(name)

                            # Take 3 last validation blocks values
                            thr = 3 * horizon
                            actuals = np.array(df['actual'][-thr:])
                            decomposed_preds = np.array(df['decomposed'][-thr:])
                            non_decomposed_preds = np.array(df['non_decompose'][-thr:])

                            # Calculate metrics
                            decomposed_mape = mean_absolute_percentage_error(actuals,
                                                                             decomposed_preds)
                            non_decomposed_mape = mean_absolute_percentage_error(actuals,
                                                                                 non_decomposed_preds)

                            decomposed_r2 = r2_score(actuals, decomposed_preds)
                            non_decomposed_r2 = r2_score(actuals, non_decomposed_preds)

                            r2s_decomposed.append(decomposed_r2)
                            r2s_non_decomposed.append(non_decomposed_r2)
                            mapes_decomposed.append(decomposed_mape)
                            mapes_non_decomposed.append(non_decomposed_mape)
                            store_horizons.append(horizon)
                            models.append(model_name)
    dataframe = pd.DataFrame({'models': models, 'horizon': store_horizons,
                              'mapes_decomposed': mapes_decomposed,
                              'mapes_non_decomposed': mapes_non_decomposed,
                              'r2s_decomposed': r2s_decomposed,
                              'r2s_non_decomposed': r2s_non_decomposed})

    mapes_decomposed = dataframe["mapes_decomposed"].mean()
    mapes_non_decomposed = dataframe["mapes_non_decomposed"].mean()
    if mapes_decomposed <= mapes_non_decomposed:
        sign = '+'
    else:
        sign = '-'
    print(f'Dataset {dataset_numbers[0]}, {mapes_decomposed:.3f} vs {mapes_non_decomposed:.3f} {sign}')
    if save_file is not None:
        dataframe.to_csv(save_file, index=False)


def show_fitted_ts(dataset_number: int = 0, horizon: int = 10, model: str = 'ridge'):
    if model == 'ridge':
        non_decompose = ridge_non_ref()
        decompose = ridge_ref()
    elif model == 'arima':
        non_decompose = arima_non_ref()
        decompose = arima_ref()
    else:
        raise ValueError('Such model doesnt exist')

    path = '../../cases/data/time_series'
    # Get folders
    folders = [d for d in os.listdir(path) if
               os.path.isdir(os.path.join(path, d))]
    for folder in folders:
        print(f'Processing folder {folder}, dataset number {dataset_number}')
        current_folder = os.path.join(path, folder, str(dataset_number))
        # Find appropriate files
        files = os.listdir(current_folder)
        for file in files:
            if ''.join((str(horizon), '_')) in file and model in file:
                name = os.path.join(current_folder, file)
                task = Task(TaskTypesEnum.ts_forecasting,
                            TsForecastingParams(forecast_length=horizon))
                ts_input = InputData.from_csv_time_series(file_path=name,
                                                          task=task,
                                                          target_column='actual')

                non_decompose = non_decompose.fine_tune_all_nodes(loss_function=mean_absolute_error,
                                                                  input_data=ts_input,
                                                                  iterations=10,
                                                                  timeout=2)

                decompose = decompose.fine_tune_all_nodes(loss_function=mean_absolute_error,
                                                          input_data=ts_input,
                                                          iterations=10,
                                                          timeout=2)

                non_decompose_trained = non_decompose.fit(ts_input)
                decompose_trained = decompose.fit(ts_input)

                fitted_non_decompose = fitted_values(non_decompose_trained, 10)
                fitted_decompose = fitted_values(decompose_trained, 10)

                plt.plot(ts_input.idx, ts_input.target,
                         label='Actual time series', alpha=0.8)
                plt.plot(fitted_non_decompose.idx, fitted_non_decompose.predict,
                         label='Non decomposed', alpha=0.6)
                plt.plot(fitted_decompose.idx, fitted_decompose.predict,
                         label='Decomposed', alpha=0.6)
                plt.legend()
                plt.grid()
                plt.show()

# for i in range(0, 41):
#     calculate_metrics('../../cases/data/time_series',
#                       dataset_numbers=[i],
#                       horizons=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
#                       save_file=None)
# show_fitted_ts(dataset_number=9, horizon=90, model='arima')
vis_predictions('../../cases/data/time_series', dataset_number=35, horizon=100)
