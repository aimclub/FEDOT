import os
from copy import deepcopy

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TsForecastingParams, TaskTypesEnum


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return round((np.mean(np.abs((y_true - y_pred) / y_true)) * 100), 4)


def prepare_data(time_series, exog_variable, len_forecast=250):
    time_series = np.array(time_series)
    exog_variable = np.array(exog_variable)

    train_input, predict_input = train_test_data_setup(
        InputData(idx=np.arange(len(time_series)),
                  features=time_series,
                  target=time_series,
                  task=Task(TaskTypesEnum.ts_forecasting,
                            TsForecastingParams(
                                forecast_length=len_forecast)),
                  data_type=DataTypesEnum.ts))

    # Exogenous time series
    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(
                    forecast_length=len_forecast))

    predict_input_exog = InputData(idx=np.arange(len(exog_variable)),
                                   features=exog_variable, target=time_series,
                                   task=task, data_type=DataTypesEnum.ts)
    train_input_exog, predict_input_exog = train_test_data_setup(predict_input_exog)

    return train_input, predict_input, train_input_exog, predict_input_exog, predict_input.target


def get_arima_nemo_pipeline():
    """ Function return complex pipeline with the following structure
        arima \
               linear
        nemo  |
    """

    node_arima = PrimaryNode('arima')
    node_nemo = PrimaryNode('exog_ts')
    node_final = SecondaryNode('linear', nodes_from=[node_arima, node_nemo])
    pipeline = Pipeline(node_final)
    return pipeline


def get_stlarima_nemo_pipeline():
    """ Function return complex pipeline with the following structure
        stl_arima \
                   linear
            nemo  |
    """

    node_arima = PrimaryNode('stl_arima')
    node_arima.parameters = {'period': 80, 'p': 2, 'd': 1, 'q': 0}
    node_nemo = PrimaryNode('exog_ts')
    node_final = SecondaryNode('linear', nodes_from=[node_arima, node_nemo])
    pipeline = Pipeline(node_final)
    return pipeline


def get_ridge_nemo_pipeline():
    """ Function return complex pipeline with the following structure
        lagged -> ridge \
                          ridge
        lagged -> ridge  |      \
                                 linear
                          nemo  /
    """

    node_lagged_1 = PrimaryNode('lagged/1')
    node_ridge_1 = SecondaryNode('ridge', nodes_from=[node_lagged_1])
    node_lagged_2 = PrimaryNode('lagged/2')
    node_ridge_2 = SecondaryNode('ridge', nodes_from=[node_lagged_2])
    node_ridge_3 = SecondaryNode('ridge', nodes_from=[node_ridge_1, node_ridge_2])
    node_nemo = PrimaryNode('exog_ts')
    node_final = SecondaryNode('linear', nodes_from=[node_ridge_3, node_nemo])
    pipeline = Pipeline(node_final)
    return pipeline


def get_arima_pipeline():
    """ Function return complex pipeline with the following structure
        arima
    """

    node_final = PrimaryNode('arima')

    pipeline = Pipeline(node_final)
    return pipeline


def get_stlarima_pipeline():
    """ Function return complex pipeline with the following structure
        stl_arima
    """

    node_final = PrimaryNode('stl_arima')
    node_final.parameters = {'period': 80, 'p': 2, 'd': 1, 'q': 0}
    pipeline = Pipeline(node_final)
    return pipeline


def get_ridge_pipeline():
    """ Function return complex pipeline with the following structure
        lagged -> ridge \
                          ridge
        lagged -> ridge  |
    """

    node_lagged_1 = PrimaryNode('lagged/1')
    node_ridge_1 = SecondaryNode('ridge', nodes_from=[node_lagged_1])

    node_lagged_2 = PrimaryNode('lagged/2')

    node_ridge_2 = SecondaryNode('ridge', nodes_from=[node_lagged_2])

    node_final = SecondaryNode('ridge', nodes_from=[node_ridge_1, node_ridge_2])
    pipeline = Pipeline(node_final)
    return pipeline


def compare_plot(predicted, real, forecast_length, model):
    plt.plot(real, linewidth=1, label="Observed", alpha=0.4)
    plt.plot(predicted, linewidth=1, label="Predicted", alpha=0.6)
    plt.legend()
    plt.xlabel('Time index')
    plt.ylabel('SSH, m')
    plt.title(f'Sea surface height forecast for {forecast_length} days with {model}')
    plt.show()


def run_nemo_based_forecasting(time_series, exog_variable, len_forecast=60, visualization=False):
    errors_df = {}

    train_input, predict_input, train_input_exog, predict_input_exog, test_data = \
        prepare_data(time_series=time_series,
                     exog_variable=exog_variable,
                     len_forecast=len_forecast)

    pipelines = {'ARIMA': {
        'tr_nodes_data': {"arima": train_input},
        'pr_nodes_data': {"arima": predict_input},
        'model': get_arima_pipeline()
    },
        'STL_ARIMA': {
            'tr_nodes_data': {"stl_arima": train_input},
            'pr_nodes_data': {"stl_arima": predict_input},
            'model': get_stlarima_pipeline()
        },
        'RIDGE':
            {'tr_nodes_data': {"lagged/1": train_input, "lagged/2": train_input},
             'pr_nodes_data': {"lagged/1": predict_input, "lagged/2": predict_input},
             'model': get_ridge_pipeline()
             },
        'ARIMA_NEMO':
            {'tr_nodes_data': {"arima": train_input, "exog_ts": train_input_exog},
             'pr_nodes_data': {"arima": predict_input,
                               "exog_ts": predict_input_exog},
             'model': get_arima_nemo_pipeline()
             },
        'STL_ARIMA_NEMO':
            {'tr_nodes_data': {"stl_arima": train_input,
                               "exog_ts": train_input_exog},
             'pr_nodes_data': {"stl_arima": predict_input,
                               "exog_ts": predict_input_exog},
             'model': get_stlarima_nemo_pipeline()
             },
        'RIDGE_NEMO':
            {'tr_nodes_data': {"lagged/1": train_input,
                               "lagged/2": train_input,
                               "exog_ts": train_input_exog},
             'pr_nodes_data': {"lagged/1": predict_input,
                               "lagged/2": predict_input,
                               "exog_ts": predict_input_exog},
             'model': get_ridge_nemo_pipeline()
             }
    }

    def multimodal_data_preparing(name):
        pipeline = pipelines[name]['model']
        train_dict = {}
        predict_dict = {}
        for key in pipelines[name]['tr_nodes_data'].keys():
            train_dict[key] = deepcopy(pipelines[name]['tr_nodes_data'][key])
        for key in pipelines[name]['pr_nodes_data'].keys():
            predict_dict[key] = deepcopy(pipelines[name]['pr_nodes_data'][key])

        train_dataset = MultiModalData(train_dict)
        predict_dataset = MultiModalData(predict_dict)
        return pipeline, train_dataset, predict_dataset

    for model_name in pipelines.keys():
        pipeline, train_dataset, predict_dataset = multimodal_data_preparing(model_name)

        pipeline.fit_from_scratch(train_dataset)
        predicted_values = pipeline.predict(predict_dataset)
        predicted_values = predicted_values.predict
        predicted = np.ravel(np.array(predicted_values))
        test_data = np.ravel(test_data)

        mse_before = mean_squared_error(test_data, predicted, squared=False)
        mae_before = mean_absolute_error(test_data, predicted)
        mape_before = mean_absolute_percentage_error(test_data, predicted)

        errors_df[model_name + '_MSE'] = mse_before
        errors_df[model_name + '_MAE'] = mae_before
        errors_df[model_name + '_MAPE'] = mape_before

        if visualization:
            compare_plot(predicted, test_data, len_forecast, model_name)
            print(model_name)
            print(f' MSE - {mse_before:.4f}')
            print(f' MAE - {mae_before:.4f}')
            print(f' MAPE - {mape_before:.4f}\n')

    return errors_df


def boxplot_visualize(df, label):
    ylabel = 'ssh, meters'
    if label == 'MAPE':
        ylabel = ''

    df.boxplot(rot=13)
    plt.ylabel(ylabel)
    plt.title(f'{label} distribution')

    plt.show()


def run_single_example(len_forecast=40, visualization=False):
    ts_name = 'sea_level'
    path_to_file = '../../cases/data/nemo/sea_surface_height.csv'
    path_to_exog_file = '../../cases/data/nemo/sea_surface_height_nemo.csv'

    df = pd.read_csv(path_to_file)
    time_series = df[ts_name]
    df = pd.read_csv(path_to_exog_file)
    exog_variable = df[ts_name]

    run_nemo_based_forecasting(time_series=time_series,
                               exog_variable=exog_variable,
                               len_forecast=len_forecast,
                               visualization=visualization)


def create_errors_df():
    df = pd.DataFrame(columns=['POINT', 'RIDGE', 'RIDGE_NEMO',
                               'ARIMA', 'ARIMA_NEMO',
                               'STL_ARIMA', 'STL_ARIMA_NEMO'])
    return df


def add_data_to_errors_df(df, error_name, point, errors):
    df = df.append({'POINT': point,
                    'RIDGE': errors['RIDGE_' + error_name],
                    'RIDGE_NEMO': errors['RIDGE_NEMO_' + error_name],
                    'ARIMA': errors['ARIMA_' + error_name],
                    'ARIMA_NEMO': errors['ARIMA_NEMO_' + error_name],
                    'STL_ARIMA': errors['STL_ARIMA_' + error_name],
                    'STL_ARIMA_NEMO': errors['STL_ARIMA_NEMO_' + error_name],
                    }, ignore_index=True)
    return df


def run_multiple_example(path_to_file, path_to_exog_file, out_path=None, visualization=False, len_forecast=40):
    mse_errors_df = create_errors_df()
    mae_errors_df = create_errors_df()
    mape_errors_df = create_errors_df()

    path_to_file = path_to_file
    path_to_exog_file = path_to_exog_file

    df = pd.read_csv(path_to_file)
    df_exog = pd.read_csv(path_to_exog_file)

    for point in df.columns.values:
        if point != 'dates':
            time_series = df[point]
            exog_variable = df_exog[point]

            errors = run_nemo_based_forecasting(time_series=time_series,
                                                exog_variable=exog_variable,
                                                len_forecast=len_forecast,
                                                visualization=visualization)

            mse_errors_df = add_data_to_errors_df(mse_errors_df, 'MSE', point, errors)
            mae_errors_df = add_data_to_errors_df(mae_errors_df, 'MAE', point, errors)
            mape_errors_df = add_data_to_errors_df(mape_errors_df, 'MAPE', point, errors)

    if out_path is not None:
        mse_errors_df.to_csv(os.path.join(out_path, 'mse_errors.csv'), index=False)
        mae_errors_df.to_csv(os.path.join(out_path, 'mae_errors.csv'), index=False)
        mape_errors_df.to_csv(os.path.join(out_path, 'mape_errors.csv'), index=False)

    if visualization:
        boxplot_visualize(mse_errors_df, 'MSE')
        boxplot_visualize(mae_errors_df, 'MAE')
        boxplot_visualize(mape_errors_df, 'MAPE')


def run_prediction_examples(mode='single', visualization=False):
    if mode == 'single':
        run_single_example(len_forecast=40, visualization=visualization)
    if mode == 'multiple':
        run_multiple_example(path_to_file='../../cases/data/nemo/SSH_points_grid.csv',
                             path_to_exog_file='../../cases/data/nemo/SSH_nemo_points_grid.csv',
                             out_path=None,
                             len_forecast=30,
                             visualization=visualization)


if __name__ == '__main__':
    run_prediction_examples(mode='multiple', visualization=True)
