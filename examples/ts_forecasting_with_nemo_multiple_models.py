import numpy as np
import pandas as pd
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
from examples.ts_forecasting_tuning import prepare_input_data
from fedot.core.chains.chain import Chain
from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.data.multi_modal import MultiModalData


def prepare_data(time_series, exog_variable, len_forecast=250):

    time_series = np.array(time_series)
    exog_variable = np.array(exog_variable)

    # Let's divide our data on train and test samples
    train_data = time_series[:-len_forecast]
    test_data = time_series[-len_forecast:]

    # Nemo feature
    train_data_exog = exog_variable[:-len_forecast]
    test_data_exog = exog_variable[-len_forecast:]

    # Source time series
    train_input, predict_input, task = prepare_input_data(len_forecast=len_forecast,
                                                          train_data_features=train_data,
                                                          train_data_target=train_data,
                                                          test_data_features=train_data)

    # Exogenous time series
    train_input_exog, predict_input_exog, _ = prepare_input_data(len_forecast=len_forecast,
                                                                 train_data_features=train_data_exog,
                                                                 train_data_target=train_data,
                                                                 test_data_features=test_data_exog)

    return train_input, predict_input, train_input_exog, predict_input_exog, test_data


def get_arima_nemo_chain():
    """ Function return complex chain with the following structure
        arima \
               linear
        nemo  |
    """

    node_arima = PrimaryNode('arima')
    node_nemo = PrimaryNode('exog_ts_data_source')
    node_final = SecondaryNode('linear', nodes_from=[node_arima, node_nemo])
    chain = Chain(node_final)
    return chain


def get_STLarima_nemo_chain():
    """ Function return complex chain with the following structure
        stl_arima \
                   linear
            nemo  |
    """

    node_arima = PrimaryNode('stl_arima')
    node_arima.custom_params = {'period': 80, 'p': 2, 'd': 1, 'q': 0}
    node_nemo = PrimaryNode('exog_ts_data_source')
    node_final = SecondaryNode('linear', nodes_from=[node_arima, node_nemo])
    chain = Chain(node_final)
    return chain


def get_ridge_nemo_chain():
    """ Function return complex chain with the following structure
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
    node_nemo = PrimaryNode('exog_ts_data_source')
    node_final = SecondaryNode('linear', nodes_from=[node_ridge_3, node_nemo])
    chain = Chain(node_final)
    return chain


def get_arima_chain():
    """ Function return complex chain with the following structure
        arima
    """

    node_final = PrimaryNode('arima')

    chain = Chain(node_final)
    return chain


def get_STLarima_chain():
    """ Function return complex chain with the following structure
        stl_arima
    """

    node_final = PrimaryNode('stl_arima')
    node_final.custom_params = {'period': 80, 'p': 2, 'd': 1, 'q': 0}
    chain = Chain(node_final)
    return chain


def get_ridge_chain():
    """ Function return complex chain with the following structure
        lagged -> ridge \
                          ridge
        lagged -> ridge  |
    """

    node_lagged_1 = PrimaryNode('lagged/1')
    node_ridge_1 = SecondaryNode('ridge', nodes_from=[node_lagged_1])

    node_lagged_2 = PrimaryNode('lagged/2')

    node_ridge_2 = SecondaryNode('ridge', nodes_from=[node_lagged_2])

    node_final = SecondaryNode('ridge', nodes_from=[node_ridge_1, node_ridge_2])
    chain = Chain(node_final)
    return chain


def compare_plot(predicted, real, forecast_length, model):
    plt.plot(real, linewidth=1, label="Observed", alpha=0.4)
    plt.plot(predicted, linewidth=1, label="Predicted", alpha=0.6)
    plt.legend()
    plt.xlabel('Time index')
    plt.ylabel('SSH, m')
    plt.title(f'Sea surface height forecast for {forecast_length} days with {model}')
    plt.show()


def run_nemo_based_forecasting(time_series, exog_variable, len_forecast=60, is_visualise=False):

    errors_df = {}

    (train_input,
     predict_input,
     train_input_exog,
     predict_input_exog,
     test_data) = prepare_data(time_series=time_series,
                               exog_variable=exog_variable,
                               len_forecast=len_forecast)

    """ Arima models """
    # simple arima
    chain = get_arima_chain()
    train_dataset = MultiModalData({
        'arima': train_input,
    })
    predict_dataset = MultiModalData({
        'arima': predict_input,
    })
    chain.fit_from_scratch(train_dataset)
    predicted_values = chain.predict(predict_dataset)
    predicted_values = predicted_values.predict
    predicted = np.ravel(np.array(predicted_values))
    test_data = np.ravel(test_data)

    if is_visualise:
        compare_plot(predicted, test_data, len_forecast, 'ARIMA')

    mse_before = mean_squared_error(test_data, predicted, squared=False)
    mae_before = mean_absolute_error(test_data, predicted)
    mape_before = mean_absolute_percentage_error(test_data, predicted)
    print(f'ARIMA MSE - {mse_before:.4f}')
    print(f'ARIMA MAE - {mae_before:.4f}')
    print(f'ARIMA MAPE - {mape_before:.4f}\n')

    errors_df['ARIMA_MSE'] = mse_before
    errors_df['ARIMA_MAE'] = mae_before
    errors_df['ARIMA_MAPE'] = mape_before

    # arima with nemo ensemble
    chain = get_arima_nemo_chain()
    train_dataset = MultiModalData({
        'arima': train_input,
        'exog_ts_data_source': train_input_exog
    })
    predict_dataset = MultiModalData({
        'arima': predict_input,
        'exog_ts_data_source': predict_input_exog
    })
    chain.fit_from_scratch(train_dataset)
    predicted_values = chain.predict(predict_dataset)

    predicted = np.ravel(np.array(predicted_values))
    test_data = np.ravel(test_data)

    if is_visualise:
        compare_plot(predicted, test_data, len_forecast, 'ARIMA with nemo')

    mse_before = mean_squared_error(test_data, predicted, squared=False)
    mae_before = mean_absolute_error(test_data, predicted)
    mape_before = mean_absolute_percentage_error(test_data, predicted)
    print(f'ARIMA with nemo MSE - {mse_before:.4f}')
    print(f'ARIMA with nemo MAE - {mae_before:.4f}')
    print(f'ARIMA with nemo MAPE - {mape_before:.4f}\n')

    errors_df['ARIMA_NEMO_MSE'] = mse_before
    errors_df['ARIMA_NEMO_MAE'] = mae_before
    errors_df['ARIMA_NEMO_MAPE'] = mape_before

    """ STL Arima models """
    # simple stl_arima
    chain = get_STLarima_chain()
    train_dataset = MultiModalData({
        'stl_arima': train_input,
    })
    predict_dataset = MultiModalData({
        'stl_arima': predict_input,
    })
    chain.fit_from_scratch(train_dataset)
    predicted_values = chain.predict(predict_dataset)

    predicted = np.ravel(np.array(predicted_values))
    test_data = np.ravel(test_data)

    if is_visualise:
        compare_plot(predicted, test_data, len_forecast, 'STL ARIMA')

    mse_before = mean_squared_error(test_data, predicted, squared=False)
    mae_before = mean_absolute_error(test_data, predicted)
    mape_before = mean_absolute_percentage_error(test_data, predicted)
    print(f'STL ARIMA MSE - {mse_before:.4f}')
    print(f'STL ARIMA MAE - {mae_before:.4f}')
    print(f'STL ARIMA MAPE - {mape_before:.4f}\n')

    errors_df['STL_ARIMA_MSE'] = mse_before
    errors_df['STL_ARIMA_MAE'] = mae_before
    errors_df['STL_ARIMA_MAPE'] = mape_before

    # stl_arima with nemo ensemble
    chain = get_STLarima_nemo_chain()

    train_dataset = MultiModalData({
        'stl_arima': train_input,
        'exog_ts_data_source': train_input_exog
    })
    predict_dataset = MultiModalData({
        'stl_arima': predict_input,
        'exog_ts_data_source': predict_input_exog
    })
    chain.fit_from_scratch(train_dataset)
    predicted_values = chain.predict(predict_dataset)

    predicted = np.ravel(np.array(predicted_values))
    test_data = np.ravel(test_data)

    if is_visualise:
        compare_plot(predicted, test_data, len_forecast, 'STL ARIMA with nemo')

    mse_before = mean_squared_error(test_data, predicted, squared=False)
    mae_before = mean_absolute_error(test_data, predicted)
    mape_before = mean_absolute_percentage_error(test_data, predicted)
    print(f'STL ARIMA with nemo MSE - {mse_before:.4f}')
    print(f'STL ARIMA with nemo MAE - {mae_before:.4f}')
    print(f'STL ARIMA with nemo MAPE - {mape_before:.4f}\n')

    errors_df['STL_ARIMA_NEMO_MSE'] = mse_before
    errors_df['STL_ARIMA_NEMO_MAE'] = mae_before
    errors_df['STL_ARIMA_NEMO_MAPE'] = mape_before

    """ Ridge models """
    # simple ridge
    chain = get_ridge_chain()
    train_dataset = MultiModalData({
        'lagged/1': train_input,
        'lagged/2': train_input
    })
    predict_dataset = MultiModalData({
        'lagged/1': predict_input,
        'lagged/2': predict_input
    })
    chain.fit_from_scratch(train_dataset)
    predicted_values = chain.predict(predict_dataset)

    predicted = np.ravel(np.array(predicted_values))
    test_data = np.ravel(test_data)

    if is_visualise:
        compare_plot(predicted, test_data, len_forecast, 'ridge')

    mse_before = mean_squared_error(test_data, predicted, squared=False)
    mae_before = mean_absolute_error(test_data, predicted)
    mape_before = mean_absolute_percentage_error(test_data, predicted)
    print(f'ridge MSE - {mse_before:.4f}')
    print(f'ridge MAE - {mae_before:.4f}')
    print(f'ridge MAPE - {mape_before:.4f}\n')

    errors_df['RIDGE_MSE'] = mse_before
    errors_df['RIDGE_MAE'] = mae_before
    errors_df['RIDGE_MAPE'] = mape_before

    # ridge with nemo ensemble
    chain = get_ridge_nemo_chain()
    train_dataset = MultiModalData({
        'lagged/1': train_input,
        'lagged/2': train_input,
        'exog_ts_data_source': train_input_exog
    })
    predict_dataset = MultiModalData({
        'lagged/1': predict_input,
        'lagged/2': predict_input,
        'exog_ts_data_source': predict_input_exog
    })
    chain.fit_from_scratch(train_dataset)
    predicted_values = chain.predict(predict_dataset)

    predicted = np.ravel(np.array(predicted_values))
    test_data = np.ravel(test_data)

    if is_visualise:
        compare_plot(predicted, test_data, len_forecast, 'ridge with nemo')

    mse_before = mean_squared_error(test_data, predicted, squared=False)
    mae_before = mean_absolute_error(test_data, predicted)
    mape_before = mean_absolute_percentage_error(test_data, predicted)
    print(f'ridge with nemo MSE - {mse_before:.4f}')
    print(f'ridge with nemo MAE - {mae_before:.4f}')
    print(f'ridge with nemo MAPE - {mape_before:.4f}\n')

    errors_df['RIDGE_NEMO_MSE'] = mse_before
    errors_df['RIDGE_NEMO_MAE'] = mae_before
    errors_df['RIDGE_NEMO_MAPE'] = mape_before

    return errors_df


def boxplot_visualize(df, label):
    ylabel = 'ssh, meters'
    if label == 'MAPE':
        ylabel = ''

    df.boxplot(rot=13)
    plt.ylabel(ylabel)
    plt.title(f'{label} distribution')

    plt.show()


def run_single_example(len_forecast=40, is_visualise=True):
    ts_name = 'sea_level'
    path_to_file = '../cases/data/nemo/sea_surface_height.csv'
    path_to_exog_file = '../cases/data/nemo/sea_surface_height_nemo.csv'

    df = pd.read_csv(path_to_file)
    time_series = df[ts_name]
    df = pd.read_csv(path_to_exog_file)
    exog_variable = df[ts_name]

    run_nemo_based_forecasting(time_series=time_series,
                               exog_variable=exog_variable,
                               len_forecast=len_forecast,
                               is_visualise=is_visualise)


def run_multiple_example(path_to_file, path_to_exog_file, out_path=None, is_boxplot_visualize=True, len_forecast=40):

    mse_errors_df = pd.DataFrame(columns=['POINT', 'RIDGE', 'RIDGE_NEMO',
                                                   'ARIMA', 'ARIMA_NEMO',
                                                   'STL_ARIMA', 'STL_ARIMA_NEMO'])
    mae_errors_df = pd.DataFrame(columns=['POINT', 'RIDGE', 'RIDGE_NEMO',
                                                   'ARIMA', 'ARIMA_NEMO',
                                                   'STL_ARIMA', 'STL_ARIMA_NEMO'])
    mape_errors_df = pd.DataFrame(columns=['POINT', 'RIDGE', 'RIDGE_NEMO',
                                                    'ARIMA', 'ARIMA_NEMO',
                                                    'STL_ARIMA', 'STL_ARIMA_NEMO'])

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
                                                is_visualise=False)

            mse_errors_df = mse_errors_df.append({'POINT': point,
                                                  'RIDGE': errors['RIDGE_MSE'],
                                                  'RIDGE_NEMO': errors['RIDGE_NEMO_MSE'],
                                                  'ARIMA': errors['ARIMA_MSE'],
                                                  'ARIMA_NEMO': errors['ARIMA_NEMO_MSE'],
                                                  'STL_ARIMA': errors['STL_ARIMA_MSE'],
                                                  'STL_ARIMA_NEMO': errors['STL_ARIMA_NEMO_MSE'],
                                                  }, ignore_index=True)

            mae_errors_df = mae_errors_df.append({'POINT': point,
                                                  'RIDGE': errors['RIDGE_MAE'],
                                                  'RIDGE_NEMO': errors['RIDGE_NEMO_MAE'],
                                                  'ARIMA': errors['ARIMA_MAE'],
                                                  'ARIMA_NEMO': errors['ARIMA_NEMO_MAE'],
                                                  'STL_ARIMA': errors['STL_ARIMA_MAE'],
                                                  'STL_ARIMA_NEMO': errors['STL_ARIMA_NEMO_MAE'],
                                                  }, ignore_index=True)

            mape_errors_df = mape_errors_df.append({'POINT': point,
                                                    'RIDGE': errors['RIDGE_MAPE'],
                                                    'RIDGE_NEMO': errors['RIDGE_NEMO_MAPE'],
                                                    'ARIMA': errors['ARIMA_MAPE'],
                                                    'ARIMA_NEMO': errors['ARIMA_NEMO_MAPE'],
                                                    'STL_ARIMA': errors['STL_ARIMA_MAPE'],
                                                    'STL_ARIMA_NEMO': errors['STL_ARIMA_NEMO_MAPE'],
                                                    }, ignore_index=True)

    if out_path is not None:
        mse_errors_df.to_csv(os.path.join(out_path, 'mse_errors.csv'), index=False)
        mae_errors_df.to_csv(os.path.join(out_path, 'mae_errors.csv'), index=False)
        mape_errors_df.to_csv(os.path.join(out_path, 'mape_errors.csv'), index=False)

    if is_boxplot_visualize:
        boxplot_visualize(mse_errors_df, 'MSE')
        boxplot_visualize(mae_errors_df, 'MAE')
        boxplot_visualize(mape_errors_df, 'MAPE')


def run_prediction_examples(mode='single'):
    if mode == 'single':
        run_single_example(len_forecast=40, is_visualise=True)
    if mode == 'multiple':
        run_multiple_example(path_to_file='../cases/data/nemo/SSH_points_grid.csv',
                             path_to_exog_file='../cases/data/nemo/SSH_nemo_points_grid.csv',
                             out_path='../cases/data/nemo/',
                             len_forecast=40,
                             is_boxplot_visualize=True)


if __name__ == '__main__':
    run_prediction_examples(mode='single')
