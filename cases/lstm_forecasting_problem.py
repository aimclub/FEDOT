import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

from sklearn.metrics import mean_squared_error as mse
from statsmodels.tsa.seasonal import seasonal_decompose

from core.composer.chain import Chain
from core.composer.composer import ComposerRequirements, DummyChainTypeEnum, DummyComposer
from core.models.data import OutputData, InputData
from core.repository.model_types_repository import ModelTypesIdsEnum
from core.repository.dataset_types import NumericalDataTypesEnum, CategoricalDataTypesEnum
from core.repository.model_types_repository import (
    ModelMetaInfoTemplate,
    ModelTypesRepository
)
from core.repository.quality_metrics_repository import MetricsRepository, RegressionMetricsEnum
from core.repository.task_types import MachineLearningTasksEnum
from core.utils import project_root, ts_to_3d


window_len = 64
prediction_len = 1


def add_root(file):
    return os.path.join(str(project_root()), 'cases/out/' + file)


def get_trend_resid_datasets(file_path, period=None):
    # load features and target (without index)
    ts = pd.read_csv(file_path, header=None).values[:, 1:]

    if period is None:
        f, Pxx_den = signal.welch(
            ts[:, -1], fs=1, scaling='spectrum', nfft=1000, nperseg=1000)
        period = int(1/f[np.argmax(Pxx_den)])

    # extract trend and resid from target
    decomposed = seasonal_decompose(
        ts[:, -1], period=period, extrapolate_trend='freq')
    trend = decomposed.trend[:, None]
    resids = ts[:, [-1]] - trend

    # concat features and trend for better prediction
    features = np.c_[ts[:, :-1], trend]

    features_file = add_root('trend_features.npy')
    # target are features with shift `prediction_len`
    features_sliding = ts_to_3d(features[:-prediction_len], window_len)
    np.save(features_file, features_sliding)

    target_file = add_root('trend_target.npy')
    target_sliding = ts_to_3d(features[prediction_len:, [-1]], window_len)
    np.save(target_file, target_sliding)

    index = np.arange(features_sliding.shape[0])
    trend_dataset = InputData.from_npy(features_file, target_file, index)

    # getting resid_dataset
    resids = ts_to_3d(resids, window_len+prediction_len)
    # make (n, window_len * features) to use as simple ML problem
    resids_features = resids[:, :-prediction_len].reshape(resids.shape[0], -1)
    # make (n, prediction_len * features) to use as simple ML problem
    resids_target = resids[:, -prediction_len:].reshape(resids.shape[0], -1)

    resids_file = add_root('resids.csv')
    # add features for better prediction
    pd.DataFrame(np.c_[resids_features, resids_target],
                 index=index).to_csv(resids_file, header=False)
    resid_dataset = InputData.from_csv(
        resids_file, task_type=MachineLearningTasksEnum.regression)

    return period, trend_dataset, resid_dataset


def calculate_validation_metric(chain: Chain, dataset_to_validate: InputData, name: str) -> float:
    # the execution of the obtained composite models
    predicted = chain.predict(dataset_to_validate).predict
    real = dataset_to_validate.target

    np.save(add_root(name+'.npy'), predicted)
    # get only last value - forecasting.
    # it's due 3d format of this problem
    if dataset_to_validate.task_type == MachineLearningTasksEnum.forecasting:
        predicted = predicted[:, -1]
        real = real[:, -1]

    # plot results
    compare_plot(predicted, real, add_root(name+'.png'))

    # the quality assessment for the simulation results
    rmse = mse(y_true=real, y_pred=predicted, squared=False)

    return rmse


def compare_plot(predicted, real, filepath):
    plt.clf()
    _, ax = plt.subplots()
    plt.plot(real, linewidth=1, label="Observed", alpha=0.4)
    plt.plot(predicted, linewidth=1, label="Predicted", alpha=0.6)
    ax.legend()

    plt.savefig(filepath)


# specify problem type
problem_class = MachineLearningTasksEnum.forecasting

period, trend_train, resid_train = get_trend_resid_datasets(
    os.path.join(str(project_root()),
                 'cases/data/ts/metocean_data_train.csv'))

_, trend_test, resid_test = get_trend_resid_datasets(
    os.path.join(str(project_root()),
                 'cases/data/ts/metocean_data_test.csv'),
    period=period)


metric_function = MetricsRepository().metric_by_id(RegressionMetricsEnum.RMSE)


# 1 step - fit trend
trend_composer_requirements = ComposerRequirements(primary=[ModelTypesIdsEnum.lstm],
                                                   secondary=[])

trend = DummyComposer(
    DummyChainTypeEnum.flat).compose_chain(data=trend_train,
                                           initial_chain=None,
                                           composer_requirements=trend_composer_requirements,
                                           metrics=metric_function)

_ = trend.fit(input_data=trend_train, verbose=True)
rmse_on_valid = calculate_validation_metric(trend, trend_test, 'trend')
print(f'Trend RMSE: {rmse_on_valid}')


# 2 step - fit resid
resid_composer_requirements = ComposerRequirements(primary=[ModelTypesIdsEnum.ridge, ModelTypesIdsEnum.rfr],
                                                   secondary=[ModelTypesIdsEnum.ridge])

resid = DummyComposer(
    DummyChainTypeEnum.hierarchical).compose_chain(data=resid_train,
                                                   initial_chain=None,
                                                   composer_requirements=resid_composer_requirements,
                                                   metrics=metric_function)

resid_prediction = resid.fit(input_data=resid_train, verbose=True)

rmse_on_valid = calculate_validation_metric(resid, resid_test, 'resid')
print(f'Resid RMSE: {rmse_on_valid}')
