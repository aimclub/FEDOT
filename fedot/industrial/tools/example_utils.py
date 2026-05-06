import os
import random
from pathlib import Path
from typing import Optional, Callable

import pandas as pd
from datasets import load_dataset
from fedot.core.data.input_data.data import InputData
from fedot.core.data.split.data_split import train_test_data_setup
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.utilities.ts_gapfilling import ModelGapFiller
from sklearn.metrics import f1_score, roc_auc_score

from fedot.industrial.core.architecture.settings.computational import backend_methods as np
from fedot.industrial.core.metrics.metrics_implementation import calculate_forecasting_metric
from fedot.industrial.tools.serialisation.path_lib import PROJECT_PATH

ts_datasets = {
    'm4_yearly': Path(PROJECT_PATH, 'examples', 'data', 'ts', 'M4Yearly.csv'),
    'm4_weekly': Path(PROJECT_PATH, 'examples', 'data', 'ts', 'M4Weekly.csv'),
    'm4_daily': Path(PROJECT_PATH, 'examples', 'data', 'ts', 'M4Daily.csv'),
    'm4_monthly': Path(PROJECT_PATH, 'examples', 'data', 'ts', 'M4Monthly.csv'),
    'm4_quarterly': Path(PROJECT_PATH, 'examples', 'data', 'ts', 'M4Quarterly.csv')}


def evaluate_metric(target, prediction):
    try:
        if len(np.unique(target)) > 2:
            metric = f1_score(target, prediction, average='weighted')
        else:
            metric = roc_auc_score(target, prediction, average='weighted')
    except Exception:
        metric = f1_score(target, np.argmax(
            prediction, axis=1), average='weighted')
    return metric


def compare_forecast_with_sota(dataset_name, horizon):
    autogluon = PROJECT_PATH + f'/benchmark/results/benchmark_results/autogluon/' \
                               f'{dataset_name}_{horizon}_forecast_vs_actual.csv'
    n_beats = PROJECT_PATH + f'/benchmark/results/benchmark_results/nbeats/' \
                             f'{dataset_name}_{horizon}_forecast_vs_actual.csv'
    n_beats = pd.read_csv(n_beats)
    autogluon = pd.read_csv(autogluon)

    n_beats_forecast = calculate_forecasting_metric(
        target=n_beats['value'].values,
        labels=n_beats['predict'].values)
    autogluon_forecast = calculate_forecasting_metric(
        target=autogluon['value'].values,
        labels=autogluon['predict'].values)
    return n_beats['predict'].values, n_beats_forecast, autogluon['predict'].values, autogluon_forecast


def read_results(forecast_result_path):
    results = os.listdir(forecast_result_path)
    df_forecast = []
    df_metrics = []
    for file in results:
        df = pd.read_csv(f'{forecast_result_path}/{file}')
        name = file.split('_')[0]
        df['dataset_name'] = name
        if file.__contains__('forecast'):
            df_forecast.append(df)
        else:
            df_metrics.append(df)
    return df_forecast, df_metrics


def create_comprasion_df(df, metric: str = 'rmse'):
    df_full = pd.concat(df)
    df_full = df_full[df_full['Unnamed: 0'] == metric]
    df_full = df_full.drop('Unnamed: 0', axis=1)
    df_full['Difference_industrial_All'] = (
        df_full.iloc[:, 1:3].min(axis=1) - df_full['industrial'])
    df_full['Difference_industrial_AG'] = (
        df_full.iloc[:, 1:2].min(axis=1) - df_full['industrial'])
    df_full['Difference_industrial_NBEATS'] = (
        df_full.iloc[:, 2:3].min(axis=1) - df_full['industrial'])
    df_full['industrial_Wins_All'] = df_full.apply(
        lambda row: 'Win' if row.loc['Difference_industrial_All'] > 0 else 'Loose', axis=1)
    df_full['industrial_Wins_AG'] = df_full.apply(
        lambda row: 'Win' if row.loc['Difference_industrial_AG'] > 0 else 'Loose', axis=1)
    df_full['industrial_Wins_NBEATS'] = df_full.apply(
        lambda row: 'Win' if row.loc['Difference_industrial_NBEATS'] > 0 else 'Loose',
        axis=1)
    return df_full


def create_feature_generator_strategy():
    stat_params = {'window_size': 0, 'stride': 1, 'add_global_features': True,
                   'channel_independent': False, 'use_sliding_window': False}
    fourier_params = {'low_rank': 5, 'output_format': 'signal', 'compute_heuristic_representation': True,
                      'approximation': 'smooth', 'threshold': 0.9, 'sampling_rate': 64e3}
    wavelet_params = {'n_components': 3, 'wavelet': 'bior3.7', 'compute_heuristic_representation': True}
    rocket_params = {'num_features': 200}
    sampling_dict = dict(samples=dict(start_idx=0, end_idx=None),
                         channels=dict(start_idx=0, end_idx=None),
                         elements=dict(start_idx=0, end_idx=None))
    feature_generator = {
        # 'minirocket': [('minirocket_extractor', rocket_params)],
        'stat_generator': [('quantile_extractor', stat_params)],
        'fourier': [('fourier_basis', fourier_params)],
        'wavelet': [('wavelet_basis', wavelet_params)],
    }
    return feature_generator, sampling_dict


def get_ts_data(dataset='m4_monthly', horizon: int = 30, m4_id=None):
    time_series = pd.read_csv(ts_datasets[dataset])

    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=horizon))
    if not m4_id:
        label = random.choice(np.unique(time_series['label']))
    else:
        label = m4_id
    print(label)
    time_series = time_series[time_series['label'] == label]

    if 'datetime' in time_series.columns:
        idx = pd.to_datetime(time_series['datetime'].values)
    else:
        # non datetime indexes
        idx = time_series['idx'].values

    time_series = time_series['value'].values
    train_input = InputData(idx=idx,
                            features=time_series,
                            target=time_series,
                            task=task,
                            data_type=DataTypesEnum.ts)
    train_data, test_data = train_test_data_setup(train_input)
    return train_data, test_data, label


def linearly_interpolate_nans(target: np.ndarray) -> np.ndarray:
    if any([np.isnan(value) for value in target]):
        X = np.vstack((np.ones(len(target)), np.arange(len(target))))
        X_fit = X[:, ~np.isnan(target)]
        target_fit = target[~np.isnan(target)].reshape(-1, 1)
        beta = np.linalg.lstsq(X_fit.T, target_fit)[0]
        target.flat[np.isnan(target)] = np.dot(X[:, np.isnan(target)].T, beta)
    return target


def get_composite_pipeline():
    node_1 = PipelineNode('lagged')
    node_1.parameters = {'window_size': 150}
    node_2 = PipelineNode('lagged')
    node_2.parameters = {'window_size': 100}
    node_linear_1 = PipelineNode('linear', nodes_from=[node_1])
    node_linear_2 = PipelineNode('linear', nodes_from=[node_2])

    node_final = PipelineNode('ridge', nodes_from=[node_linear_1,
                                                   node_linear_2])
    pipeline = Pipeline(node_final)
    return pipeline


def get_gapfilled_with_fedot(target: np.ndarray) -> np.ndarray:
    if any([np.isnan(value) for value in target]):
        composite_gapfiller = ModelGapFiller(gap_value=-100.0,
                                             pipeline=get_composite_pipeline())
        filtered_composite = composite_gapfiller.forward_filling(target)
        return filtered_composite
    return target


def load_monash_dataset(dataset_name: str,
                        gapfilling_func: Optional[Callable] = get_gapfilled_with_fedot) -> pd.DataFrame:
    dataset = load_dataset('monash_tsf', dataset_name, trust_remote_code=True)
    wide_data = {}

    for series in dataset['test']:
        label, start_date = series['item_id'], series['start']
        if any(isinstance(value, list) for value in series['target']):
            break

        row = None
        if gapfilling_func is not None:
            try:
                row = gapfilling_func(np.array(series['target']))
            except Exception:
                pass
        wide_data[label] = row if row is not None else linearly_interpolate_nans(np.array(series['target']))

    return pd.DataFrame.from_dict(wide_data, orient='index').transpose()
