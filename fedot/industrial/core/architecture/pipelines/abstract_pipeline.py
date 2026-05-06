import json
import os
import time
from datetime import date as current_date
from typing import Union

import numpy as np
import pandas as pd
from fedot.core.data.input_data.data import InputData
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from pymonad.either import Either

from fedot.industrial.api.main import FedotIndustrial
from fedot.industrial.api.utils.checkers_collections import DataCheck
from fedot.industrial.core.metrics.metrics_implementation import RMSE, Accuracy, F1, R2
from fedot.industrial.core.repository.constanst_repository import MONASH_FORECASTING_BENCH, M4_SEASONALITY
from fedot.industrial.core.repository.industrial_implementations.abstract import build_tuner
from fedot.industrial.core.repository.initializer_industrial_models import IndustrialModels
from fedot.industrial.core.repository.model_repository import NEURAL_MODEL
from fedot.industrial.tools.example_utils import load_monash_dataset
from fedot.industrial.tools.loader import DataLoader
from fedot.industrial.tools.serialisation.path_lib import EXAMPLES_DATA_PATH, PATH_TO_DEFAULT_PARAMS

BENCHMARK = 'M4'


class AbstractPipeline:

    def __init__(self, task, task_params={}, task_metric: str = None):
        self.repo = IndustrialModels().setup_repository()
        self.task = task
        self.task_params = task_params
        _metric_dict = {'classification': Accuracy,
                        'regression': RMSE,
                        'ts_forecasting': RMSE,
                        'rmse': RMSE,
                        'f1': F1,
                        'R2': R2
                        }
        if task_metric is not None:
            self.base_metric = _metric_dict[task_metric]
        else:
            self.base_metric = _metric_dict[self.task]

    @staticmethod
    def create_pipeline(node_list: dict, build: bool = True):
        pipeline = PipelineBuilder()
        for branch, nodes in node_list.items():
            for node in nodes:
                if isinstance(branch, int):
                    if isinstance(node, tuple):
                        pipeline.add_node(operation_type=node[0], params=node[1], branch_idx=branch)
                    else:
                        with open(PATH_TO_DEFAULT_PARAMS) as json_data:
                            default_operation_params = json.load(json_data)
                        pipeline.add_node(operation_type=node,
                                          params=default_operation_params[node], branch_idx=branch)
                else:
                    pipeline.join_branches(operation_type=node)
        return pipeline.build() if build else pipeline

    def tune_pipeline(
            self,
            model_to_tune,
            tuning_params,
            tune_data: InputData = None):
        if tune_data is None:
            tune_data = self.train_data
        pipeline_tuner, tuned_model = build_tuner(
            self, model_to_tune, tuning_params, tune_data, 'head')
        return tuned_model

    def create_input_data(self, dataset_name):
        dataset_is_dict = isinstance(dataset_name, dict)
        custom_dataset_strategy = self.task_params['industrial_strategy'] if 'industrial_strategy' \
                                                                             in self.task_params.keys() else self.task
        loader = DataLoader(dataset_name=dataset_name)

        input_train, input_test = Either(value=dataset_name,
                                         monoid=[dataset_name,
                                                 dataset_is_dict]). \
            either(left_function=loader.load_data,
                   right_function=lambda dataset_dict: loader.load_custom_data(custom_dataset_strategy))

        input_train = DataCheck(
            input_data=input_train,
            task=custom_dataset_strategy,
            task_params=self.task_params,
            industrial_task_params=None).check_input_data()
        input_test = DataCheck(
            input_data=input_test,
            task=custom_dataset_strategy,
            task_params=self.task_params,
            industrial_task_params=None).check_input_data()
        return input_train, input_test

    def evaluate_pipeline(self, node_list, dataset):
        test_model = self.create_pipeline(node_list)
        self.train_data, self.test_data = self.create_input_data(dataset)
        test_model.fit(self.train_data)
        if self.task == 'ts_forecasting':
            predict = test_model.predict(self.train_data)
            predict_proba = predict
            target = self.train_data.features[-self.task_params['forecast_length']:].flatten()
        else:
            predict = test_model.predict(self.test_data, 'labels')
            predict_proba = test_model.predict(self.test_data, 'probs')
            target = self.test_data.target
        metric = self.base_metric(target=target,
                                  predicted_probs=predict_proba.predict,
                                  predicted_labels=predict.predict).metric()
        return dict(fitted_model=test_model,
                    predict_labels=predict.predict,
                    predict_probs=predict_proba.predict,
                    quality_metric=metric)


class ApiTemplate:

    def __init__(self,
                 api_config,
                 metric_list):
        self.api_config = api_config
        self.metric_names = metric_list
        self.industrial_class = None
        self.train_data, self.test_data = None, None
        self.seasonality = 1

    def _prepare_dataset(self, dataset):
        dataset_is_dict = isinstance(dataset, dict)
        industrial_config = self.api_config.get('industrial_config', {})
        have_specified_industrial_strategy = 'strategy' in industrial_config.keys() \
                                             or 'strategy_params' in industrial_config.keys()

        if have_specified_industrial_strategy:
            custom_dataset_strategy = industrial_config['strategy']
        else:
            custom_dataset_strategy = industrial_config.get('problem')

        loader = DataLoader(dataset_name=dataset)

        train_data, test_data = Either(value=dataset,
                                       monoid=[dataset,
                                               dataset_is_dict or have_specified_industrial_strategy]). \
            either(left_function=loader.load_data,
                   right_function=lambda dataset_dict: loader.load_custom_data(custom_dataset_strategy))
        return train_data, test_data

    def _get_result(self, test_data):
        labels = self.industrial_class.predict(test_data)
        probs = self.industrial_class.predict_proba(test_data)
        metrics = self.industrial_class.get_metrics(labels=labels,
                                                    probs=probs,
                                                    target=test_data[1],
                                                    rounding_order=3,
                                                    metric_names=self.metric_names,
                                                    train_data=self.train_data[0],
                                                    seasonality=self.seasonality)
        train_data = self.train_data[0]
        self.industrial_class.save('all')
        result_dict = dict(industrial_model=self.industrial_class,
                           labels=labels,
                           metrics=metrics,
                           train_data=train_data)
        return result_dict

    def eval(self,
             dataset: Union[str, dict] = None,
             finetune: bool = False,
             initial_assumption: Union[list, dict, str] = None):
        self.train_data, self.test_data = self._prepare_dataset(dataset)
        have_init_assumption = initial_assumption is not None and len(initial_assumption) != 0
        pipeline_to_tune = None
        if have_init_assumption:
            pipeline_to_tune = AbstractPipeline.create_pipeline(initial_assumption, build=False)
            return_only_fitted = pipeline_to_tune.heads[0].name in list(NEURAL_MODEL.keys())
        self.industrial_class = FedotIndustrial(**self.api_config)
        start_time = time.time()
        Either(
            value=self.train_data,
            monoid=[
                dict(
                    train_data=self.train_data,
                    model_to_tune=pipeline_to_tune,
                    tuning_params={
                        'tuning_iterations': 5}),
                not finetune]).either(
            left_function=lambda tuning_data: self.industrial_class.finetune(
                **tuning_data,
                return_only_fitted=return_only_fitted),
            right_function=self.industrial_class.fit)
        end_time = time.time() - start_time
        self.industrial_class.shutdown()
        result = self._get_result(self.test_data)
        result['time'] = end_time
        return result

    def load_result(self, benchmark_path):
        dir_list = os.listdir(benchmark_path)
        result_dict = {}
        for model_dir in dir_list:
            datasets_dir = os.listdir(f'{benchmark_path}/{model_dir}')
            df_with_results = [pd.read_csv(f'{benchmark_path}/{model_dir}/{dataset}/metrics.csv')
                               for dataset in datasets_dir]
            df_with_results = pd.concat(df_with_results)
            del df_with_results['Unnamed: 0']
            df_with_results.columns = [f'{x}_{model_dir}' for x in df_with_results.columns]
            df_with_results['dataset'] = datasets_dir
            result_dict.update({model_dir: df_with_results})
        return result_dict

    def evaluate_benchmark(self, benchmark_name, benchmark_params: dict):
        for dataset in benchmark_params['datasets']:
            print(f'\nEvaluating {dataset} dataset')

            if benchmark_name.__contains__('M4'):
                dataset_for_eval = self._prepare_forecasting_data(dataset, benchmark_name, benchmark_params)
            elif benchmark_name.__contains__('SKAB'):
                dataset_for_eval = self._prepare_skab_data(dataset, benchmark_name, benchmark_params)
            elif benchmark_name in MONASH_FORECASTING_BENCH:
                dataset_for_eval = self._prepare_monash_data(dataset, benchmark_name, benchmark_params)
            else:
                dataset_for_eval = dataset
            for model_impl, model_name, finetune_strategy in zip(*benchmark_params['model_to_compare']):
                date_ = benchmark_params.get('experiment_date', current_date.today().isoformat())
                benchmark_folder = benchmark_params.get('benchmark_folder', './benchmark_results')
                output_folder = os.path.join(benchmark_folder, f'{date_}_{benchmark_name}', model_name, dataset)
                self.api_config['compute_config']['output_folder'] = output_folder
                try:
                    result_dict = self.eval(
                        dataset=dataset_for_eval,
                        initial_assumption=model_impl,
                        finetune=finetune_strategy)
                    print(result_dict['metrics'])
                    np.save(os.path.join(output_folder, 'results.npy'), result_dict)
                except BaseException:
                    result_dict = {}

    def _prepare_forecasting_data(self, dataset, benchmark_name, benchmark_dict):
        prefix = dataset[0]
        horizon = benchmark_dict['metadata'][prefix]
        dataset_for_eval = {'benchmark': benchmark_name[:2],
                            'dataset': dataset,
                            'task_params': {'forecast_length': horizon}}
        self.seasonality = M4_SEASONALITY[prefix]
        self.api_config['industrial_config']['task_params']['forecast_length'] = horizon
        self.api_config['automl_config']['task_params']['forecast_length'] = horizon
        return dataset_for_eval

    def _prepare_skab_data(self, dataset, benchmark_name, benchmark_dict):
        folder = benchmark_dict['metadata']['folder']
        path_to_result = EXAMPLES_DATA_PATH + f'/benchmark/detection/data/{folder}/{dataset}.csv'
        df = pd.read_csv(path_to_result, index_col='datetime', sep=';', parse_dates=True)
        train_idx = self.api_config['industrial_config']['strategy_params']['train_data_size']
        if isinstance(train_idx, str):
            train_data = EXAMPLES_DATA_PATH + f'/benchmark/detection/data/{train_idx}/{train_idx}.csv'
            train_data = pd.read_csv(train_data, index_col='datetime', sep=';', parse_dates=True)
            label = np.array([0 for x in range(len(train_data))])
            dataset_for_eval = {'train_data': (train_data.values, label),
                                'test_data': (df.iloc[:, :-2].values, df.iloc[:, -2].values)}
        else:
            dataset_for_eval = {'train_data': (df.iloc[:train_idx, :-2].values, df.iloc[:train_idx, -2].values),
                                'test_data': (df.iloc[train_idx:, :-2].values, df.iloc[train_idx:, -2].values)}
        return dataset_for_eval

    def _prepare_monash_data(self, dataset, benchmark_name, benchmark_dict):
        monash_df = load_monash_dataset(benchmark_name)
        horizon = benchmark_dict['metadata'][benchmark_name]
        features = monash_df[dataset].values
        target = monash_df[dataset][-horizon:].values
        dataset_for_eval = dict(train_data=(features, target),
                                test_data=(features, target))
        self.seasonality = horizon
        self.api_config['industrial_config']['task_params']['forecast_length'] = horizon
        self.api_config['automl_config']['task_params']['forecast_length'] = horizon
        return dataset_for_eval
