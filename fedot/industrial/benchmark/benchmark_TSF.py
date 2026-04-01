import gc
import logging
import os
import warnings
from abc import ABC
from copy import deepcopy
from typing import Union

import pandas as pd

from benchmark.v2.api import run_forecasting_benchmark_from_legacy_config

LEGACY_IMPORT_ERROR = None

try:
    import matplotlib
    from fedot.core.repository.tasks import TsForecastingParams
    from matplotlib import pyplot as plt

    from benchmark.abstract_bench import AbstractBenchmark
    from benchmark.feature_utils import DatasetFormatting
    from fedot_ind.api.main import FedotIndustrial
    from fedot_ind.core.architecture.postprocessing.results_picker import ResultsPicker
    from fedot_ind.core.architecture.settings.computational import backend_methods as np
    from fedot_ind.core.metrics.metrics_implementation import RMSE, SMAPE
    from fedot_ind.core.repository.constanst_repository import M4_FORECASTING_LENGTH, MULTI_CLF_BENCH, UNI_CLF_BENCH
    from fedot_ind.tools.loader import DataLoader
    from fedot_ind.tools.serialisation.path_lib import PROJECT_PATH
except Exception as exc:  # pragma: no cover - legacy-only fallback for lightweight envs
    LEGACY_IMPORT_ERROR = exc


    class AbstractBenchmark:
        def __init__(self, output_dir=None):
            self.output_dir = output_dir


    class DatasetFormatting:
        def format_univariate_forecasting_data(self, *args, **kwargs):
            raise ImportError('Legacy forecasting dependencies are unavailable.') from LEGACY_IMPORT_ERROR

        def format_global_forecasting_data(self, *args, **kwargs):
            raise ImportError('Legacy forecasting dependencies are unavailable.') from LEGACY_IMPORT_ERROR


    class ResultsPicker:
        def __init__(self, path=None):
            self.path = path


    TsForecastingParams = None
    FedotIndustrial = None
    np = None
    RMSE = None
    SMAPE = None
    M4_FORECASTING_LENGTH = {'D': 14, 'W': 13, 'M': 18, 'Q': 8, 'Y': 6}
    MULTI_CLF_BENCH = []
    UNI_CLF_BENCH = []
    DataLoader = None
    PROJECT_PATH = os.getcwd()
    matplotlib = None
    plt = None


class BenchmarkTSF(AbstractBenchmark, ABC):
    def __init__(self,
                 experiment_setup: dict = None,
                 custom_datasets: Union[list, str] = None,
                 use_small_datasets: bool = False):

        super(BenchmarkTSF, self).__init__(
            output_dir='./tser/benchmark_results')

        self.logger = logging.getLogger(self.__class__.__name__)

        self.experiment_setup = experiment_setup
        self.multi_TSC = MULTI_CLF_BENCH
        self.uni_TSC = UNI_CLF_BENCH
        self.automl_TSC = False
        if custom_datasets is None:
            if use_small_datasets:
                self.custom_datasets = self.uni_TSC
            else:
                self.custom_datasets = self.multi_TSC
        else:
            self.custom_datasets = custom_datasets

        if isinstance(self.custom_datasets, str):
            if self.custom_datasets.__contains__('automl'):
                self.automl_TSC = True

        if use_small_datasets:
            self.path_to_result = '/benchmark/results/time_series_uni_forecats_comparasion.csv'
            self.path_to_save = '/benchmark/results/ts_uni_forecasting'
        else:
            self.path_to_result = '/benchmark/results/m4_results.csv'
            self.path_to_save = '/benchmark/results/ts_uni_forecasting'

        self.results_picker = ResultsPicker(
            path=os.path.abspath(self.output_dir))
        self.automl_loader = DatasetFormatting()

    @staticmethod
    def _ensure_legacy_dependencies():
        if LEGACY_IMPORT_ERROR is not None:
            raise ImportError(
                'Legacy BenchmarkTSF dependencies are unavailable in this environment. '
                'Use `use_benchmark_v2=True` or install the legacy stack.'
            ) from LEGACY_IMPORT_ERROR

    def evaluate_loop(self, dataset, experiment_setup: dict = None):
        self._ensure_legacy_dependencies()
        matplotlib.use('TkAgg')
        if self.automl_TSC:
            train_data = dataset[1]
            experiment_setup['task_params'] = TsForecastingParams(
                forecast_length=self.automl_TSC_metadata[self.automl_TSC_metadata['file'] == dataset[0]]['horizon'][0])
        else:
            train_data = DataLoader(dataset_name=dataset).load_forecast_data()
            experiment_setup['task_params'] = TsForecastingParams(
                forecast_length=M4_FORECASTING_LENGTH[dataset[0]])
        target = train_data.iloc[-experiment_setup['task_params'].forecast_length:, :].values.ravel()
        train_data = train_data.iloc[:-experiment_setup['task_params'].forecast_length, :]
        model = FedotIndustrial(**experiment_setup)
        model.fit(train_data)
        prediction = model.predict(train_data)
        plt.close('all')
        return prediction, target, model

    def _save_with_basic_results(self,
                                 dataset_name,
                                 basic_results,
                                 metric,
                                 target,
                                 prediction,
                                 model):
        dataset_path = os.path.join(
            self.experiment_setup['output_folder'], f'{dataset_name}')
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)
        basic_results.loc[dataset_name, 'Fedot_Industrial'] = metric
        basic_results.to_csv(os.path.join(
            dataset_path, 'metrics_report.csv'))
        pred_df = pd.DataFrame([target, prediction]).T
        pred_df.columns = ['label', 'prediction']
        pred_df.to_csv(os.path.join(dataset_path, 'prediction.csv'))
        model.solver.save(dataset_path)
        gc.collect()

    def run(self, path: str = None):
        if self.experiment_setup.get('use_benchmark_v2'):
            warnings.warn(
                'BenchmarkTSF.run is delegating to benchmark.v2. '
                'Use benchmark.v2.run_forecasting_benchmark_suite directly for new code.',
                DeprecationWarning,
                stacklevel=2,
            )
            return run_forecasting_benchmark_from_legacy_config(self.experiment_setup)

        self._ensure_legacy_dependencies()
        self.logger.info('Benchmark test started')
        metric_dict = {}
        if self.automl_TSC and path is not None:
            dataset_list = self.load_automl_benchmark(path).items()
            basic_results = None
        else:
            dataset_list = self.custom_datasets
            basic_results = self.load_local_basic_results()
        for dataset_name in dataset_list:
            experiment_setup = deepcopy(self.experiment_setup)
            prediction, target, model = self.evaluate_loop(
                dataset_name, experiment_setup)
            metric = SMAPE(prediction, target).metric()
            metric_dict.update({dataset_name[0]: metric})
            if basic_results is not None:
                self._save_with_basic_results(
                    dataset_name,
                    basic_results,
                    metric,
                    target,
                    prediction,
                    model)
        if basic_results is not None:
            basic_path = os.path.join(
                self.experiment_setup['output_folder'],
                'comprasion_metrics_report.csv')
            basic_results.to_csv(basic_path)
        else:
            pd.DataFrame(metric_dict).to_csv('./metric_result.csv')
        self.logger.info("Benchmark test finished")

    def finetune(self):
        self._ensure_legacy_dependencies()
        self.logger.info('Benchmark finetune started')
        for dataset_name in self.custom_datasets:
            composed_model_path = PROJECT_PATH + self.path_to_save + \
                f'/{dataset_name}' + '/0_pipeline_saved'
            if os.path.isdir(composed_model_path):
                self.experiment_setup['output_folder'] = PROJECT_PATH + \
                    self.path_to_save
                experiment_setup = deepcopy(self.experiment_setup)
                prediction, target = self.finetune_loop(
                    dataset_name, experiment_setup)
                metric = RMSE(target, prediction).metric()
                dataset_path = os.path.join(
                    self.experiment_setup['output_folder'],
                    f'{dataset_name}',
                    'metrics_report.csv')
                fedot_results = pd.read_csv(dataset_path, index_col=0)
                fedot_results.loc[dataset_name,
                                  'Fedot_Industrial_finetuned'] = metric

                fedot_results.to_csv(dataset_path)
            else:
                print(f"No composed model for dataset - {dataset_name}")
            gc.collect()
        self.logger.info("Benchmark finetune finished")

    def load_local_basic_results(self, path: str = None):
        self._ensure_legacy_dependencies()
        path = PROJECT_PATH + self.path_to_result
        results = pd.read_csv(path, sep=',', index_col=0).T
        results = results.dropna(axis=1, how='all')
        results = results.dropna(axis=0, how='all')
        self.experiment_setup['output_folder'] = PROJECT_PATH + \
            self.path_to_save
        return results

    def create_report(self):
        self._ensure_legacy_dependencies()
        _ = []
        names = []
        for dataset_name in self.custom_datasets:
            model_result_path = PROJECT_PATH + self.path_to_save + \
                f'/{dataset_name}' + '/metrics_report.csv'
            if os.path.isfile(model_result_path):
                df = pd.read_csv(model_result_path, index_col=0, sep=',')
                df = df.fillna(0)
                if 'Fedot_Industrial_finetuned' not in df.columns:
                    df['Fedot_Industrial_finetuned'] = 0
                metrics = df.loc[dataset_name,
                                 'Fedot_Industrial':'Fedot_Industrial_finetuned']
                _.append(metrics.T.values)
                names.append(dataset_name)
        stacked_results = np.stack(_, axis=1).T
        df_res = pd.DataFrame(stacked_results, index=names)
        df_res.columns = ['Fedot_Industrial', 'Fedot_Industrial_finetuned']
        del df['Fedot_Industrial'], df['Fedot_Industrial_finetuned']
        df = df.join(df_res)
        df = df.fillna(0)
        return df

    def load_automl_benchmark(self, path):
        self._ensure_legacy_dependencies()
        load_method_dict = dict(automl_univariate=self.automl_loader.format_univariate_forecasting_data,
                                automl_global=self.automl_loader.format_global_forecasting_data)
        self.automl_TSC_metadata, data_dict = load_method_dict[self.custom_datasets](path, True)
        return data_dict
