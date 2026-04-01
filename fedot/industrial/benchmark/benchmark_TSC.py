import gc
import logging
import os
import shutil
import warnings
from abc import ABC
from copy import deepcopy
from typing import Union

import pandas as pd

from benchmark.v2.api import run_tsc_benchmark_from_legacy_config

LEGACY_IMPORT_ERROR = None

try:
    from benchmark.abstract_bench import AbstractBenchmark
    from fedot_ind import __version__
    from fedot_ind.core.architecture.pipelines.abstract_pipeline import ApiTemplate
    from fedot_ind.core.architecture.postprocessing.results_picker import ResultsPicker
    from fedot_ind.core.architecture.settings.computational import backend_methods as np
    from fedot_ind.core.metrics.metrics_implementation import Accuracy
    from fedot_ind.core.repository.config_repository import DEFAULT_COMPUTE_CONFIG
    from fedot_ind.core.repository.constanst_repository import MULTI_CLF_BENCH, UNI_CLF_BENCH
    from fedot_ind.tools.serialisation.path_lib import PROJECT_PATH, BENCHMARK_RESULTS_PATH
except Exception as exc:  # pragma: no cover - legacy-only fallback
    LEGACY_IMPORT_ERROR = exc


    class AbstractBenchmark:
        def __init__(self, output_dir=None):
            self.output_dir = output_dir


    __version__ = 'unknown'


    class ResultsPicker:
        def __init__(self, path=None):
            self.path = path

        def run(self, *args, **kwargs):
            return pd.DataFrame()


    ApiTemplate = None
    np = None
    Accuracy = None
    DEFAULT_COMPUTE_CONFIG = {'output_folder': './benchmark/results'}
    MULTI_CLF_BENCH = []
    UNI_CLF_BENCH = []
    PROJECT_PATH = os.getcwd()
    BENCHMARK_RESULTS_PATH = os.path.join(PROJECT_PATH, 'benchmark', 'results')


class BenchmarkTSC(AbstractBenchmark, ABC):
    def __init__(self,
                 experiment_setup: dict = None,
                 custom_datasets: list = None,
                 use_small_datasets: bool = False,
                 metric_names: Union[list, tuple] = None,
                 initial_assumptions: Union[list, dict] = None,
                 finetune: bool = True):

        super(BenchmarkTSC, self).__init__(output_dir=BENCHMARK_RESULTS_PATH)

        self.logger = logging.getLogger(self.__class__.__name__)

        self.experiment_setup = experiment_setup
        self.industrial_config = experiment_setup.get('industrial_config', {})
        self.automl_config = experiment_setup.get('automl_config', {})
        self.learning_config = experiment_setup.get('learning_config', {})
        self.compute_config = experiment_setup.get('compute_config', DEFAULT_COMPUTE_CONFIG)
        self.experiment_setup['compute_config'] = self.compute_config
        self.metric_names = metric_names
        self.need_finetune = finetune
        self.init_assumption = deepcopy(initial_assumptions)

        if custom_datasets is None:
            if use_small_datasets:
                self.custom_datasets = UNI_CLF_BENCH
            else:
                self.custom_datasets = MULTI_CLF_BENCH
        else:
            self.custom_datasets = custom_datasets

        if use_small_datasets:
            self.comparison_file_path = os.path.join(BENCHMARK_RESULTS_PATH, 'time_series_uni_clf_comparison.csv')
            self.result_dir_name = 'ts_uni_classification'
        else:
            self.comparison_file_path = os.path.join(BENCHMARK_RESULTS_PATH, 'time_series_multi_clf_comparison.csv')
            self.result_dir_name = 'ts_multi_classification'

        output_folder = self.experiment_setup['compute_config'].get('output_folder', BENCHMARK_RESULTS_PATH)
        self.result_dir = os.path.join(output_folder, self.result_dir_name)
        self.results_picker = ResultsPicker(path=os.path.abspath(output_folder))

    @staticmethod
    def _ensure_legacy_dependencies():
        if LEGACY_IMPORT_ERROR is not None:
            raise ImportError(
                'Legacy BenchmarkTSC dependencies are unavailable in this environment. '
                'Use `use_benchmark_v2=True` or install the legacy stack.'
            ) from LEGACY_IMPORT_ERROR

    def _run_model_versus_model(self, dataset_name, comparison_dict: dict):
        self._ensure_legacy_dependencies()
        approach_dict = {}
        metric_name = self.learning_config.get('optimisation_loss', {}).get('quality_loss', 'accuracy')
        for approach, node_dict in comparison_dict.items():
            result_dict = ApiTemplate(api_config=self.experiment_setup,
                                      metric_list=self.metric_names)\
                .eval(dataset=dataset_name,
                      initial_assumption=node_dict,
                      finetune=self.need_finetune)
            metric = result_dict['metrics'][metric_name][0]
            approach_dict.update({approach: metric})
        return approach_dict

    def _run_industrial_versus_sota(self, dataset_name):
        self._ensure_legacy_dependencies()
        experiment_setup = deepcopy(self.experiment_setup)
        prediction, target = self.evaluate_loop(dataset_name, experiment_setup)
        return Accuracy(target, prediction).metric()

    def run(self):
        if self.experiment_setup.get('use_benchmark_v2'):
            warnings.warn(
                'BenchmarkTSC.run is delegating to benchmark.v2. '
                'Use benchmark.v2.run_tsc_benchmark_suite directly for new code.',
                DeprecationWarning,
                stacklevel=2,
            )
            return run_tsc_benchmark_from_legacy_config(self.experiment_setup)

        self._ensure_legacy_dependencies()
        self.logger.info('Benchmark run started')
        basic_results = self.load_local_basic_results()
        for dataset_name in self.custom_datasets:
            try:
                if isinstance(self.init_assumption, dict):
                    metric_dict = self._run_model_versus_model(dataset_name, self.init_assumption)
                else:
                    approach = f'Fedot_Industrial_{__version__}'
                    metric_dict = {approach: self._run_industrial_versus_sota()}

                for approach, metric in metric_dict.items():
                    basic_results.loc[dataset_name, approach] = metric

                os.makedirs(self.result_dir, exist_ok=True)
                basic_results.to_csv(self.comparison_file_path)
            except Exception:
                self.logger.exception(f'Evaluation failed - Dataset: {dataset_name}')

        self.logger.info('Benchmark run finished')

    def finetune(self):
        self._ensure_legacy_dependencies()
        # TODO: fix finetune method, set valid paths and refactor
        self.logger.info('Benchmark finetune started')
        dataset_result = {}
        for dataset_name in self.custom_datasets:
            path_to_results = self.result_dir + f'/{dataset_name}'
            composed_model_path = [
                path_to_results +
                f'/{x}' for x in os.listdir(path_to_results) if x.__contains__('pipeline_saved')]
            metric_result = {}
            for p in composed_model_path:
                if os.path.isdir(p):
                    try:
                        self.experiment_setup['compute_config']['output_folder'] = self.result_dir
                        experiment_setup = deepcopy(self.experiment_setup)
                        prediction, model = self.finetune_loop(
                            dataset_name, experiment_setup, p)
                        metric_result.update({p:
                                              {'metric': Accuracy(model.predict_data.target,
                                                                  prediction.ravel()).metric(),
                                               'tuned_model': model}})
                    except ModuleNotFoundError as ex:
                        print(f'{ex}.OLD VERSION OF PIPELINE. DELETE DIRECTORY')
                        if len(composed_model_path) != 1:
                            print(f'OLD VERSION OF PIPELINE. DELETE DIRECTORY')
                            shutil.rmtree(p)
                        else:
                            print(
                                f'OLD VERSION OF PIPELINE. IT IS A LAST SAVED MODEL')
                else:
                    print(f"No composed model for dataset - {dataset_name}")
            dataset_path = os.path.join(
                self.experiment_setup['compute_config']['output_folder'],
                f'{dataset_name}',
                'metrics_report.csv')
            fedot_results = pd.read_csv(dataset_path, index_col=0)
            if len(metric_result) != 0:
                best_metric = 0
                for _ in metric_result.keys():
                    if best_metric == 0:
                        best_metric, best_model, path = metric_result[_][
                            'metric'], metric_result[_]['tuned_model'], _
                    elif metric_result[_]['metric'] > best_metric:
                        best_metric, best_model, path = metric_result[_][
                            'metric'], metric_result[_]['tuned_model'], _
                fedot_results.loc[dataset_name,
                                  'Fedot_Industrial_finetuned'] = best_metric
                best_model.output_folder = f'{_}_tuned'
                best_model.save_best_model()
                fedot_results.to_csv(dataset_path)
            else:
                fedot_results.to_csv(dataset_path)
            gc.collect()
            dataset_result.update({dataset_name: metric_result})
        self.logger.info("Benchmark finetune finished")

    def load_local_basic_results(self):
        self._ensure_legacy_dependencies()
        try:
            results = pd.read_csv(self.comparison_file_path, index_col=0)
        except Exception as e:
            self.logger.info(f'Unable to load local benchmark results from {self.comparison_file_path} file due to {e}')
            results = self.results_picker.run(get_metrics_df=True, add_info=True)
        return results

    def create_report(self):
        self._ensure_legacy_dependencies()
        # TODO: fix create_report method, set valid paths and refactor
        _ = []
        names = []
        for dataset_name in self.custom_datasets:
            model_result_path = PROJECT_PATH + self.result_dir_name + \
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
        stacked_resutls = np.stack(_, axis=1).T
        df_res = pd.DataFrame(stacked_resutls, index=names)
        df_res.columns = ['Fedot_Industrial', 'Fedot_Industrial_finetuned']
        del df['Fedot_Industrial'], df['Fedot_Industrial_finetuned']
        df = df.join(df_res)
        df = df.fillna(0)
        return df
