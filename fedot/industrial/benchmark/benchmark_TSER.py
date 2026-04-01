import logging
import os
import warnings
from abc import ABC
from copy import deepcopy

import pandas as pd

from benchmark.v2.api import run_tser_benchmark_from_legacy_config

LEGACY_IMPORT_ERROR = None

try:
    import matplotlib
    from fedot.core.pipelines.node import PipelineNode
    from fedot.core.pipelines.pipeline import Pipeline

    from benchmark.abstract_bench import AbstractBenchmark
    from fedot_ind.api.main import FedotIndustrial
    from fedot_ind.core.architecture.pipelines.abstract_pipeline import ApiTemplate
    from fedot_ind.core.architecture.postprocessing.results_picker import ResultsPicker
    from fedot_ind.core.metrics.metrics_implementation import RMSE
    from fedot_ind.core.repository.constanst_repository import MULTI_REG_BENCH
    from fedot_ind.tools.loader import DataLoader
    from fedot_ind.tools.serialisation.path_lib import PROJECT_PATH

    matplotlib.use('TkAgg')
except Exception as exc:  # pragma: no cover - legacy-only fallback
    LEGACY_IMPORT_ERROR = exc


    class AbstractBenchmark:
        def __init__(self, output_dir=None):
            self.output_dir = output_dir


    class ResultsPicker:
        def __init__(self, path=None):
            self.path = path

        def run(self, *args, **kwargs):
            return pd.DataFrame()


    matplotlib = None
    PipelineNode = None
    Pipeline = None
    FedotIndustrial = None
    ApiTemplate = None
    RMSE = None
    MULTI_REG_BENCH = []
    DataLoader = None
    PROJECT_PATH = os.getcwd()


class BenchmarkTSER(AbstractBenchmark, ABC):
    def __init__(self,
                 experiment_setup: dict = None,
                 custom_datasets: list = None,
                 use_small_datasets: bool = False):

        super(BenchmarkTSER, self).__init__(
            output_dir='./tser/benchmark_results')

        self.logger = logging.getLogger(self.__class__.__name__)

        self.experiment_setup = experiment_setup
        self.init_assumption = deepcopy(self.experiment_setup['initial_assumption'])
        self.monash_regression = MULTI_REG_BENCH
        if custom_datasets is None:
            self.custom_datasets = self.monash_regression
        else:
            self.custom_datasets = custom_datasets
        self.use_small_datasets = use_small_datasets
        self.results_picker = ResultsPicker(
            path=os.path.abspath(self.output_dir))

    @staticmethod
    def _ensure_legacy_dependencies():
        if LEGACY_IMPORT_ERROR is not None:
            raise ImportError(
                'Legacy BenchmarkTSER dependencies are unavailable in this environment. '
                'Use `use_benchmark_v2=True` or install the legacy stack.'
            ) from LEGACY_IMPORT_ERROR

    def _run_model_versus_model(self, dataset_name, comparasion_dict):
        self._ensure_legacy_dependencies()
        approach_dict = {}
        for approach in comparasion_dict.keys():
            result_dict = ApiTemplate(api_config=self.experiment_setup,
                                      metric_list=self.experiment_setup['metric_names']). \
                eval(dataset=dataset_name,
                     initial_assumption=comparasion_dict[approach],
                     finetune=self.experiment_setup['finetune'])
            metric = result_dict['metrics'][self.experiment_setup['metric']][0]
            approach_dict.update({approach: metric})
        return approach_dict

    def _run_industrial_versus_sota(self, dataset_name):
        self._ensure_legacy_dependencies()
        experiment_setup = deepcopy(self.experiment_setup)
        prediction, target = self.evaluate_loop(dataset_name, experiment_setup)
        return RMSE(target, prediction).metric()

    def run(self):
        if self.experiment_setup.get('use_benchmark_v2'):
            warnings.warn(
                'BenchmarkTSER.run is delegating to benchmark.v2. '
                'Use benchmark.v2.run_tser_benchmark_suite directly for new code.',
                DeprecationWarning,
                stacklevel=2,
            )
            return run_tser_benchmark_from_legacy_config(self.experiment_setup)

        self._ensure_legacy_dependencies()
        self.logger.info('Benchmark test started')
        basic_results = self.load_local_basic_results()
        metric_dict = {}
        for dataset_name in self.custom_datasets:
            try:
                if isinstance(self.init_assumption, dict):
                    model_name = list(self.init_assumption.keys())
                    metric = self._run_model_versus_model(dataset_name, self.init_assumption)
                else:
                    metric = self._run_industrial_versus_sota(dataset_name)
                    model_name = 'Fedot_Industrial'
                metric_dict.update({dataset_name: metric})
                basic_results.loc[dataset_name, model_name] = metric
                basic_path = os.path.join(self.experiment_setup['output_folder'])
                if not os.path.exists(basic_path):
                    os.makedirs(basic_path)
                basic_results.to_csv(os.path.join(basic_path, 'comprasion_metrics_report.csv'))
            except Exception:
                self.logger.info(f"{dataset_name} problem with eval")
        self.logger.info("Benchmark test finished")

    def load_local_basic_results(self, path: str = None):
        self._ensure_legacy_dependencies()
        if path is None:
            path = PROJECT_PATH + '/benchmark/results/time_series_multi_reg_comparasion.csv'
            results = pd.read_csv(path, sep=';', index_col=0)
            results = results.dropna(axis=1, how='all')
            results = results.dropna(axis=0, how='all')
            self.experiment_setup['output_folder'] = PROJECT_PATH + \
                '/benchmark/results/ts_regression'
            return results
        else:
            return self.results_picker.run(get_metrics_df=True, add_info=True)

    def finetune(self):
        self._ensure_legacy_dependencies()
        for dataset_name in self.custom_datasets:
            experiment_setup = deepcopy(self.experiment_setup)
            path_to_results = PROJECT_PATH + \
                '/benchmark/results/ts_regression' + f'/{dataset_name}'
            composed_model_path = [
                path_to_results +
                f'/{x}' for x in os.listdir(path_to_results) if x.__contains__('pipeline_saved')]
            for p in composed_model_path:
                experiment_setup['output_folder'] = path_to_results
                prediction, model = self.finetune_loop(
                    dataset_name, experiment_setup, p)
                metric = RMSE(model.predict_data.target, prediction).metric()
                metric_path = PROJECT_PATH + '/benchmark/results/ts_regression' + \
                    f'/{dataset_name}' + '/metrics_report.csv'
                fedot_results = pd.read_csv(metric_path, index_col=0)
                fedot_results.loc[dataset_name,
                                  'Fedot_Industrial_finetuned'] = metric
                fedot_results.to_csv(metric_path)
                model.save_best_model()
        self.logger.info("Benchmark finetune finished")

    def finetune_loop(self, dataset, experiment_setup, composed_model_path):
        self._ensure_legacy_dependencies()
        train_data, test_data = DataLoader(dataset_name=dataset).load_data()
        if 'tuning_params' in experiment_setup.keys():
            tuning_params = experiment_setup['tuning_params']
            del experiment_setup['tuning_params']
        else:
            tuning_params = None
        model = FedotIndustrial(**experiment_setup)
        model.load(path=composed_model_path)

        model.finetune(train_data, tuning_params)
        prediction = model.predict(test_data)
        return prediction, model

    def show_composite_pipeline(self):
        self._ensure_legacy_dependencies()
        for dataset_name in self.custom_datasets:
            composed_model_path = PROJECT_PATH + '/benchmark/results/ts_regression' + \
                f'/{dataset_name}' + '/0_pipeline_saved'
            experiment_setup = deepcopy(self.experiment_setup)
            experiment_setup['output_folder'] = composed_model_path
            del experiment_setup['industrial_preprocessing']
            model = FedotIndustrial(**experiment_setup)
            model.load(path=composed_model_path)
            batch_pipelines = [automl_branch.fitted_operation.model.current_pipeline
                               for automl_branch in model.current_pipeline.nodes if automl_branch.name == 'fedot_regr']
            pr = PipelineNode('ridge', nodes_from=[
                              p.root_node for p in batch_pipelines])
            composed_pipeline = Pipeline(pr)
            composed_pipeline.show()
