import logging
from pathlib import Path
from typing import Union, Callable, List

from fedot.core.repository.tasks import TsForecastingParams
from joblib import cpu_count
from fedot.industrial.api.utils.api_init_rules import (
    build_api_manager_state_plan,
    build_industrial_context_plan,
    build_learning_loss_plan,
    resolve_initial_assumption_problem,
)
from fedot.industrial.api.utils.industrial_strategy import IndustrialStrategy
from fedot.industrial.core.architecture.preprocessing.data_convertor import ApiConverter
from fedot.industrial.core.optimizer.FedotEvoOptimizer import FedotEvoOptimizer
from fedot.industrial.core.optimizer.IndustrialEvoOptimizer import IndustrialEvoOptimizer
from fedot.industrial.core.repository.constanst_repository import \
    fedot_init_assumptions, FEDOT_INDUSTRIAL_STRATEGY
from fedot.industrial.core.repository.model_repository import default_industrial_availiable_operation
from fedot.industrial.tools.explain.explain import PointExplainer, RecurrenceExplainer
from fedot.industrial.tools.serialisation.path_lib import DEFAULT_PATH_RESULTS as default_path_to_save_results


class ConfigTemplate:
    def __init__(self):
        self.keys = {}
        self.config = {}

    def build(self, config: dict = None):
        for key, method in self.keys.items():
            val = method(config[key]) if key in config.keys() else method()
            self.config.update({key: val})
        return self


class IndustrialConfig(ConfigTemplate):
    def __init__(self):
        super().__init__()
        self.keys = {'default_fedot_context': self.with_default_fedot_context,
                     'regression_context': self.with_regression_context,
                     'forecasting_context': self.with_forecasting_context,
                     'initial_assumption': self.with_industrial_initial_assumption,
                     'optimizer': self.with_industrial_optimizer,
                     'use_input_preprocessing': self.with_input_preprocessing,
                     'strategy_params': self.with_industrial_strategy_params
                     }
        self.explain_methods = {'point': PointExplainer,
                                'recurrence': RecurrenceExplainer,
                                'shap': NotImplementedError,
                                'lime': NotImplementedError}
        self.regression_tasks = ['ts_forecasting', 'regression']
        self.custom_industrial_strategy = FEDOT_INDUSTRIAL_STRATEGY

    def with_default_fedot_context(self, kwargs):
        context_plan = build_industrial_context_plan(
            problem=kwargs['problem'],
            strategy=kwargs.get('strategy', 'default'),
            task_params=kwargs.get('task_params', {}),
            regression_tasks=self.regression_tasks,
        )
        self.strategy = context_plan.strategy_name
        self.is_default_fedot_context = context_plan.is_default_fedot_context
        return self.is_default_fedot_context

    def with_regression_context(self, kwargs):
        context_plan = build_industrial_context_plan(
            problem=kwargs['problem'],
            strategy=kwargs.get('strategy', 'default'),
            task_params=kwargs.get('task_params', {}),
            regression_tasks=self.regression_tasks,
        )
        self.is_regression_task_context = context_plan.is_regression_task_context
        return self.is_regression_task_context

    def with_industrial_strategy_params(self, kwargs):
        self.strategy_params = kwargs.get('strategy_params', None)
        return self.strategy_params

    def with_forecasting_context(self, kwargs):
        context_plan = build_industrial_context_plan(
            problem=kwargs['problem'],
            strategy=kwargs.get('strategy', 'default'),
            task_params=kwargs.get('task_params', {}),
            regression_tasks=self.regression_tasks,
        )
        self.task_params = context_plan.normalized_task_params or kwargs.get('task_params', {})
        self.is_forecasting_context = context_plan.is_forecasting_context
        return self.is_forecasting_context

    def with_industrial_initial_assumption(self, kwargs):
        self.initial_assumption = kwargs.get('initial_assumption', None)
        problem = resolve_initial_assumption_problem(
            problem=kwargs['problem'],
            strategy_name=self.strategy,
            is_default_fedot_context=self.is_default_fedot_context,
        )
        if self.initial_assumption is None:
            self.initial_assumption = fedot_init_assumptions(problem)

        return self.initial_assumption

    def with_industrial_optimizer(self, kwargs):
        self.industrial_opt = kwargs.get('optimizer', IndustrialEvoOptimizer)
        return self.industrial_opt

    def with_input_preprocessing(self, kwargs):
        self.use_input_preprocessing = kwargs.get('use_input_preprocessing', False)
        return self.use_input_preprocessing

    def build(self, config: dict = None):
        for key, method in self.keys.items():
            val = method(config)
            self.config.update({key: val})
        if self.strategy in FEDOT_INDUSTRIAL_STRATEGY:
            self.strategy = IndustrialStrategy(industrial_strategy=self.strategy,
                                               industrial_strategy_params=self.strategy_params,
                                               api_config=self.config)
        return self


class ComputationalConfig(ConfigTemplate):
    def __init__(self):
        super().__init__()
        self.keys = {'backend': self.with_backend,
                     'distributed': self.with_distributed,
                     'output_folder': self.with_output_folder,
                     'use_cache': self.with_cache,
                     'automl_folder': self.with_automl_folder}
        self.default_dask_params = dict(processes=False,
                                        n_workers=1,
                                        threads_per_worker=round(cpu_count() / 2),
                                        memory_limit=0.3
                                        )

    def with_backend(self, backend: str = 'cpu'):
        self.backend = backend
        return self.backend

    def with_distributed(self, distributed: dict = None):
        self.distributed = distributed if distributed is not None else self.default_dask_params
        return self.distributed

    def with_output_folder(self, output_folder: str = None):
        self.output_folder = output_folder
        return self.output_folder

    def with_cache(self, cache_dict: dict = None):
        self.cache = cache_dict
        return self.cache

    def with_automl_folder(self, automl_folder: str = None):
        self.automl_folder = automl_folder
        return self.automl_folder


class AutomlConfig(ConfigTemplate):
    def __init__(self):
        super().__init__()
        self.keys = {'task': self.with_task,
                     'task_params': self.with_task_params,
                     'initial_assumption': self.with_initial_assumption,
                     'use_automl': self.with_automl,
                     'available_operations': self.with_available_operations,
                     'optimisation_strategy': self.with_optimisation_strategy}

    def with_task(self, task: str = None):
        self.task = task
        return self.task

    def with_task_params(self, task_params: dict = None):
        self.task_params = task_params
        return self.task_params

    def with_initial_assumption(self, initial_assumption: str = None):
        self.initial_assumption = initial_assumption
        return self.initial_assumption

    def with_automl(self, use_automl: bool = False):
        self.use_automl = use_automl
        return self.use_automl

    def with_available_operations(self, available_operations: List[str] = None):
        self.available_operations = available_operations
        if self.available_operations is None:
            self.available_operations = default_industrial_availiable_operation(self.task)
        return self.available_operations

    def with_optimisation_strategy(self, optimisation_strategy: dict = None):
        self.optimisation_strategy = optimisation_strategy
        return self.optimisation_strategy


class LearningConfig(ConfigTemplate):
    def __init__(self):
        super().__init__()
        self.keys = {'learning_strategy': self.with_learning_strategy,
                     'learning_strategy_params': self.with_learning_strategy_params,
                     'optimisation_loss': self.with_loss}

    def with_learning_strategy(self, learning_strategy: str = None):
        self.learning_strategy = learning_strategy
        return self.learning_strategy

    def with_learning_strategy_params(self, learning_strategy_params: dict = None):
        self.learning_strategy_params = learning_strategy_params
        return self.learning_strategy_params

    def with_loss(self, loss: Union[Callable, str, dict] = None):
        loss_plan = build_learning_loss_plan(loss)
        self.quality_loss = loss_plan.quality_loss
        self.computational_loss = loss_plan.computational_loss
        self.structural_loss = loss_plan.structural_loss
        return self.quality_loss


class ApiManager(ConfigTemplate):
    def __init__(self):
        super().__init__()
        self.null_state_object()
        self.logger = logging.getLogger("FedCoreAPI")
        self.keys = {'industrial_config': self.with_industrial_config,
                     'automl_config': self.with_automl_config,
                     'learning_config': self.with_learning_config,
                     'compute_config': self.with_compute_config}
        self.optimisation_agent = {"Industrial": IndustrialEvoOptimizer,
                                   'Fedot': FedotEvoOptimizer}
        self.condition_check = ApiConverter()

    def null_state_object(self):
        state_plan = build_api_manager_state_plan()
        self.solver = state_plan.solver
        self.predicted_labels = state_plan.predicted_labels
        self.predicted_probs = state_plan.predicted_probs
        self.predict_data = state_plan.predict_data
        self.dask_client = state_plan.dask_client
        self.dask_cluster = state_plan.dask_cluster
        self.target_encoder = state_plan.target_encoder
        self.is_finetuned = state_plan.is_finetuned

    def create_folder(self, output_folder):
        # create dirs with results
        output_folder = default_path_to_save_results if output_folder is None else output_folder
        Path(output_folder).mkdir(parents=True, exist_ok=True)

    def with_industrial_config(self, config: dict):
        self.industrial_config = IndustrialConfig().build(config)
        return self.industrial_config

    def with_automl_config(self, config: dict):
        self.automl_config = AutomlConfig().build(config)
        return self.automl_config

    def with_learning_config(self, config: dict):
        self.learning_config = LearningConfig().build(config)
        return self.learning_config

    def with_compute_config(self, config: dict):
        self.compute_config = ComputationalConfig().build(config)
        return self.compute_config

    def build(self, config: dict = None):
        for key, method in self.keys.items():
            if key in config.keys():
                method(config[key])
            else:
                method()
        return self
