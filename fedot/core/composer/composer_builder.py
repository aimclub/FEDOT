import platform
from functools import partial
from multiprocessing import set_start_method
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Type, Union

from fedot.core.caching.pipelines_cache import OperationsCache
from fedot.core.caching.preprocessing_cache import PreprocessingCache
from fedot.core.composer.composer import Composer
from fedot.core.composer.gp_composer.gp_composer import GPComposer
from fedot.core.log import LoggerAdapter, default_log
from fedot.core.optimisers.gp_comp.gp_optimizer import EvoGraphOptimizer, GPGraphOptimizerParameters
from fedot.core.optimisers.gp_comp.operators.regularization import RegularizationTypesEnum
from fedot.core.optimisers.gp_comp.pipeline_composer_requirements import PipelineComposerRequirements
from fedot.core.optimisers.initial_graphs_generator import InitialPopulationGenerator, GenerationFunction
from fedot.core.optimisers.objective.objective import Objective
from fedot.core.optimisers.opt_history import OptHistory, log_to_history
from fedot.core.optimisers.optimizer import GraphOptimizer, GraphOptimizerParameters, GraphGenerationParams
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.pipeline_graph_generation_params import get_pipeline_generation_params
from fedot.core.pipelines.verification import rules_by_task
from fedot.core.repository.operation_types_repository import get_operations_for_task
from fedot.core.repository.quality_metrics_repository import (
    ComplexityMetricsEnum,
    MetricsEnum
)
from fedot.core.repository.tasks import Task
from fedot.core.utilities.data_structures import ensure_wrapped_in_sequence
from fedot.core.utils import default_fedot_data_dir
from fedot.utilities.define_metric_by_task import MetricByTask


def set_multiprocess_start_method():
    system = platform.system()
    if system == 'Linux':
        set_start_method("spawn", force=True)


class ComposerBuilder:

    def __init__(self, task: Task):
        self.log: LoggerAdapter = default_log(self)

        self.task: Task = task
        self.metrics: Sequence[MetricsEnum] = MetricByTask(task.task_type).get_default_quality_metrics()

        self.optimiser_cls: Type[GraphOptimizer] = EvoGraphOptimizer  # default optimizer class
        self.optimiser_parameters: GPGraphOptimizerParameters = GPGraphOptimizerParameters()
        self.optimizer_external_parameters: dict = {}

        self.composer_cls: Type[Composer] = GPComposer  # default composer class
        self.composer_requirements: Optional[PipelineComposerRequirements] = None
        self.graph_generation_params: Optional[GraphGenerationParams] = None

        self.initial_population: Union[Pipeline, Sequence[Pipeline]] = ()
        self.initial_population_generation_function: Optional[GenerationFunction] = None

        self._keep_history: bool = False
        self._full_history_dir: Optional[Path] = None

        self.pipelines_cache: Optional[OperationsCache] = None
        self.preprocessing_cache: Optional[PreprocessingCache] = None

    def with_composer(self, composer_cls: Optional[Type[Composer]]):
        if composer_cls is not None:
            self.composer_cls = composer_cls
        return self

    def with_optimiser(self, optimiser_cls: Optional[Type[GraphOptimizer]] = None):
        if optimiser_cls is not None:
            self.optimiser_cls = optimiser_cls
        return self

    def with_optimiser_params(self, parameters: Optional[GraphOptimizerParameters] = None,
                              external_parameters: Optional[Dict] = None):
        if parameters is not None:
            self.optimiser_parameters = parameters
        if external_parameters is not None:
            self.optimizer_external_parameters = external_parameters
        return self

    def with_requirements(self, requirements: PipelineComposerRequirements):
        self.composer_requirements = requirements
        if self.composer_requirements.max_pipeline_fit_time:
            set_multiprocess_start_method()
        return self

    def with_graph_generation_param(self, graph_generation_params: GraphGenerationParams):
        self.graph_generation_params = graph_generation_params
        return self

    def with_metrics(self, metrics: Union[MetricsEnum, List[MetricsEnum]]):
        self.metrics = ensure_wrapped_in_sequence(metrics)
        return self

    def with_initial_pipelines(self, initial_pipelines: Union[Pipeline, Sequence[Pipeline]]):
        self.initial_population = initial_pipelines
        return self

    def with_initial_pipelines_generation_function(self, generation_function: GenerationFunction):
        self.initial_population_generation_function = generation_function
        return self

    def with_history(self, history_folder: Optional[str] = None):
        self._keep_history = True
        if history_folder:
            history_folder = Path(default_fedot_data_dir(), history_folder)
        self._full_history_dir = history_folder
        return self

    def with_cache(self, pipelines_cache: Optional[OperationsCache] = None,
                   preprocessing_cache: Optional[PreprocessingCache] = None):
        self.pipelines_cache = pipelines_cache
        self.preprocessing_cache = preprocessing_cache
        return self

    @staticmethod
    def _get_default_composer_params(task: Task) -> PipelineComposerRequirements:
        # Get all available operations for task
        operations = get_operations_for_task(task=task, mode='all')
        return PipelineComposerRequirements(primary=operations, secondary=operations)

    def _get_default_graph_generation_params(self) -> GraphGenerationParams:
        return get_pipeline_generation_params(
            rules_for_constraint=rules_by_task(self.task.task_type),
            task=self.task,
            requirements=self.composer_requirements)

    @staticmethod
    def _get_default_complexity_metrics() -> List[MetricsEnum]:
        return [ComplexityMetricsEnum.node_num]

    def build(self) -> Composer:
        if not self.composer_requirements:
            self.composer_requirements = self._get_default_composer_params(self.task)
        if not self.graph_generation_params:
            self.graph_generation_params = self._get_default_graph_generation_params()

        if len(self.metrics) > 1:
            # TODO: avoid post-init of optimiz
            # TODO add possibility of using regularization in MO alg
            self.optimiser_parameters.multi_objective = True
            self.optimiser_parameters.regularization_type = RegularizationTypesEnum.none
        else:
            # Add default complexity metric for supplementary comparison of individuals with equal fitness
            self.optimiser_parameters.multi_objective = False
            self.metrics = self.metrics + self._get_default_complexity_metrics()

        objective = Objective(self.metrics, self.optimiser_parameters.multi_objective)

        initial_population = InitialPopulationGenerator(self.graph_generation_params, self.composer_requirements) \
            .with_initial_graphs(self.initial_population) \
            .with_custom_generation_function(self.initial_population_generation_function)()

        optimiser = self.optimiser_cls(objective=objective,
                                       initial_graphs=initial_population,
                                       requirements=self.composer_requirements,
                                       graph_generation_params=self.graph_generation_params,
                                       parameters=self.optimiser_parameters,
                                       **self.optimizer_external_parameters)
        history = None
        if self._keep_history:
            # Clean results of the previous run
            history = OptHistory(objective)
            history.clean_results(self._full_history_dir)
            history_callback = partial(log_to_history, history=history, save_dir=self._full_history_dir)
            optimiser.set_optimisation_callback(history_callback)

        composer = self.composer_cls(optimiser,
                                     self.composer_requirements,
                                     history,
                                     self.pipelines_cache,
                                     self.preprocessing_cache)

        return composer
