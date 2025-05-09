import platform
from multiprocessing import set_start_method
from pathlib import Path
from typing import List, Optional, Sequence, Type, Union

from golem.core.log import LoggerAdapter, default_log
from golem.core.optimisers.genetic.gp_optimizer import EvoGraphOptimizer
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.initial_graphs_generator import GenerationFunction, InitialPopulationGenerator
from golem.core.optimisers.optimizer import AlgorithmParameters, GraphGenerationParams, GraphOptimizer
from golem.utilities.data_structures import ensure_wrapped_in_sequence

from fedot.core.caching.operations_cache import OperationsCache
from fedot.core.caching.preprocessing_cache import PreprocessingCache
from fedot.core.caching.predictions_cache import PredictionsCache
from fedot.core.composer.composer import Composer
from fedot.core.composer.gp_composer.gp_composer import GPComposer
from fedot.core.optimisers.objective.metrics_objective import MetricsObjective
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.pipeline_composer_requirements import PipelineComposerRequirements
from fedot.core.pipelines.pipeline_graph_generation_params import get_pipeline_generation_params
from fedot.core.pipelines.verification import rules_by_task
from fedot.core.repository.operation_types_repository import get_operations_for_task
from fedot.core.repository.metrics_repository import ComplexityMetricsEnum, MetricIDType, MetricsEnum
from fedot.core.repository.tasks import Task
from fedot.remote.remote_evaluator import RemoteEvaluator
from fedot.utilities.define_metric_by_task import MetricByTask


def set_multiprocess_start_method():
    system = platform.system()
    if system == 'Linux':
        set_start_method("spawn", force=True)


class ComposerBuilder:

    def __init__(self, task: Task):
        self.log: LoggerAdapter = default_log(self)

        self.task: Task = task
        self.metrics: Sequence[MetricsEnum] = MetricByTask.get_default_quality_metrics(task.task_type)

        self.optimizer_cls: Type[GraphOptimizer] = EvoGraphOptimizer  # default optimizer class
        self.optimizer_parameters: Optional[AlgorithmParameters] = None

        self.composer_cls: Type[Composer] = GPComposer  # default composer class
        self.composer_requirements: Optional[PipelineComposerRequirements] = None
        self.graph_generation_params: Optional[GraphGenerationParams] = None

        self.initial_population: Union[Pipeline, Sequence[Pipeline]] = ()
        self.initial_population_generation_function: Optional[GenerationFunction] = None

        self._keep_history: bool = True
        self._full_history_dir: Optional[Path] = None

        self.operations_cache: Optional[OperationsCache] = None
        self.preprocessing_cache: Optional[PreprocessingCache] = None
        self.predictions_cache: Optional[PredictionsCache] = None

    def with_composer(self, composer_cls: Optional[Type[Composer]]):
        if composer_cls is not None:
            self.composer_cls = composer_cls
        return self

    def with_optimizer(self, optimizer_cls: Optional[Type[GraphOptimizer]]):
        if optimizer_cls:
            self.optimizer_cls = optimizer_cls
        return self

    def with_optimizer_params(self, parameters: Optional[AlgorithmParameters] = None,
                              dispatcher=None):
        if parameters is not None:
            self.optimizer_parameters = parameters
        if dispatcher is not None:
            self.optimizer_parameters = dispatcher
        return self

    def with_requirements(self, requirements: PipelineComposerRequirements):
        self.composer_requirements = requirements
        if self.composer_requirements.max_graph_fit_time:
            set_multiprocess_start_method()
        return self

    def with_graph_generation_param(self, graph_generation_params: GraphGenerationParams):
        self.graph_generation_params = graph_generation_params
        return self

    def with_metrics(self, metrics: Union[MetricIDType, Sequence[MetricIDType]]):
        self.metrics = ensure_wrapped_in_sequence(metrics)
        return self

    def with_initial_pipelines(self, initial_pipelines: Union[Pipeline, Sequence[Pipeline]]):
        self.initial_population = initial_pipelines
        return self

    def with_initial_pipelines_generation_function(self, generation_function: GenerationFunction):
        self.initial_population_generation_function = generation_function
        return self

    def with_cache(self,
                   operations_cache: Optional[OperationsCache] = None,
                   preprocessing_cache: Optional[PreprocessingCache] = None,
                   predictions_cache: Optional[PredictionsCache] = None
                   ):
        self.operations_cache = operations_cache
        self.preprocessing_cache = preprocessing_cache
        self.predictions_cache = predictions_cache
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
            requirements=self.composer_requirements
        )

    @staticmethod
    def _get_default_complexity_metrics() -> List[MetricsEnum]:
        return [ComplexityMetricsEnum.node_number]

    def build(self) -> Composer:
        multi_objective = len(self.metrics) > 1
        if not self.composer_requirements:
            self.composer_requirements = self._get_default_composer_params(self.task)
        if not self.graph_generation_params:
            self.graph_generation_params = self._get_default_graph_generation_params()
        if not self.optimizer_parameters:
            self.optimizer_parameters = GPAlgorithmParameters(multi_objective=multi_objective)
        if not multi_objective:
            # Add default complexity metric for supplementary comparison of individuals with equal fitness
            self.metrics = self.metrics + self._get_default_complexity_metrics()
        if RemoteEvaluator().is_enabled:
            # This explicit passing of singleton evaluator isolates optimizer
            #  from knowing about it. Needed for separation of FEDOT from GOLEM.
            #  Best solution would be to avoid singleton init & access altogether,
            #  instead passing remote evaluator & its params explicitly through API.
            self.graph_generation_params.remote_evaluator = RemoteEvaluator()

        objective = MetricsObjective(self.metrics, multi_objective)

        initial_population = InitialPopulationGenerator(self.optimizer_parameters.pop_size,
                                                        self.graph_generation_params,
                                                        self.composer_requirements) \
            .with_initial_graphs(self.initial_population) \
            .with_custom_generation_function(self.initial_population_generation_function)()

        optimiser = self.optimizer_cls(objective=objective,
                                       initial_graphs=initial_population,
                                       requirements=self.composer_requirements,
                                       graph_generation_params=self.graph_generation_params,
                                       graph_optimizer_params=self.optimizer_parameters)

        composer = self.composer_cls(optimiser,
                                     self.composer_requirements,
                                     self.operations_cache,
                                     self.preprocessing_cache,
                                     self.predictions_cache)

        return composer
