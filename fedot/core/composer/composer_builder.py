from typing import Optional, Union, List, Dict, Sequence, Type, Iterable

from fedot.core.composer.advisor import PipelineChangeAdvisor
from fedot.core.composer.cache import OperationsCache
from fedot.core.composer.composer import Composer
from fedot.core.composer.gp_composer.gp_composer import GPComposer, PipelineComposerRequirements
from fedot.core.log import Log
from fedot.core.optimisers.adapters import PipelineAdapter
from fedot.core.optimisers.gp_comp.gp_optimiser import EvoGraphOptimiser, GPGraphOptimiserParameters
from fedot.core.optimisers.gp_comp.operators.inheritance import GeneticSchemeTypesEnum
from fedot.core.optimisers.gp_comp.operators.regularization import RegularizationTypesEnum
from fedot.core.optimisers.optimizer import GraphGenerationParams, GraphOptimiser, GraphOptimiserParameters
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.operation_types_repository import get_operations_for_task
from fedot.core.repository.quality_metrics_repository import (
    MetricsEnum,
    ClassificationMetricsEnum,
    RegressionMetricsEnum,
    ComplexityMetricsEnum
)
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.utilities.data_structures import ensure_wrapped_in_sequence
from fedot.core.optimisers.objective.objective import Objective


class ComposerBuilder:

    def __init__(self, task: Task):
        self.task: Task = task

        self.optimiser_cls: Type[GraphOptimiser] = EvoGraphOptimiser
        self.optimiser_parameters: GraphOptimiserParameters = GPGraphOptimiserParameters()
        self.optimizer_external_parameters: dict = {}

        self.composer_cls: Type[Composer] = GPComposer
        self.initial_pipelines: Optional[Sequence[Pipeline]] = None
        self.log: Optional[Log] = None
        self.cache: Optional[OperationsCache] = None
        self.composer_requirements: PipelineComposerRequirements = self._get_default_composer_params()
        self.metrics: Sequence[MetricsEnum] = self._get_default_quality_metrics(task)

    def with_optimiser(self, optimiser_cls: Optional[Type[GraphOptimiser]] = None):
        if optimiser_cls is not None:
            self.optimiser_cls = optimiser_cls

    def with_optimiser_params(self, parameters: Optional[GraphOptimiserParameters] = None,
                              external_parameters: Optional[Dict] = None):
        if parameters is not None:
            self.optimiser_parameters = parameters
        if external_parameters is not None:
            self.optimizer_external_parameters = external_parameters
        return self

    def with_requirements(self, requirements: PipelineComposerRequirements):
        self.composer_requirements = requirements
        return self

    def with_metrics(self, metrics: Union[MetricsEnum, List[MetricsEnum]]):
        self.metrics = ensure_wrapped_in_sequence(metrics)
        return self

    def with_initial_pipelines(self, initial_pipelines: Optional[Union[Pipeline, Sequence[Pipeline]]]):
        if isinstance(initial_pipelines, Pipeline):
            self.initial_pipelines = [initial_pipelines]
        elif isinstance(initial_pipelines, Iterable):
            self.initial_pipelines = list(initial_pipelines)
        else:
            raise ValueError(f'Incorrect type of initial_assumption: '
                             f'Sequence[Pipeline] or Pipeline needed, but has {type(initial_pipelines)}')
        return self

    def with_logger(self, logger):
        self.log = logger
        return self

    def with_cache(self, cache: Optional[OperationsCache]):
        self.cache = cache
        return self

    def _get_default_composer_params(self) -> PipelineComposerRequirements:
        # Get all available operations for task
        operations = get_operations_for_task(task=self.task, mode='all')
        return PipelineComposerRequirements(primary=operations, secondary=operations)

    @staticmethod
    def _get_default_quality_metrics(task: Task) -> List[MetricsEnum]:
        # Set metrics
        metric_function = ClassificationMetricsEnum.ROCAUC_penalty
        if task.task_type in (TaskTypesEnum.regression, TaskTypesEnum.ts_forecasting):
            metric_function = RegressionMetricsEnum.RMSE
        return [metric_function]

    @staticmethod
    def _get_default_complexity_metrics() -> List[MetricsEnum]:
        return [ComplexityMetricsEnum.node_num]

    def build(self) -> Composer:
        graph_generation_params = GraphGenerationParams(adapter=PipelineAdapter(self.log),
                                                        advisor=PipelineChangeAdvisor(self.task))

        if len(self.metrics) > 1:
            # TODO add possibility of using regularization in MO alg
            self.optimiser_parameters.regularization_type = RegularizationTypesEnum.none
            self.optimiser_parameters.multi_objective = True
        else:
            # Add default complexity metric for supplementary comparison of individuals with equal fitness
            self.metrics = self.metrics + self._get_default_complexity_metrics()
            self.optimiser_parameters.multi_objective = False

        objective = Objective(self.metrics, self.optimiser_parameters.multi_objective, log=self.log)

        optimiser = self.optimiser_cls(objective=objective,
                                       initial_graph=self.initial_pipelines,
                                       requirements=self.composer_requirements,
                                       graph_generation_params=graph_generation_params,
                                       parameters=self.optimiser_parameters,
                                       log=self.log,
                                       **self.optimizer_external_parameters)

        composer = self.composer_cls(optimiser,
                                     self.composer_requirements,
                                     self.initial_pipelines,
                                     self.log,
                                     self.cache)

        return composer
