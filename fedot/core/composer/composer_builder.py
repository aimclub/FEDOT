from typing import Optional, Union, List, Dict, Sequence

from fedot.core.composer.advisor import PipelineChangeAdvisor
from fedot.core.composer.cache import OperationsCache
from fedot.core.composer.composer import Composer
from fedot.core.composer.gp_composer.gp_composer import GPComposer, PipelineComposerRequirements
from fedot.core.composer.gp_composer.specific_operators import boosting_mutation, parameter_change_mutation
from fedot.core.optimisers.adapters import PipelineAdapter
from fedot.core.optimisers.gp_comp.gp_optimiser import EvoGraphOptimiser, GPGraphOptimiserParameters
from fedot.core.optimisers.gp_comp.operators.inheritance import GeneticSchemeTypesEnum
from fedot.core.optimisers.gp_comp.operators.mutation import MutationTypesEnum
from fedot.core.optimisers.gp_comp.operators.regularization import RegularizationTypesEnum
from fedot.core.optimisers.gp_comp.param_free_gp_optimiser import EvoGraphParameterFreeOptimiser
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


class ComposerBuilder:

    def __init__(self, task: Task):
        self.task = task

        self.optimiser_cls = EvoGraphOptimiser
        self.optimiser_parameters = GPGraphOptimiserParameters()
        self.optimizer_external_parameters = {}

        self.composer_cls = GPComposer
        self.initial_pipelines = None
        self.log = None
        self.cache = None
        self.composer_requirements = self._get_default_composer_params()
        self.metrics = self._get_default_quality_metrics(task)

    def with_optimiser(self, optimiser: Optional[GraphOptimiser] = None,
                       parameters: Optional[GraphOptimiserParameters] = None,
                       optimizer_external_parameters: Optional[Dict] = None):
        if optimiser is not None:
            self.optimiser_cls = optimiser
        if parameters is not None:
            self.optimiser_parameters = parameters
        if optimizer_external_parameters is not None:
            self.optimizer_external_parameters = optimizer_external_parameters
        return self

    def with_requirements(self, requirements: PipelineComposerRequirements):
        self.composer_requirements = requirements
        return self

    def with_metrics(self, metrics: Union[MetricsEnum, List[MetricsEnum]]):
        self.metrics = ensure_wrapped_in_sequence(metrics)
        return self

    def with_initial_pipelines(self, initial_pipelines: Optional[Pipeline]):
        self.initial_pipelines = initial_pipelines
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
        optimiser_type = self.optimiser_cls
        if (optimiser_type is EvoGraphOptimiser and
                self.optimiser_parameters.genetic_scheme_type is GeneticSchemeTypesEnum.parameter_free):
            optimiser_type = EvoGraphParameterFreeOptimiser

        graph_generation_params = GraphGenerationParams(adapter=PipelineAdapter(self.log),
                                                        advisor=PipelineChangeAdvisor())
        if len(self.metrics) > 1:
            # TODO add possibility of using regularization in MO alg
            self.optimiser_parameters.regularization_type = RegularizationTypesEnum.none
            self.optimiser_parameters.multi_objective = True
        else:
            # Add default complexity metric for supplementary comparison of individuals with equal fitness
            self.metrics = self.metrics + self._get_default_complexity_metrics()
            self.optimiser_parameters.multi_objective = False

        if self.optimiser_parameters.mutation_types is None:
            self.optimiser_parameters.mutation_types = [boosting_mutation, parameter_change_mutation,
                                                        MutationTypesEnum.single_edge,
                                                        MutationTypesEnum.single_change,
                                                        MutationTypesEnum.single_drop,
                                                        MutationTypesEnum.single_add]

        optimiser = optimiser_type(initial_graph=self.initial_pipelines,
                                   requirements=self.composer_requirements,
                                   graph_generation_params=graph_generation_params,
                                   parameters=self.optimiser_parameters,
                                   log=self.log,
                                   metrics=self.metrics,
                                   **self.optimizer_external_parameters)

        composer = self.composer_cls(optimiser,
                                     self.composer_requirements,
                                     self.metrics,
                                     self.initial_pipelines,
                                     self.log,
                                     self.cache)

        return composer
