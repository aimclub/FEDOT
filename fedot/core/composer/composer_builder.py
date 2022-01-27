from typing import Optional, Union, List, Dict

from deap import tools

from fedot.core.composer.advisor import PipelineChangeAdvisor
from fedot.core.composer.composer import Composer
from fedot.core.composer.gp_composer.gp_composer import GPComposer, PipelineComposerRequirements
from fedot.core.composer.gp_composer.specific_operators import boosting_mutation, parameter_change_mutation
from fedot.core.optimisers.adapters import PipelineAdapter
from fedot.core.optimisers.gp_comp.gp_optimiser import EvoGraphOptimiser, GPGraphOptimiserParameters
from fedot.core.optimisers.gp_comp.operators.inheritance import GeneticSchemeTypesEnum
from fedot.core.optimisers.gp_comp.operators.mutation import MutationTypesEnum
from fedot.core.optimisers.gp_comp.operators.regularization import RegularizationTypesEnum
from fedot.core.optimisers.gp_comp.param_free_gp_optimiser import EvoGraphParameterFreeOptimiser
from fedot.core.optimisers.optimizer import GraphOptimiser, GraphOptimiserParameters, GraphGenerationParams
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.operation_types_repository import get_operations_for_task
from fedot.core.repository.quality_metrics_repository import MetricsEnum, ClassificationMetricsEnum, \
    RegressionMetricsEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum


class ComposerBuilder:
    def __init__(self, task: Task):
        self.optimizer_external_parameters = {}
        self._composer = GPComposer()
        self.optimiser = EvoGraphOptimiser
        self.optimiser_parameters = GPGraphOptimiserParameters()
        self.task = task
        self.set_default_composer_params()

    def with_optimiser(self, optimiser: Optional[GraphOptimiser] = None,
                       parameters: Optional[GraphOptimiserParameters] = None,
                       optimizer_external_parameters: Optional[Dict] = None):
        if optimiser is not None:
            self.optimiser = optimiser
        if parameters is not None:
            self.optimiser_parameters = parameters
        if optimizer_external_parameters is not None:
            self.optimizer_external_parameters = optimizer_external_parameters
        return self

    def with_requirements(self, requirements: PipelineComposerRequirements):
        self._composer.composer_requirements = requirements
        return self

    def with_metrics(self, metrics: Union[List[MetricsEnum], MetricsEnum]):
        if type(metrics) is not list:
            metrics = [metrics]
        self._composer.metrics = metrics
        return self

    def with_initial_pipelines(self, initial_pipelines: Optional[Pipeline]):
        self._composer.initial_pipelines = initial_pipelines
        return self

    def with_logger(self, logger):
        self._composer.log = logger
        return self

    def with_cache(self, cache_path: str = None, use_existing=False):
        self._composer.cache_path = cache_path
        self._composer.use_existing_cache = use_existing
        return self

    def set_default_composer_params(self):
        """ Method set metrics and composer requirements """
        if not self._composer.composer_requirements:
            # Get all available operations for task
            operations = get_operations_for_task(task=self.task, mode='all')

            # Set protected attributes to composer
            self._composer.composer_requirements = PipelineComposerRequirements(primary=operations,
                                                                                secondary=operations)
        if not self._composer.metrics:
            metric_function = ClassificationMetricsEnum.ROCAUC_penalty
            if self.task.task_type in (TaskTypesEnum.regression, TaskTypesEnum.ts_forecasting):
                metric_function = RegressionMetricsEnum.RMSE

            # Set metric
            self._composer.metrics = [metric_function]

    def build(self) -> Composer:
        optimiser_type = self.optimiser
        if (optimiser_type == EvoGraphOptimiser and
                self.optimiser_parameters.genetic_scheme_type == GeneticSchemeTypesEnum.parameter_free):
            optimiser_type = EvoGraphParameterFreeOptimiser

        graph_generation_params = GraphGenerationParams(adapter=PipelineAdapter(self._composer.log),
                                                        advisor=PipelineChangeAdvisor())
        if len(self._composer.metrics) > 1:
            self.optimiser_parameters.archive_type = tools.ParetoFront()
            # TODO add possibility of using regularization in MO alg
            self.optimiser_parameters.regularization_type = RegularizationTypesEnum.none
            self.optimiser_parameters.multi_objective = True

        if self.optimiser_parameters.mutation_types is None:
            self.optimiser_parameters.mutation_types = [boosting_mutation, parameter_change_mutation,
                                                        MutationTypesEnum.single_edge,
                                                        MutationTypesEnum.single_change,
                                                        MutationTypesEnum.single_drop,
                                                        MutationTypesEnum.single_add]

        optimiser = optimiser_type(initial_graph=self._composer.initial_pipelines,
                                   requirements=self._composer.composer_requirements,
                                   graph_generation_params=graph_generation_params,
                                   parameters=self.optimiser_parameters, log=self._composer.log,
                                   metrics=self._composer.metrics,
                                   **self.optimizer_external_parameters)

        self._composer.optimiser = optimiser

        return self._composer
