import platform
from dataclasses import dataclass
from functools import partial
from multiprocessing import set_start_method
from typing import (Any, Callable, List, Optional, Tuple, Union)

from deap import tools

from fedot.core.chains.chain import Chain
from fedot.core.chains.chain_validation import validate
from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.composer.cache import OperationsCache
from fedot.core.composer.composer import Composer, ComposerRequirements
from fedot.core.composer.optimisers.gp_comp.gp_optimiser import GPChainOptimiser, GPChainOptimiserParameters
from fedot.core.composer.optimisers.gp_comp.operators.inheritance import GeneticSchemeTypesEnum
from fedot.core.composer.optimisers.gp_comp.operators.mutation import MutationStrengthEnum
from fedot.core.composer.optimisers.gp_comp.operators.regularization import RegularizationTypesEnum
from fedot.core.composer.optimisers.gp_comp.param_free_gp_optimiser import GPChainParameterFreeOptimiser
from fedot.core.data.data import InputData, train_test_data_setup
from fedot.core.log import Log, default_log
from fedot.core.repository.operation_types_repository import OperationTypesRepository, get_operations_for_task
from fedot.core.repository.quality_metrics_repository import ClassificationMetricsEnum, MetricsRepository, \
    RegressionMetricsEnum, MetricsEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum

sample_split_ration_for_tasks = {
    TaskTypesEnum.classification: 0.8,
    TaskTypesEnum.regression: 0.8,
    TaskTypesEnum.ts_forecasting: 0.5
}


def set_multiprocess_start_method():
    system = platform.system()
    if system == 'Linux':
        set_start_method("spawn", force=True)


@dataclass
class GPComposerRequirements(ComposerRequirements):
    """
    Dataclass is for defining the requirements for composition process of genetic programming composer

    :param pop_size: population size
    :param num_of_generations: maximal number of evolutionary algorithm generations
    :param crossover_prob: crossover probability (the chance that two chromosomes exchange some of their parts)
    :param mutation_prob: mutation probability
    :param mutation_strength: strength of mutation in tree (using in certain mutation types)
    :param start_depth: start value of tree depth
    """
    pop_size: Optional[int] = 20
    num_of_generations: Optional[int] = 20
    crossover_prob: Optional[float] = 0.8
    mutation_prob: Optional[float] = 0.8
    mutation_strength: MutationStrengthEnum = MutationStrengthEnum.mean
    start_depth: int = None


@dataclass
class ChainGenerationParams:
    """
    This dataclass is for defining the parameters using in chain generation process

    :param primary_node_func: the function for primary node generation
    :param secondary_node_func: the function for secondary node generation
    :param chain_class: class for the chain object
    """
    primary_node_func: Callable = PrimaryNode
    secondary_node_func: Callable = SecondaryNode
    chain_class: Callable = Chain


class GPComposer(Composer):
    """
    Genetic programming based composer
    :param optimiser: optimiser generated in GPComposerBuilder
    :param metrics: metrics used to define the quality of found solution.
    :param composer_requirements: requirements for composition process
    :param initial_chain: defines the initial state of the population. If None then initial population is random.
    """

    def __init__(self, optimiser=None,
                 composer_requirements: Optional[GPComposerRequirements] = None,
                 metrics: Union[List[MetricsEnum], MetricsEnum] = None,
                 initial_chain: Optional[Chain] = None,
                 logger: Log = None):

        super().__init__(metrics=metrics, composer_requirements=composer_requirements, initial_chain=initial_chain)

        self.cache = OperationsCache()

        self.optimiser = optimiser
        self.cache_path = None
        self.use_existing_cache = False

        if not logger:
            self.log = default_log(__name__)
        else:
            self.log = logger

    def compose_chain(self, data: InputData, is_visualise: bool = False, is_tune: bool = False,
                      on_next_iteration_callback: Optional[Callable] = None) -> Union[Chain, List[Chain]]:
        """ Function for optimal chain structure searching

        :param data: InputData for chain composing
        :param is_visualise: is it needed to visualise
        :param is_tune: is it needed to tune chain after composing TODO integrate new tuner
        :param on_next_iteration_callback: TODO add description

        :return best_chain: obtained result after composing: one chain for single-objective optimization;
            For the multi-objective case, the list of the chain is returned.
            In the list, the chains are ordered by the descending of primary metric (the first is the best)
        """

        if self.composer_requirements.max_chain_fit_time:
            set_multiprocess_start_method()

        if not self.optimiser:
            raise AttributeError(f'Optimiser for chain composition is not defined')

        train_data, test_data = train_test_data_setup(data,
                                                      sample_split_ration_for_tasks[data.task.task_type])
        if self.cache_path is None:
            self.cache.clear()
        else:
            self.cache = OperationsCache(self.cache_path, clear_exiting=not self.use_existing_cache)

        metric_function_for_nodes = partial(self.composer_metric, self.metrics, train_data, test_data)

        best_chain = self.optimiser.optimise(metric_function_for_nodes,
                                             on_next_iteration_callback=on_next_iteration_callback)

        self.log.info('GP composition finished')

        if is_tune:
            self.tune_chain(best_chain, data, self.composer_requirements.max_lead_time)
        return best_chain

    def composer_metric(self, metrics, train_data: InputData,
                        test_data: InputData, chain: Chain) -> Optional[Tuple[Any]]:
        try:
            validate(chain)
            chain.log = self.log

            if type(metrics) is not list:
                metrics = [metrics]

            if self.cache is not None:
                # TODO improve cache
                chain.fit_from_cache(self.cache)

            if not chain.is_fitted():
                self.log.debug(f'Chain {chain.root_node.descriptive_id} fit started')
                chain.fit(input_data=train_data, time_constraint=self.composer_requirements.max_chain_fit_time)
                self.cache.save_chain(chain)

            evaluated_metrics = ()
            for metric in metrics:
                if callable(metric):
                    metric_func = metric
                else:
                    metric_func = MetricsRepository().metric_by_id(metric)
                evaluated_metrics = evaluated_metrics + (metric_func(chain, reference_data=test_data),)

            self.log.debug(f'Chain {chain.root_node.descriptive_id} with metrics: {list(evaluated_metrics)}')

        except Exception as ex:
            self.log.info(f'Chain assessment warning: {ex}. Continue.')
            evaluated_metrics = None

        return evaluated_metrics

    @staticmethod
    def tune_chain(chain: Chain, data: InputData, time_limit):
        raise NotImplementedError()

    @property
    def history(self):
        return self.optimiser.history


class GPComposerBuilder:
    def __init__(self, task: Task):
        self._composer = GPComposer()
        self.optimiser_parameters = GPChainOptimiserParameters()
        self.task = task
        self.set_default_composer_params()

    def can_be_secondary_requirement(self, operation):
        models_repo = OperationTypesRepository()
        data_operations_repo = OperationTypesRepository('data_operation_repository.json')

        operation_name = models_repo.operation_info_by_id(operation)
        if operation_name is None:
            operation_name = data_operations_repo.operation_info_by_id(operation)
        operation_tags = operation_name.tags

        secondary_model = True
        # TODO remove 'data_model'
        if 'data_model' in operation_tags:
            secondary_model = False
        return secondary_model

    def with_optimiser_parameters(self, optimiser_parameters: GPChainOptimiserParameters):
        self.optimiser_parameters = optimiser_parameters
        return self

    def with_requirements(self, requirements: GPComposerRequirements):
        # TODO move this functionality in composer
        requirements.secondary = list(filter(self.can_be_secondary_requirement, requirements.secondary))
        self._composer.composer_requirements = requirements
        return self

    def with_metrics(self, metrics: Union[List[MetricsEnum], MetricsEnum]):
        if type(metrics) is not list:
            metrics = [metrics]
        self._composer.metrics = metrics
        return self

    def with_initial_chain(self, initial_chain: Optional[Chain]):
        self._composer.initial_chain = initial_chain
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
            self._composer.composer_requirements = GPComposerRequirements(primary=operations, secondary=operations)
        if not self._composer.metrics:
            metric_function = ClassificationMetricsEnum.ROCAUC_penalty
            if self.task.task_type in (TaskTypesEnum.regression, TaskTypesEnum.ts_forecasting):
                metric_function = RegressionMetricsEnum.RMSE

            # Set metric
            self._composer.metrics = [metric_function]

    def build(self) -> Composer:
        optimiser_type = GPChainOptimiser
        if self.optimiser_parameters.genetic_scheme_type == GeneticSchemeTypesEnum.parameter_free:
            optimiser_type = GPChainParameterFreeOptimiser

        chain_generation_params = ChainGenerationParams()

        archive_type = None
        if len(self._composer.metrics) > 1:
            archive_type = tools.ParetoFront()
            # TODO add possibility of using regularization in MO alg
            self.optimiser_parameters.regularization_type = RegularizationTypesEnum.none
            self.optimiser_parameters.multi_objective = True

        optimiser = optimiser_type(initial_chain=self._composer.initial_chain,
                                   requirements=self._composer.composer_requirements,
                                   chain_generation_params=chain_generation_params,
                                   parameters=self.optimiser_parameters, log=self._composer.log,
                                   archive_type=archive_type, metrics=self._composer.metrics)

        self._composer.optimiser = optimiser

        return self._composer
