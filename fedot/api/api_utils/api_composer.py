import datetime
import gc
import traceback
from typing import Callable, List, Optional, Union, Tuple, Collection, Sequence

from fedot.api.api_utils.assumptions.assumptions_handler import AssumptionsHandler
from fedot.api.api_utils.metrics import ApiMetrics
from fedot.api.api_utils.presets import change_preset_based_on_initial_fit, OperationsPreset
from fedot.api.time import ApiTime
from fedot.core.composer.cache import OperationsCache
from fedot.core.composer.composer_builder import ComposerBuilder
from fedot.core.composer.gp_composer.gp_composer import GPComposer, PipelineComposerRequirements
from fedot.core.composer.gp_composer.specific_operators import boosting_mutation, parameter_change_mutation
from fedot.core.constants import DEFAULT_TUNING_ITERATIONS_NUMBER, MINIMAL_SECONDS_FOR_TUNING
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.log import Log
from fedot.core.optimisers.archive import HallOfFame
from fedot.core.optimisers.gp_comp.gp_optimiser import GeneticSchemeTypesEnum, GPGraphOptimiserParameters
from fedot.core.optimisers.gp_comp.operators.crossover import CrossoverTypesEnum
from fedot.core.optimisers.gp_comp.operators.mutation import MutationTypesEnum
from fedot.core.optimisers.opt_history import OptHistory
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.operation_types_repository import get_operations_for_task
from fedot.core.repository.quality_metrics_repository import MetricsRepository, MetricType
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.utilities.define_metric_by_task import MetricByTask, TunerMetricByTask


class ApiComposer:

    def __init__(self, problem: str):
        self.metrics = ApiMetrics(problem)
        self.cache: Optional[OperationsCache] = None
        self.preset_name = None
        self.timer = None

    def obtain_metric(self, task: Task, composer_metric: Union[str, Callable]):
        # the choice of the metric for the pipeline quality assessment during composition
        if composer_metric is None:
            composer_metric = MetricByTask(task.task_type).metric_cls.get_value

        if isinstance(composer_metric, (str, Callable)):
            composer_metric = [composer_metric]

        metric_function = []
        for specific_metric in composer_metric:
            if isinstance(specific_metric, Callable):
                specific_metric_function = specific_metric
            else:
                # Composer metric was defined by name (str)
                metric_id = self.metrics.get_composer_metrics_mapping(metric_name=specific_metric)
                if metric_id is None:
                    raise ValueError(f'Incorrect metric {specific_metric}')
                specific_metric_function = MetricsRepository().metric_by_id(metric_id)
            metric_function.append(specific_metric_function)
        return metric_function

    def obtain_model(self, **common_dict):
        # Prepare parameters
        api_params_dict, composer_params_dict, tuner_params_dict = _divide_parameters(common_dict)
        # Start composing - pipeline structure search
        return self.compose_fedot_model(api_params_dict, composer_params_dict, tuner_params_dict)

    @staticmethod
    def divide_operations(available_operations, task):
        """ Function divide operations for primary and secondary """

        if task.task_type is TaskTypesEnum.ts_forecasting:
            # Get time series operations for primary nodes
            ts_data_operations = get_operations_for_task(task=task,
                                                         mode='data_operation',
                                                         tags=["non_lagged"])
            # Remove exog data operation from the list
            ts_data_operations.remove('exog_ts')

            ts_primary_models = get_operations_for_task(task=task,
                                                        mode='model',
                                                        tags=["non_lagged"])
            # Union of the models and data operations
            ts_primary_operations = ts_data_operations + ts_primary_models

            # Filter - remain only operations, which were in available ones
            primary_operations = list(set(ts_primary_operations).intersection(available_operations))
            secondary_operations = available_operations
        else:
            primary_operations = available_operations
            secondary_operations = available_operations
        return primary_operations, secondary_operations

    def init_cache(self, use_cache: bool):
        if use_cache:
            self.cache = OperationsCache()
            #  in case of previously generated singleton cache
            self.cache.reset()

    @staticmethod
    def _init_composer_requirements(api_params: dict,
                                    composer_params: dict,
                                    datetime_composing: Optional[datetime.timedelta],
                                    preset: str) -> PipelineComposerRequirements:

        task = api_params['task']

        # define available operations
        available_operations = composer_params.get('available_operations',
                                                   OperationsPreset(task, preset).filter_operations_by_preset())
        primary_operations, secondary_operations = \
            ApiComposer.divide_operations(available_operations, task)

        composer_requirements = PipelineComposerRequirements(primary=primary_operations,
                                                             secondary=secondary_operations,
                                                             max_arity=composer_params['max_arity'],
                                                             max_depth=composer_params['max_depth'],
                                                             pop_size=composer_params['pop_size'],
                                                             max_pipeline_fit_time=composer_params[
                                                                 'max_pipeline_fit_time'],
                                                             num_of_generations=composer_params['num_of_generations'],
                                                             cv_folds=composer_params['cv_folds'],
                                                             validation_blocks=composer_params['validation_blocks'],
                                                             timeout=datetime_composing,
                                                             n_jobs=api_params['n_jobs'],
                                                             collect_intermediate_metric=composer_params[
                                                                 'collect_intermediate_metric'])
        return composer_requirements

    @staticmethod
    def _init_optimiser_params(task: Task, composer_params: dict) -> GPGraphOptimiserParameters:

        genetic_scheme_type = GeneticSchemeTypesEnum.parameter_free
        if composer_params['genetic_scheme'] == 'steady_state':
            genetic_scheme_type = GeneticSchemeTypesEnum.steady_state

        mutations = [boosting_mutation, parameter_change_mutation,
                     MutationTypesEnum.single_change,
                     MutationTypesEnum.single_drop,
                     MutationTypesEnum.single_add]
        # TODO remove workaround after validation fix
        if task.task_type is not TaskTypesEnum.ts_forecasting:
            mutations.append(MutationTypesEnum.single_edge)

        optimiser_parameters = GPGraphOptimiserParameters(
            genetic_scheme_type=genetic_scheme_type,
            mutation_types=mutations,
            crossover_types=[CrossoverTypesEnum.one_point, CrossoverTypesEnum.subtree],
            stopping_after_n_generation=composer_params.get('stopping_after_n_generation')
        )
        return optimiser_parameters

    def compose_fedot_model(self, api_params: dict, composer_params: dict, tuning_params: dict) \
            -> Tuple[Pipeline, Sequence[Pipeline], OptHistory]:
        """ Function for composing FEDOT pipeline model """
        log = api_params['logger']
        task = api_params['task']
        train_data = api_params['train_data']
        timeout = api_params['timeout']
        with_tuning = tuning_params['with_tuning']
        available_operations = composer_params['available_operations']
        preset = composer_params['preset']

        self.timer = ApiTime(time_for_automl=timeout, with_tuning=with_tuning)

        # Work with initial assumptions
        assumption_handler = AssumptionsHandler(log, train_data)

        # Set initial assumption and check correctness
        initial_assumption = assumption_handler.propose_assumptions(composer_params['initial_assumption'],
                                                                    available_operations)
        with self.timer.launch_assumption_fit():
            fitted_assumption = assumption_handler.fit_assumption_and_check_correctness(initial_assumption[0],
                                                                                        self.cache)
        log.message(f'Initial pipeline was fitted for {self.timer.assumption_fit_spend_time.total_seconds()} sec.')
        self.preset_name = assumption_handler.propose_preset(preset, self.timer)

        composer_requirements = self._init_composer_requirements(api_params, composer_params,
                                                                 self.timer.timedelta_composing, self.preset_name)

        # Get optimiser, its parameters, and composer
        metric_function = self.obtain_metric(task, composer_params['composer_metric'])

        log.message(f"AutoML configured."
                    f" Parameters tuning: {with_tuning}."
                    f" Time limit: {timeout} min."
                    f" Set of candidate models: {available_operations}.")

        builder = ComposerBuilder(task=task) \
            .with_requirements(composer_requirements) \
            .with_initial_pipelines(initial_assumption) \
            .with_optimiser(composer_params.get('optimizer')) \
            .with_optimiser_params(parameters=self._init_optimiser_params(task, composer_params),
                                   external_parameters=composer_params.get('optimizer_external_params')) \
            .with_metrics(metric_function) \
            .with_history(composer_params.get('history_folder')) \
            .with_logger(log) \
            .with_cache(self.cache)
        gp_composer: GPComposer = builder.build()

        if self.timer.have_time_for_composing(composer_params['pop_size']):
            # Launch pipeline structure composition
            with self.timer.launch_composing():
                log.message(f'Pipeline composition started.')
                best_pipelines = gp_composer.compose_pipeline(data=train_data)
                best_pipeline_candidates = gp_composer.best_models
        else:
            # Use initial pipeline as final solution
            log.message(f'Timeout is too small for composing and is skipped '
                        f'because fit_time is {self.timer.assumption_fit_spend_time.total_seconds()} sec.')
            best_pipelines = fitted_assumption
            best_pipeline_candidates = [fitted_assumption]

        best_pipeline = best_pipelines[0] if isinstance(best_pipelines, Sequence) else best_pipelines

        # Workaround for logger missing after adapting/restoring
        for pipeline in best_pipeline_candidates:
            pipeline.log = log

        if with_tuning:
            timeout_for_tuning = self.timer.determine_resources_for_tuning()
            self.tune_final_pipeline(task, train_data,
                                     tuning_params['tuner_metric'],
                                     composer_requirements,
                                     best_pipeline,
                                     timeout_for_tuning,
                                     log)
        # enforce memory cleaning
        gc.collect()

        log.message('Model generation finished')
        return best_pipeline, best_pipeline_candidates, gp_composer.history

    def tune_final_pipeline(self, task: Task,
                            train_data: InputData,
                            tuner_metric: Optional[MetricType],
                            composer_requirements: PipelineComposerRequirements,
                            pipeline_gp_composed: Pipeline,
                            timeout_for_tuning: int,
                            log: Log):
        """ Launch tuning procedure for obtained pipeline by composer """

        if timeout_for_tuning < MINIMAL_SECONDS_FOR_TUNING:
            log.info(f'Time for pipeline composing was {str(self.timer.composing_spend_time)}.\n'
                     f'The remaining {max(0, timeout_for_tuning)} seconds are not enough '
                     f'to tune the hyperparameters.')
            log.info('Composed pipeline returned without tuning.')
        else:
            if tuner_metric is None:
                # Default metric for tuner
                tune_metrics = TunerMetricByTask(task.task_type)
                tuner_loss, loss_params = tune_metrics.get_metric_and_params(train_data)
                log.message(f'Tuner metric is None, {tuner_loss.__name__} is set as default')
            else:
                # Get metric and parameters by name
                loss_params = None
                tuner_loss = MetricsRepository().metric_by_id(tuner_metric, default_callable=tuner_metric)

            # Tune all nodes in the pipeline
            with self.timer.launch_tuning():
                log.message('Hyperparameters tuning started')
                vb_number = composer_requirements.validation_blocks
                folds = composer_requirements.cv_folds
                timeout_for_tuning = abs(timeout_for_tuning) / 60
                pipeline_gp_composed = pipeline_gp_composed. \
                    fine_tune_all_nodes(loss_function=tuner_loss,
                                        loss_params=loss_params,
                                        input_data=train_data,
                                        iterations=DEFAULT_TUNING_ITERATIONS_NUMBER,
                                        timeout=timeout_for_tuning,
                                        cv_folds=folds,
                                        validation_blocks=vb_number)
                log.message('Hyperparameters tuning finished')
        return pipeline_gp_composed


def _divide_parameters(common_dict: dict) -> List[dict]:
    """ Divide common dictionary into dictionary with parameters for API, Composer and Tuner

    :param common_dict: dictionary with parameters for all AutoML modules
    """
    api_params_dict = dict(train_data=None, task=Task, logger=Log, timeout=5, n_jobs=1,
                           use_cache=False)

    composer_params_dict = dict(max_depth=None, max_arity=None, pop_size=None, num_of_generations=None,
                                available_operations=None, composer_metric=None, validation_blocks=None,
                                cv_folds=None, genetic_scheme=None, history_folder=None,
                                stopping_after_n_generation=None, optimizer=None, optimizer_external_params=None,
                                collect_intermediate_metric=False, max_pipeline_fit_time=None, initial_assumption=None,
                                preset='auto')

    tuner_params_dict = dict(with_tuning=False, tuner_metric=None)

    dict_list = [api_params_dict, composer_params_dict, tuner_params_dict]
    for k, v in common_dict.items():
        is_unknown_key = True
        for i, dct in enumerate(dict_list):
            if k in dict_list[i]:
                dict_list[i][k] = v
                is_unknown_key = False
        if is_unknown_key:
            raise KeyError(f"Invalid key parameter {k}")

    return dict_list
