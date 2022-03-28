import datetime
import gc
import traceback
from typing import Callable, Dict, List, Optional, Type, Union

import numpy as np
from deap import tools
from sklearn.metrics import mean_squared_error, roc_auc_score as roc_auc

from fedot.api.api_utils.assumptions.assumptions_builder import AssumptionsBuilder
from fedot.api.api_utils.metrics import ApiMetrics
from fedot.api.api_utils.presets import update_builder
from fedot.api.time import ApiTime
from fedot.core.composer.composer_builder import ComposerBuilder
from fedot.core.composer.gp_composer.gp_composer import (PipelineComposerRequirements)
from fedot.core.composer.gp_composer.specific_operators import boosting_mutation, parameter_change_mutation
from fedot.core.constants import MINIMAL_SECONDS_FOR_TUNING, DEFAULT_TUNING_ITERATIONS_NUMBER
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.log import Log
from fedot.core.optimisers.gp_comp.gp_optimiser import EvoGraphOptimiser, GPGraphOptimiserParameters, \
    GeneticSchemeTypesEnum
from fedot.core.optimisers.gp_comp.operators.crossover import CrossoverTypesEnum
from fedot.core.optimisers.gp_comp.operators.mutation import MutationTypesEnum
from fedot.core.optimisers.optimizer import GraphOptimiser, GraphOptimiserParameters
from fedot.core.optimisers.utils.pareto import ParetoFront
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.operation_types_repository import get_operations_for_task
from fedot.core.repository.quality_metrics_repository import (MetricsRepository)
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.utilities.define_metric_by_task import MetricByTask, TunerMetricByTask


class ApiComposer:

    def __init__(self, problem: str):
        self.metrics = ApiMetrics(problem)
        self.optimiser = EvoGraphOptimiser
        self.optimizer_external_parameters = None
        self.current_model = None
        self.best_models = None
        self.history = None

        self.preset_name = None
        self.timer = None

    def obtain_metric(self, task: Task, composer_metric: Union[str, Callable]):
        # the choice of the metric for the pipeline quality assessment during composition
        if composer_metric is None:
            composer_metric = MetricByTask(task.task_type).metric_cls.get_value

        if isinstance(composer_metric, str) or isinstance(composer_metric, Callable):
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
        self.best_models = None
        self.history = None
        preset = common_dict['preset']
        # Prepare parameters
        api_params_dict, composer_params_dict, tuner_params_dict = _divide_parameters(common_dict)

        # Start composing - pipeline structure search
        self.current_model, self.best_models, self.history = self.compose_fedot_model(
            api_params=api_params_dict,
            composer_params=composer_params_dict,
            tuning_params=tuner_params_dict,
            preset=preset)

        if isinstance(self.best_models, tools.ParetoFront):
            self.best_models.__class__ = ParetoFront
            self.best_models.objective_names = common_dict['composer_metric']

        return self.current_model, self.best_models, self.history

    def get_gp_composer_builder(self, task: Task,
                                metric_function,
                                preset: str,
                                full_minutes_timeout: int,
                                composer_requirements: PipelineComposerRequirements,
                                optimiser: Type[GraphOptimiser],
                                optimizer_parameters: GraphOptimiserParameters,
                                data: Union[InputData, MultiModalData],
                                logger: Log,
                                initial_assumption: Union[Pipeline, List[Pipeline]] = None,
                                optimizer_external_parameters: Optional[Dict] = None):
        """
        Return GPComposerBuilder with parameters and if it is necessary
        initial_assumption in it

        :param task: task for solving
        :param metric_function: function for individuals evaluating
        :param preset: name of using preset
        :param full_minutes_timeout: number of minutes for all AutoML
        :param composer_requirements: params for composer
        :param optimiser: optimiser for composer
        :param optimizer_parameters: params for optimizer
        :param data: data for evaluating
        :param logger: log object
        :param initial_assumption: list of initial pipelines
        :param optimizer_external_parameters: eternal parameters for optimizer
        """

        builder = ComposerBuilder(task=task). \
            with_requirements(composer_requirements). \
            with_optimiser(optimiser, optimizer_parameters, optimizer_external_parameters). \
            with_metrics(metric_function).with_logger(logger)

        if initial_assumption is None:
            initial_pipelines = AssumptionsBuilder.get(task, data).build()
        elif isinstance(initial_assumption, Pipeline):
            initial_pipelines = [initial_assumption]
        else:
            if not isinstance(initial_assumption, list):
                prefix = 'Incorrect type of initial_assumption'
                raise ValueError(f'{prefix}: List[Pipeline] or Pipeline needed, but has {type(initial_assumption)}')
            initial_pipelines = initial_assumption
        # Check initial assumption
        fitted_pipeline, fit_time = fit_and_check_correctness(initial_pipelines[0], data, logger=logger,
                                                              n_jobs=composer_requirements.n_jobs)
        builder = builder.with_initial_pipelines(initial_pipelines)

        # Update builder and preset if required
        builder, new_preset = update_builder(builder, composer_requirements, fit_time, full_minutes_timeout, preset)
        # Store information about preset
        self.preset_name = new_preset
        return builder, fitted_pipeline, fit_time

    def divide_operations(self,
                          available_operations,
                          task):
        """ Function divide operations for primary and secondary """

        if task.task_type == TaskTypesEnum.ts_forecasting:
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

    def _set_initial_assumption(self, api_params: dict, composer_params: dict):
        """ Generates and sets an initial assumption, if not given, from the given available operations
        Also, since it is impossible to form a valid pipeline for the time series problem without 'lagged' operation,
        if it is not in the list of available ones, it is added """

        if api_params['task'] == TaskTypesEnum.ts_forecasting and \
                'lagged' not in api_params['available_operations']:
            api_params['available_operations'].append('lagged')

        if not api_params['initial_assumption']:
            is_input = isinstance(api_params['train_data'], InputData)
            if is_input and api_params['train_data'].data_type == DataTypesEnum.multi_ts:
                # It is needed to use data operations also for multi_ts data
                repository_name = 'all'
            else:
                repository_name = 'model'
            assumptions_builder = AssumptionsBuilder(task=api_params['task'],
                                                     data=api_params['train_data'],
                                                     repository_name=repository_name)\
                .get(api_params['task'], api_params['train_data'])\
                .with_logger(api_params['logger'])\
                .from_operations(composer_params['available_operations'])
            api_params['initial_assumption'] = assumptions_builder.build()

    def compose_fedot_model(self, api_params: dict, composer_params: dict, tuning_params: dict,
                            preset: str):
        """ Function for composing FEDOT pipeline model """
        # Initialize timer for all AutoMl operations
        self.timer = ApiTime(time_for_automl=api_params['timeout'],
                             with_tuning=tuning_params['with_tuning'])

        metric_function = self.obtain_metric(api_params['task'], composer_params['composer_metric'])

        if not composer_params['available_operations']:
            composer_params['available_operations'] = get_operations_for_task(api_params['task'], mode='model')
        else:
            self._set_initial_assumption(api_params=api_params, composer_params=composer_params)

        api_params['logger'].message('AutoML started. Parameters tuning: {}. ''Set of candidate models: {}. '
                                     'Time limit: {} min'.format(tuning_params['with_tuning'],
                                                                 composer_params['available_operations'],
                                                                 api_params['timeout']))

        primary_operations, secondary_operations = self.divide_operations(composer_params['available_operations'],
                                                                          api_params['task'])

        # the choice and initialisation of the GP composer
        composer_requirements = PipelineComposerRequirements(primary=primary_operations,
                                                             secondary=secondary_operations,
                                                             max_arity=composer_params['max_arity'],
                                                             max_depth=composer_params['max_depth'],
                                                             pop_size=composer_params['pop_size'],
                                                             num_of_generations=composer_params['num_of_generations'],
                                                             cv_folds=composer_params['cv_folds'],
                                                             validation_blocks=composer_params['validation_blocks'],
                                                             timeout=self.timer.datetime_composing,
                                                             n_jobs=api_params['n_jobs'])

        genetic_scheme_type = GeneticSchemeTypesEnum.parameter_free

        if composer_params['genetic_scheme'] == 'steady_state':
            genetic_scheme_type = GeneticSchemeTypesEnum.steady_state

        mutations = [boosting_mutation, parameter_change_mutation,
                     MutationTypesEnum.single_change,
                     MutationTypesEnum.single_drop,
                     MutationTypesEnum.single_add]

        # TODO remove workaround after validation fix
        if api_params['task'].task_type != TaskTypesEnum.ts_forecasting:
            mutations.append(MutationTypesEnum.single_edge)

        optimizer_parameters = GPGraphOptimiserParameters(
            genetic_scheme_type=genetic_scheme_type,
            mutation_types=mutations,
            crossover_types=[CrossoverTypesEnum.one_point, CrossoverTypesEnum.subtree],
            history_folder=composer_params.get('history_folder'),
            stopping_after_n_generation=composer_params.get('stopping_after_n_generation')
        )
        if 'optimizer' in composer_params:
            self.optimiser = composer_params['optimizer']
        if 'optimizer_external_params' in composer_params:
            self.optimizer_external_parameters = composer_params['optimizer_external_params']

        builder, fitted_initial_pipeline, init_pipeline_fit_time = \
            self.get_gp_composer_builder(task=api_params['task'],
                                         metric_function=metric_function,
                                         preset=preset,
                                         full_minutes_timeout=api_params['timeout'],
                                         composer_requirements=composer_requirements,
                                         optimiser=self.optimiser,
                                         optimizer_parameters=optimizer_parameters,
                                         optimizer_external_parameters=self.optimizer_external_parameters,
                                         data=api_params['train_data'],
                                         initial_assumption=api_params['initial_assumption'],
                                         logger=api_params['logger'])
        gp_composer = None
        timeout_were_set = self.timer.datetime_composing is not None
        if timeout_were_set and init_pipeline_fit_time >= self.timer.datetime_composing / composer_params['pop_size']:
            api_params['logger'].message(f'Timeout is too small for composing '
                                         f'because fit_time is {init_pipeline_fit_time.total_seconds()} sec.,'
                                         ' so it is skipped.')
            # Use initial pipeline as final solution
            pipeline_gp_composed = fitted_initial_pipeline
        else:
            # Launch pipeline structure composition
            with self.timer.launch_composing():
                gp_composer = builder.build()
                api_params['logger'].message(f'Pipeline composition started. Initial pipeline was fitted '
                                             f'for fit_time {init_pipeline_fit_time.total_seconds()} sec.')
                pipeline_gp_composed = gp_composer.compose_pipeline(data=api_params['train_data'])

        if isinstance(pipeline_gp_composed, list):
            for pipeline in pipeline_gp_composed:
                pipeline.log = api_params['logger']
            pipeline_gp_composed = pipeline_gp_composed[0]
            best_candidates = gp_composer.optimiser.archive
        else:
            best_candidates = [pipeline_gp_composed]
            pipeline_gp_composed.log = api_params['logger']

        pipeline_gp_composed = self.tune_final_pipeline(api_params=api_params, tuning_params=tuning_params,
                                                        composer_requirements=composer_requirements,
                                                        pipeline_gp_composed=pipeline_gp_composed,
                                                        timer=self.timer,
                                                        init_pipeline_fit_time=init_pipeline_fit_time)

        api_params['logger'].message('Model generation finished')

        if gp_composer is not None:
            history = gp_composer.optimiser.history
        else:
            history = None

        # enforce memory cleaning
        gc.collect()
        return pipeline_gp_composed, best_candidates, history

    def tuner_metric_by_name(self, metric_name, train_data: InputData, task: Task):
        """ Function allow to obtain metric for tuner by its name

        :param metric_name: name of metric
        :param train_data: InputData for train
        :param task: task to solve

        :return tuner_loss: loss function for tuner
        :return loss_params: parameters for tuner loss (can be None in some cases)
        """
        loss_params_dict = {roc_auc: {'multi_class': 'ovr', 'average': 'macro'},
                            mean_squared_error: {'squared': False}}

        if task.task_type == TaskTypesEnum.regression or task.task_type == TaskTypesEnum.ts_forecasting:
            loss_function = mean_squared_error
            loss_params = {'squared': False}
        elif task.task_type == TaskTypesEnum.classification:
            # Default metric for time classification
            amount_of_classes = len(np.unique(np.array(train_data.target)))
            loss_function = roc_auc
            if amount_of_classes == 2:
                loss_params = None
            else:
                loss_params = loss_params_dict[loss_function]
        else:
            raise NotImplementedError(f'Metric for "{task.task_type}" is not supported')

        if loss_function is None:
            raise ValueError(f'Incorrect tuner metric {loss_function}')

        return loss_function, loss_params

    def tune_final_pipeline(self, api_params, tuning_params,
                            composer_requirements, pipeline_gp_composed,
                            timer: ApiTime, init_pipeline_fit_time):
        """ Launch tuning procedure for obtained pipeline by composer """
        if tuning_params['with_tuning'] is False:
            # Return source pipeline
            return pipeline_gp_composed

        timeout_for_tuning = timer.determine_resources_for_tuning(init_pipeline_fit_time)

        if timeout_for_tuning < MINIMAL_SECONDS_FOR_TUNING:
            api_params['logger'].info(f'Time for pipeline composing was {str(timer.composing_spend_time)}.\n'
                                      f'The remaining {max(0, timeout_for_tuning)} seconds are not enough '
                                      f'to tune the hyperparameters.')
            api_params['logger'].info('Composed pipeline returned without tuning.')
        else:
            if tuning_params['tuner_metric'] is None:
                # Default metric for tuner
                tune_metrics = TunerMetricByTask(api_params['task'].task_type)
                tuner_loss, loss_params = tune_metrics.get_metric_and_params(api_params['train_data'])
                api_params['logger'].message(f'Tuner metric is None, '
                                             f'{tuner_loss.__name__} was set as default')
            else:
                # Get metric and parameters by name
                tuner_loss, loss_params = self.tuner_metric_by_name(metric_name=tuning_params['tuner_metric'],
                                                                    train_data=api_params['train_data'],
                                                                    task=api_params['task'])

            # Tune all nodes in the pipeline
            with timer.launch_tuning():
                api_params['logger'].message('Hyperparameters tuning started')
                vb_number = composer_requirements.validation_blocks
                folds = composer_requirements.cv_folds
                timeout_for_tuning = abs(timeout_for_tuning) / 60
                pipeline_gp_composed = pipeline_gp_composed.\
                    fine_tune_all_nodes(loss_function=tuner_loss,
                                        loss_params=loss_params,
                                        input_data=api_params['train_data'],
                                        iterations=DEFAULT_TUNING_ITERATIONS_NUMBER,
                                        timeout=timeout_for_tuning,
                                        cv_folds=folds,
                                        validation_blocks=vb_number)
                api_params['logger'].message('Hyperparameters tuning finished')
        return pipeline_gp_composed


def fit_and_check_correctness(pipeline: Pipeline,
                              data: Union[InputData, MultiModalData],
                              logger: Log, n_jobs=1):
    """ Test is initial pipeline can be fitted on presented data and give predictions """
    try:
        _, data_test = train_test_data_setup(data)
        start_init_fit = datetime.datetime.now()

        logger.message('Initial pipeline fitting started')

        pipeline.fit(data, n_jobs=n_jobs)
        pipeline.predict(data_test)

        fit_time = datetime.datetime.now() - start_init_fit
        logger.message('Initial pipeline was fitted successfully')
    except Exception as ex:
        fit_failed_info = f'Initial pipeline fit was failed due to: {ex}.'
        advice_info = f'{fit_failed_info} Check pipeline structure and the correctness of the data'

        logger.info(fit_failed_info)
        print(traceback.format_exc())
        raise ValueError(advice_info)
    return pipeline, fit_time


def _divide_parameters(common_dict: dict) -> List[dict]:
    """ Divide common dictionary into dictionary with parameters for API, Composer and Tuner

    :param common_dict: dictionary with parameters for all AutoML modules
    """
    api_params_dict = dict(train_data=None, task=Task, logger=Log, timeout=5, initial_assumption=None, n_jobs=1)

    composer_params_dict = dict(max_depth=None, max_arity=None, pop_size=None, num_of_generations=None,
                                available_operations=None, composer_metric=None, validation_blocks=None,
                                cv_folds=None, genetic_scheme=None, history_folder=None,
                                stopping_after_n_generation=None, optimizer=None, optimizer_external_params=None)

    tuner_params_dict = dict(with_tuning=False, tuner_metric=None)

    dict_list = [api_params_dict, composer_params_dict, tuner_params_dict]
    for i, dct in enumerate(dict_list):
        update_dict = dct.copy()
        update_dict.update(common_dict)
        for key in common_dict.keys():
            if key not in dct.keys():
                update_dict.pop(key)
        dict_list[i] = update_dict

    return dict_list
