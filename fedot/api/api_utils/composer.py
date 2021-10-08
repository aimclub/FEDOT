import datetime
from typing import Callable, Union

import numpy as np
from deap import tools
from sklearn.metrics import roc_auc_score as roc_auc, mean_squared_error

from fedot.api.api_utils.initial_assumptions import ApiInitialAssumptionsHelper
from fedot.api.api_utils.metrics import ApiMetricsHelper
from fedot.core.composer.gp_composer.gp_composer import (GPComposerBuilder,
                                                         GPComposerRequirements)
from fedot.core.composer.gp_composer.specific_operators import boosting_mutation, parameter_change_mutation
from fedot.core.data.data import InputData
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.log import Log
from fedot.core.optimisers.gp_comp.gp_optimiser import GeneticSchemeTypesEnum, GPGraphOptimiserParameters
from fedot.core.optimisers.gp_comp.operators.crossover import CrossoverTypesEnum
from fedot.core.optimisers.gp_comp.operators.mutation import single_drop_mutation, single_edge_mutation, \
    single_change_mutation, single_add_mutation, MutationTypesEnum
from fedot.core.optimisers.utils.pareto import ParetoFront
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.operation_types_repository import get_operations_for_task
from fedot.core.repository.quality_metrics_repository import (MetricsRepository)
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.utilities.define_metric_by_task import MetricByTask, TunerMetricByTask


class ApiComposerHelper(ApiMetricsHelper, ApiInitialAssumptionsHelper):

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
                metric_id = self.get_composer_metrics_mapping(metric_name=specific_metric)
                if metric_id is None:
                    raise ValueError(f'Incorrect metric {specific_metric}')
                specific_metric_function = MetricsRepository().metric_by_id(metric_id)
            metric_function.append(specific_metric_function)
        return metric_function

    def obtain_initial_assumption(self,
                                  task: Task,
                                  data) -> Pipeline:

        init_pipeline = self.get_initial_assumption(data=data, task=task)

        return init_pipeline

    def get_composer_dict(self, composer_dict):

        api_params_dict = dict(train_data=None, task=Task, logger=Log, timeout=5, initial_pipeline=None)

        composer_params_dict = dict(max_depth=None, max_arity=None, pop_size=None, num_of_generations=None,
                                    available_operations=None, composer_metric=None, validation_blocks=None,
                                    cv_folds=None, genetic_scheme=None)

        tuner_params_dict = dict(with_tuning=False, tuner_metric=None)

        dict_list = [api_params_dict, composer_params_dict, tuner_params_dict]
        for i, dct in enumerate(dict_list):
            update_dict = dct.copy()
            update_dict.update(composer_dict)
            for key in composer_dict.keys():
                if key not in dct.keys():
                    update_dict.pop(key)
            dict_list[i] = update_dict

        return dict_list

    def obtain_model(self, **composer_dict):
        self.best_models = None
        self.history = None
        self.current_model = composer_dict['current_model']

        if composer_dict['is_composing_required']:
            api_params_dict, composer_params_dict, tuner_params_dict = self.get_composer_dict(composer_dict)

            self.current_model, self.best_models, self.history = self.compose_fedot_model(
                api_params=api_params_dict,
                composer_params=composer_params_dict,
                tuning_params=tuner_params_dict)

        if isinstance(self.best_models, tools.ParetoFront):
            self.best_models.__class__ = ParetoFront
            self.best_models.objective_names = composer_dict['composer_metric']

        self.current_model.fit_from_scratch(composer_dict['train_data'])

        return self.current_model, self.best_models, self.history

    def get_gp_composer_builder(self, task: Task,
                                metric_function,
                                composer_requirements: GPComposerRequirements,
                                optimizer_parameters: GPGraphOptimiserParameters,
                                data: Union[InputData, MultiModalData],
                                logger: Log,
                                initial_pipeline: Pipeline = None):
        """ Return GPComposerBuilder with parameters and if it is necessary
        init_pipeline in it """

        builder = GPComposerBuilder(task=task). \
            with_requirements(composer_requirements). \
            with_optimiser_parameters(optimizer_parameters). \
            with_metrics(metric_function).with_logger(logger)

        if initial_pipeline is None:
            initial_pipeline = self.obtain_initial_assumption(task, data)

        if initial_pipeline is not None:
            if not isinstance(initial_pipeline, Pipeline):
                prefix = 'Incorrect type of initial_pipeline'
                raise ValueError(f'{prefix}: Pipeline needed, but has {type(initial_pipeline)}')
            builder = builder.with_initial_pipeline(initial_pipeline)

        return builder

    def divide_operations(self,
                          available_operations,
                          task):
        """ Function divide operations for primary and secondary """

        if task.task_type == TaskTypesEnum.ts_forecasting:
            ts_data_operations = get_operations_for_task(task=task,
                                                         mode='data_operation',
                                                         tags=["non_lagged"])
            # Remove exog data operation from the list
            ts_data_operations.remove('exog_ts')

            primary_operations = ts_data_operations
            secondary_operations = available_operations
        else:
            primary_operations = available_operations
            secondary_operations = available_operations
        return primary_operations, secondary_operations

    def compose_fedot_model(self,
                            api_params: dict,
                            composer_params: dict,
                            tuning_params: dict
                            ):
        """ Function for composing FEDOT pipeline model """

        metric_function = self.obtain_metric(api_params['task'], composer_params['composer_metric'])

        if composer_params['available_operations'] is None:
            composer_params['available_operations'] = get_operations_for_task(api_params['task'], mode='model')

        api_params['logger'].message('Composition started. Parameters tuning: {}. ''Set of candidate models: {}. '
                                     'Time limit: {} min'.format(tuning_params['with_tuning'],
                                                                 composer_params['available_operations'],
                                                                 api_params['timeout']))

        primary_operations, secondary_operations = self.divide_operations(composer_params['available_operations'],
                                                                          api_params['task'])

        timeout_for_composing = api_params['timeout'] / 2 if tuning_params['with_tuning'] else api_params['timeout']
        # the choice and initialisation of the GP composer
        composer_requirements = \
            GPComposerRequirements(primary=primary_operations,
                                   secondary=secondary_operations,
                                   max_arity=composer_params['max_arity'],
                                   max_depth=composer_params['max_depth'],
                                   pop_size=composer_params['pop_size'],
                                   num_of_generations=composer_params['num_of_generations'],
                                   cv_folds=composer_params['cv_folds'],
                                   validation_blocks=composer_params['validation_blocks'],
                                   timeout=datetime.timedelta(minutes=timeout_for_composing))

        genetic_scheme_type = GeneticSchemeTypesEnum.parameter_free

        if composer_params['genetic_scheme'] == 'steady_state':
            genetic_scheme_type = GeneticSchemeTypesEnum.steady_state

        optimizer_parameters = GPGraphOptimiserParameters(genetic_scheme_type=genetic_scheme_type,
                                                          mutation_types=[boosting_mutation, parameter_change_mutation,
                                                                          MutationTypesEnum.single_edge,
                                                                          MutationTypesEnum.single_change,
                                                                          MutationTypesEnum.single_drop,
                                                                          MutationTypesEnum.single_add],
                                                          crossover_types=[CrossoverTypesEnum.one_point,
                                                                           CrossoverTypesEnum.subtree])

        builder = self.get_gp_composer_builder(task=api_params['task'],
                                               metric_function=metric_function,
                                               composer_requirements=composer_requirements,
                                               optimizer_parameters=optimizer_parameters,
                                               data=api_params['train_data'],
                                               initial_pipeline=api_params['initial_pipeline'],
                                               logger=api_params['logger'])

        gp_composer = builder.build()

        api_params['logger'].message('Pipeline composition started')
        pipeline_gp_composed = gp_composer.compose_pipeline(data=api_params['train_data'])

        pipeline_for_return = pipeline_gp_composed

        if isinstance(pipeline_gp_composed, list):
            for pipeline in pipeline_gp_composed:
                pipeline.log = api_params['logger']
            pipeline_for_return = pipeline_gp_composed[0]
            best_candidates = gp_composer.optimiser.archive
        else:
            best_candidates = [pipeline_gp_composed]
            pipeline_gp_composed.log = api_params['logger']

        if tuning_params['with_tuning']:
            api_params['logger'].message('Hyperparameters tuning started')

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

            iterations = 20 if api_params['timeout'] is None else 1000
            timeout_for_tuning = api_params['timeout'] / 2

            # Tune all nodes in the pipeline

            vb_number = composer_requirements.validation_blocks
            folds = composer_requirements.cv_folds
            if api_params['train_data'].task.task_type != TaskTypesEnum.ts_forecasting:
                # TODO remove after implementation of CV for class/regr
                api_params['logger'].warn('Cross-validation is not supported for tuning of ts-forecasting pipeline: '
                                          'hold-out validation used instead')
                folds = None
            pipeline_for_return = pipeline_for_return.fine_tune_all_nodes(loss_function=tuner_loss,
                                                                          loss_params=loss_params,
                                                                          input_data=api_params['train_data'],
                                                                          iterations=iterations,
                                                                          timeout=timeout_for_tuning,
                                                                          cv_folds=folds,
                                                                          validation_blocks=vb_number)

        api_params['logger'].message('Model composition finished')

        history = gp_composer.optimiser.history

        return pipeline_for_return, best_candidates, history

    def tuner_metric_by_name(self, metric_name, train_data: InputData, task: Task):
        """ Function allow to obtain metric for tuner by its name

        :param metric_name: name of metric
        :param train_data: InputData for train
        :param task: task to solve

        :return tuner_loss: loss function for tuner
        :return loss_params: parameters for tuner loss (can be None in some cases)
        """
        loss_params_dict = {roc_auc: {'multi_class': 'ovr'},
                            mean_squared_error: {'squared': False}}
        loss_function = None

        if type(metric_name) is str:
            loss_function = self.get_tuner_metrics_mapping(metric_name)

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
