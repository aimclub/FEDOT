import datetime
from typing import Callable, Union, Optional

import numpy as np
from sklearn.metrics import roc_auc_score as roc_auc, mean_squared_error
from fedot.api.api_utils.initial_assumptions import API_initial_assumptions_helper
from fedot.api.api_utils.metrics import API_metrics_helper
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.composer.gp_composer.gp_composer import (GPComposerBuilder,
                                                         GPComposerRequirements)
from fedot.core.optimisers.gp_comp.gp_optimiser import GeneticSchemeTypesEnum, GPGraphOptimiserParameters
from fedot.core.data.data import InputData
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.log import Log
from fedot.core.repository.operation_types_repository import get_operations_for_task
from fedot.core.repository.quality_metrics_repository import (MetricsRepository)
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.utilities.define_metric_by_task import MetricByTask, TunerMetricByTask

from deap import tools
from fedot.core.optimisers.utils.pareto import ParetoFront


class API_composer_helper(API_metrics_helper, API_initial_assumptions_helper):

    def obtain_metric(self, task: Task, composer_metric: Union[str, Callable]):
        # the choice of the metric for the chain quality assessment during composition
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

        # Create init chain
        node_from_task = self.assumption_by_task(task=task)
        node_final = self.assumption_by_data(data=data, node_from_task=node_from_task)

        init_chain = Pipeline(node_final)
        return init_chain

    def get_composer_dict(self, composer_dict):
        filtred_dict = composer_dict.copy()
        params_dict = dict(train_data=None, task=Task, logger=Log, max_depth=None, max_arity=None, pop_size=None,
                           num_of_generations=None, available_operations=None, composer_metric=None, timeout=5,
                           with_tuning=False, tuner_metric=None, cv_folds=None, initial_chain=None)
        for key in composer_dict.keys():
            if key not in params_dict.keys():
                filtred_dict.pop(key)
        return filtred_dict

    def obtain_model(self, **composer_dict):
        self.best_models = None
        self.history = None
        self.current_model = composer_dict['current_model']

        if composer_dict['is_composing_required']:
            execution_dict = self.get_composer_dict(composer_dict)
            self.current_model, self.best_models, self.history = self.compose_fedot_model(**execution_dict)

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
        init_chain in it """

        builder = GPComposerBuilder(task=task). \
            with_requirements(composer_requirements). \
            with_optimiser_parameters(optimizer_parameters). \
            with_metrics(metric_function).with_logger(logger)

        if initial_pipeline is not None:
            initial_pipeline = initial_pipeline
        else:
            initial_pipeline = self.obtain_initial_assumption(task, data)

        if initial_pipeline is not None:
            builder = builder.with_initial_pipeline(initial_pipeline)

        return builder

    def divide_operations(self,
                          available_operations,
                          task):
        """ Function divide operations for primary and secondary """

        if task.task_type == TaskTypesEnum.ts_forecasting:
            ts_data_operations = get_operations_for_task(task=task,
                                                         mode='data_operation',
                                                         tags=["ts_specific"])
            # Remove exog data operation from the list
            ts_data_operations.remove('exog_ts_data_source')

            primary_operations = ts_data_operations
            secondary_operations = available_operations
        else:
            primary_operations = available_operations
            secondary_operations = available_operations
        return primary_operations, secondary_operations

    def compose_fedot_model(self, train_data: [InputData, MultiModalData],
                            task: Task,
                            logger: Log,
                            max_depth: int,
                            max_arity: int,
                            pop_size: int,
                            num_of_generations: int,
                            available_operations: list = None,
                            composer_metric=None,
                            timeout: float = 5,
                            with_tuning=False,
                            tuner_metric=None,
                            cv_folds: Optional[int] = None,
                            validation_blocks: int = None,
                            initial_pipeline=None
                            ):
        """ Function for composing FEDOT chain model """

        metric_function = self.obtain_metric(task, composer_metric)

        if available_operations is None:
            available_operations = get_operations_for_task(task, mode='model')

        logger.message(f'Composition started. Parameters tuning: {with_tuning}. '
                       f'Set of candidate models: {available_operations}. Composing time limit: {timeout} min')

        primary_operations, secondary_operations = self.divide_operations(available_operations,
                                                                          task)

        timeout_for_composing = timeout / 2 if with_tuning else timeout
        # the choice and initialisation of the GP composer
        composer_requirements = \
            GPComposerRequirements(primary=primary_operations,
                                   secondary=secondary_operations,
                                   max_arity=max_arity,
                                   max_depth=max_depth,
                                   pop_size=pop_size,
                                   num_of_generations=num_of_generations,
                                   cv_folds=cv_folds,
                                   validation_blocks=validation_blocks,
                                   timeout=datetime.timedelta(minutes=timeout_for_composing))

        optimizer_parameters = GPGraphOptimiserParameters(genetic_scheme_type=GeneticSchemeTypesEnum.parameter_free)

        builder = self.get_gp_composer_builder(task=task,
                                               metric_function=metric_function,
                                               composer_requirements=composer_requirements,
                                               optimizer_parameters=optimizer_parameters,
                                               data=train_data,
                                               initial_pipeline=initial_pipeline,
                                               logger=logger)

        gp_composer = builder.build()

        logger.message('Pipeline composition started')
        pipeline_gp_composed = gp_composer.compose_pipeline(data=train_data)

        pipeline_for_return = pipeline_gp_composed

        if isinstance(pipeline_gp_composed, list):
            for pipeline in pipeline_gp_composed:
                pipeline.log = logger
            pipeline_for_return = pipeline_gp_composed[0]
            best_candidates = gp_composer.optimiser.archive
        else:
            best_candidates = [pipeline_gp_composed]
            pipeline_gp_composed.log = logger

        if with_tuning:
            logger.message('Hyperparameters tuning started')

            if tuner_metric is None:
                # Default metric for tuner
                tune_metrics = TunerMetricByTask(task.task_type)
                tuner_loss, loss_params = tune_metrics.get_metric_and_params(train_data)
                logger.message(f'Tuner metric is None, '
                               f'{tuner_loss.__name__} was set as default')
            else:
                # Get metric and parameters by name
                tuner_loss, loss_params = self.tuner_metric_by_name(metric_name=tuner_metric,
                                                                    train_data=train_data,
                                                                    task=task)

            iterations = 20 if timeout is None else 1000
            timeout_for_tuning = timeout / 2

            # Tune all nodes in the pipeline

            vb_number = composer_requirements.validation_blocks
            folds = composer_requirements.cv_folds
            if train_data.task.task_type != TaskTypesEnum.ts_forecasting:
                # TODO remove after implementation of CV for class/regr
                logger.warn('Cross-validation is not supported for tuning of ts-forecasting pipeline: '
                            'hold-out validation used instead')
                folds = None
            pipeline_for_return = pipeline_for_return.fine_tune_all_nodes(loss_function=tuner_loss,
                                                                          loss_params=loss_params,
                                                                          input_data=train_data,
                                                                          iterations=iterations,
                                                                          timeout=timeout_for_tuning,
                                                                          cv_folds=folds,
                                                                          validation_blocks=vb_number)

        logger.message('Model composition finished')

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

        if task.task_type == TaskTypesEnum.regression:
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
