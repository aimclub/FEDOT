import datetime
from typing import Callable, Union, Optional

import numpy as np

from fedot.api.api_utils.metrics import Fedot_metrics_helper
from fedot.core.chains.chain import Chain
from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.composer.gp_composer.gp_composer import (GPChainOptimiserParameters, GPComposerBuilder,
                                                         GPComposerRequirements)
from fedot.core.composer.optimisers.gp_comp.gp_optimiser import GeneticSchemeTypesEnum
from fedot.core.composer.optimisers.gp_comp.operators.crossover import CrossoverTypesEnum
from fedot.core.composer.optimisers.gp_comp.operators.mutation import MutationTypesEnum
from fedot.core.data.data import InputData
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.log import Log
from fedot.core.repository.operation_types_repository import get_operations_for_task, get_ts_operations
from fedot.core.repository.quality_metrics_repository import (MetricsRepository)
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.utilities.define_metric_by_task import MetricByTask, TunerMetricByTask

from deap import tools
from fedot.core.composer.optimisers.utils.pareto import ParetoFront


class Fedot_composer_helper(Fedot_metrics_helper):

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

    def _assumption_by_data(self,
                            data,
                            node_from_task) -> Chain:

        if isinstance(data, MultiModalData):
            node_final = SecondaryNode('ridge', nodes_from=[])
            for data_source_name in data.keys():
                last_node_for_sub_chain = \
                    SecondaryNode('ridge', [SecondaryNode('lagged', [PrimaryNode(data_source_name)])])
                node_final.nodes_from.append(last_node_for_sub_chain)
        else:
            node_final = node_from_task

        return node_final

    def _assumption_by_task(self,
                            task: Task) -> Chain:

        node_lagged = PrimaryNode('scaling')
        initial_assumption_dict = {TaskTypesEnum.classification: SecondaryNode('xgboost', nodes_from=[node_lagged]),
                                   TaskTypesEnum.regression: SecondaryNode('ridge', nodes_from=[node_lagged]),
                                   TaskTypesEnum.ts_forecasting: SecondaryNode('ridge',
                                                                               nodes_from=[PrimaryNode('lagged')])}

        return initial_assumption_dict[task.task_type]

    def obtain_initial_assumption(self,
                                  task: Task,
                                  data) -> Chain:

        # Create init chain
        node_from_task = self._assumption_by_task(task=task)
        node_final = self._assumption_by_data(data=data, node_from_task=node_from_task)

        init_chain = Chain(node_final)
        return init_chain

    def get_composer_dict(self, composer_dict):
        filtred_dict = composer_dict.copy()
        params_dict = dict(train_data=None, task=Task, logger=Log, max_depth=None, max_arity=None, pop_size=None,
                           num_of_generations=None, available_operations=None, composer_metric=None, learning_time=5,
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
                                optimizer_parameters: GPChainOptimiserParameters,
                                data: Union[InputData, MultiModalData],
                                logger: Log,
                                initial_chain: Chain = None):
        """ Return GPComposerBuilder with parameters and if it is necessary
        init_chain in it """

        builder = GPComposerBuilder(task=task). \
            with_requirements(composer_requirements). \
            with_optimiser_parameters(optimizer_parameters). \
            with_metrics(metric_function).with_logger(logger)

        if initial_chain is not None:
            init_chain = initial_chain
        else:
            init_chain = self.obtain_initial_assumption(task, data)

        if init_chain is not None:
            builder = builder.with_initial_chain(init_chain)

        return builder

    def divide_operations(self,
                          available_operations,
                          task):
        """ Function divide operations for primary and secondary """

        if task.task_type == TaskTypesEnum.ts_forecasting:
            ts_data_operations = get_ts_operations(mode='data_operations',
                                                   tags=["ts_specific"])
            # Remove exog data operation from the list
            try:
                ts_data_operations.remove('exog')
            except ValueError:
                print('Exog operations was deleted from ts_operations')

            primary_operations = ts_data_operations
            secondary_operations = available_operations
        else:
            primary_operations = available_operations
            secondary_operations = available_operations
        return primary_operations, secondary_operations

    def compose_fedot_model(self,
                            train_data: [InputData, MultiModalData],
                            task: Task,
                            logger: Log,
                            max_depth: int,
                            max_arity: int,
                            pop_size: int,
                            num_of_generations: int,
                            available_operations: list = None,
                            composer_metric=None,
                            learning_time: float = 5,
                            with_tuning=False,
                            tuner_metric=None,
                            cv_folds: Optional[int] = None,
                            initial_chain=None
                            ):
        """ Function for composing FEDOT chain model """
        metric_function = self.obtain_metric(task, composer_metric)

        if available_operations is None:
            available_operations = get_operations_for_task(task, mode='models')

        logger.message(f'Composition started. Parameters tuning: {with_tuning}. '
                       f'Set of candidate models: {available_operations}. Composing time limit: {learning_time} min')

        primary_operations, secondary_operations = self.divide_operations(available_operations,
                                                                          task)

        learning_time_for_composing = learning_time / 2 if with_tuning else learning_time
        # the choice and initialisation of the GP composer
        composer_requirements = \
            GPComposerRequirements(primary=primary_operations,
                                   secondary=secondary_operations,
                                   max_arity=max_arity,
                                   max_depth=max_depth,
                                   pop_size=pop_size,
                                   num_of_generations=num_of_generations,
                                   max_lead_time=datetime.timedelta(minutes=learning_time_for_composing),
                                   allow_single_operations=False,
                                   cv_folds=cv_folds)

        optimizer_parameters = GPChainOptimiserParameters(genetic_scheme_type=GeneticSchemeTypesEnum.parameter_free,
                                                          mutation_types=[MutationTypesEnum.parameter_change,
                                                                          MutationTypesEnum.simple,
                                                                          MutationTypesEnum.reduce,
                                                                          MutationTypesEnum.growth,
                                                                          MutationTypesEnum.local_growth],
                                                          crossover_types=[CrossoverTypesEnum.one_point,
                                                                           CrossoverTypesEnum.subtree])

        builder = self.get_gp_composer_builder(task=task,
                                               metric_function=metric_function,
                                               composer_requirements=composer_requirements,
                                               optimizer_parameters=optimizer_parameters,
                                               data=train_data,
                                               logger=logger)
        gp_composer = builder.build()

        logger.message('Pipeline composition started')
        chain_gp_composed = gp_composer.compose_chain(data=train_data)
        chain_for_return = chain_gp_composed

        if isinstance(chain_gp_composed, list):
            for chain in chain_gp_composed:
                chain.log = logger
            chain_for_return = chain_gp_composed[0]
            best_candidates = gp_composer.optimiser.archive
        else:
            best_candidates = [chain_gp_composed]
            chain_gp_composed.log = logger

        if with_tuning:
            logger.message('Hyperparameters tuning started')

            if tuner_metric is None:
                logger.message('Default loss function was set')
                # Default metric for tuner
                tune_metrics = TunerMetricByTask(task.task_type)
                tuner_loss, loss_params = tune_metrics.get_metric_and_params(train_data)
            else:
                # Get metric and parameters by name
                tuner_loss, loss_params = self.tuner_metric_by_name(metric_name=tuner_metric,
                                                                    train_data=train_data,
                                                                    task=task)

            iterations = 20 if learning_time is None else 1000
            learning_time_for_tuning = learning_time / 2

            # Tune all nodes in the chain
            chain_for_return.fine_tune_all_nodes(loss_function=tuner_loss,
                                                 loss_params=loss_params,
                                                 input_data=train_data,
                                                 iterations=iterations, max_lead_time=learning_time_for_tuning)

        logger.message('Model composition finished')

        history = gp_composer.optimiser.history

        return chain_for_return, best_candidates, history

    def tuner_metric_by_name(self, metric_name, train_data: InputData, task: Task):
        """ Function allow to obtain metric for tuner by its name

        :param metric_name: name of metric
        :param train_data: InputData for train
        :param task: task to solve

        :return tuner_loss: loss function for tuner
        :return loss_params: parameters for tuner loss (can be None in some cases)
        """
        loss_params = None
        if type(metric_name) is not str:
            tuner_loss = metric_name
        else:
            tuner_loss = self.get_tuner_metrics_mapping(metric_name)
        if tuner_loss is None:
            raise ValueError(f'Incorrect tuner metric {tuner_loss}')

        if metric_name == 'rmse':
            loss_params = {'squared': False}
        elif metric_name == 'roc_auc' and task == TaskTypesEnum.classification:
            amount_of_classes = len(np.unique(np.array(train_data.target)))
            if amount_of_classes == 2:
                # Binary classification
                loss_params = None
            else:
                # Metric for multiclass classification
                loss_params = {'multi_class': 'ovr'}
        return tuner_loss, loss_params
