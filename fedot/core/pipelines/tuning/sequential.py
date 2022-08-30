from datetime import timedelta
from functools import partial
from typing import Callable, ClassVar

from hyperopt import fmin, space_eval, tpe

from fedot.core.pipelines.tuning.search_space import SearchSpace, convert_params
from fedot.core.pipelines.tuning.tuner_interface import HyperoptTuner, _greater_is_better


class SequentialTuner(HyperoptTuner):
    """
    Class for hyperparameters optimization for all nodes sequentially
    """

    def __init__(self, pipeline, task,
                 iterations=100, early_stopping_rounds=None,
                 timeout: timedelta = timedelta(minutes=5),
                 inverse_node_order=False,
                 search_space: ClassVar = SearchSpace(),
                 algo: Callable = tpe.suggest,
                 n_jobs: int = -1):
        super().__init__(pipeline=pipeline, task=task,
                         iterations=iterations, early_stopping_rounds=early_stopping_rounds,
                         timeout=timeout,
                         search_space=search_space,
                         algo=algo,
                         n_jobs=n_jobs)
        self.inverse_node_order = inverse_node_order

    def tune_pipeline(self, input_data, loss_function,
                      cv_folds: int = None, validation_blocks: int = None):
        """ Method for hyperparameters sequential tuning """
        # Define folds for cross validation
        self.cv_folds = cv_folds
        self.validation_blocks = validation_blocks
        is_need_to_maximize = _greater_is_better(loss_function=loss_function)
        self.is_need_to_maximize = is_need_to_maximize

        # Check source metrics for data
        self.init_check(input_data, loss_function)

        self.pipeline.replace_n_jobs_in_nodes(n_jobs=self.n_jobs)

        # Calculate amount of iterations we can apply per node
        nodes_amount = self.pipeline.length
        iterations_per_node = round(self.iterations / nodes_amount)
        iterations_per_node = int(iterations_per_node)
        if iterations_per_node == 0:
            iterations_per_node = 1

        # Calculate amount of seconds we can apply per node
        if self.max_seconds is not None:
            seconds_per_node = round(self.max_seconds / nodes_amount)
            seconds_per_node = int(seconds_per_node)
        else:
            seconds_per_node = None

        # Tuning performed sequentially for every node - so get ids of nodes
        nodes_ids = self.get_nodes_order(nodes_number=nodes_amount)
        for node_id in nodes_ids:
            node = self.pipeline.nodes[node_id]
            operation_name = node.operation.operation_type

            # Get node's parameters to optimize
            node_params = self.search_space.get_node_params(node_id=node_id,
                                                            operation_name=operation_name)

            if node_params is None:
                self.log.info(f'"{operation_name}" operation has no parameters to optimize')
            else:
                # Apply tuning for current node
                self._optimize_node(node_id=node_id,
                                    data=input_data,
                                    node_params=node_params,
                                    iterations_per_node=iterations_per_node,
                                    seconds_per_node=seconds_per_node,
                                    loss_function=loss_function)

        # Validation is the optimization do well
        final_pipeline = self.final_check(data=input_data,
                                          tuned_pipeline=self.pipeline,
                                          loss_function=loss_function)

        return final_pipeline

    def tune_node(self, input_data, loss_function, node_index,
                  cv_folds: int = None, validation_blocks: int = None):
        """ Method for hyperparameters tuning for particular node"""
        self.cv_folds = cv_folds
        self.validation_blocks = validation_blocks
        is_need_to_maximize = _greater_is_better(loss_function=loss_function)
        self.is_need_to_maximize = is_need_to_maximize

        # Check source metrics for data
        self.init_check(input_data, loss_function)

        node = self.pipeline.nodes[node_index]
        operation_name = node.operation.operation_type

        # Get node's parameters to optimize
        node_params = self.search_space.get_node_params(node_id=node_index,
                                                        operation_name=operation_name)

        if node_params is None:
            self.log.info(f'"{operation_name}" operation has no parameters to optimize')
        else:
            # Apply tuning for current node
            self._optimize_node(node_id=node_index,
                                data=input_data,
                                node_params=node_params,
                                iterations_per_node=self.iterations,
                                seconds_per_node=self.max_seconds,
                                loss_function=loss_function)

        # Validation is the optimization do well
        final_pipeline = self.final_check(data=input_data,
                                          tuned_pipeline=self.pipeline,
                                          loss_function=loss_function)
        return final_pipeline

    def get_nodes_order(self, nodes_number):
        """ Method returns list with indices of nodes in the pipeline """

        if self.inverse_node_order is True:
            # From source data to output
            nodes_ids = range(nodes_number - 1, -1, -1)
        else:
            # From output to source data
            nodes_ids = range(0, nodes_number)

        return nodes_ids

    def _optimize_node(self, node_id, data, node_params, iterations_per_node,
                       seconds_per_node, loss_function: Callable):
        """
        Method for node optimization

        :param node_id: id of the current node in the pipeline
        :param data: InputData for validation
        :param node_params: dictionary with parameters for node
        :param iterations_per_node: amount of iterations to produce
        :param seconds_per_node: amount of seconds to produce
        :param loss_function: loss function to minimize

        :return : updated pipeline with tuned parameters in particular node
        """
        best_parameters = fmin(partial(self._objective,
                                       pipeline=self.pipeline,
                                       node_id=node_id,
                                       data=data,
                                       loss_function=loss_function),
                               node_params,
                               algo=self.algo,
                               max_evals=iterations_per_node,
                               early_stop_fn=self.early_stop_fn,
                               timeout=seconds_per_node)

        best_parameters = space_eval(space=node_params,
                                     hp_assignment=best_parameters)

        # Set best params for this node in the pipeline
        self.pipeline = self.set_arg_node(pipeline=self.pipeline,
                                          node_id=node_id,
                                          node_params=best_parameters)
        return self.pipeline

    @staticmethod
    def set_arg_node(pipeline, node_id, node_params):
        """ Method for parameters setting to a pipeline

        :param pipeline: pipeline with nodes
        :param node_id: id of the node to which parameters should ba assigned
        :param node_params: dictionary with labeled parameters to set
        :return pipeline: pipeline with new hyperparameters in each node
        """

        # Remove label prefixes
        node_params = convert_params(node_params)

        # Update parameters in nodes
        pipeline.nodes[node_id].custom_params = node_params

        return pipeline

    def _objective(self, node_params, pipeline, node_id, data, loss_function):
        """
        Objective function for minimization / maximization problem

        :param node_params: dictionary with parameters for node
        :param pipeline: pipeline to optimize
        :param data: InputData for validation
        :param loss_function: loss function to optimize

        :return metric_value: value of objective function
        """

        # Set hyperparameters for node
        pipeline = self.set_arg_node(pipeline=pipeline, node_id=node_id,
                                     node_params=node_params)

        metric_value = self.get_metric_value(data=data,
                                             pipeline=pipeline,
                                             loss_function=loss_function)
        return metric_value
