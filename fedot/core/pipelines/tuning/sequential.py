from datetime import timedelta
from functools import partial

import numpy as np
from hyperopt import fmin, space_eval, tpe

from fedot.core.data.data_split import train_test_data_setup
from fedot.core.log import Log
from fedot.core.pipelines.tuning.hyperparams import convert_params, get_node_params
from fedot.core.pipelines.tuning.tuner_interface import HyperoptTuner, _greater_is_better


class SequentialTuner(HyperoptTuner):
    """
    Class for hyperparameters optimization for all nodes sequentially
    """

    def __init__(self, pipeline, task, iterations=100,
                 timeout: timedelta = timedelta(minutes=5),
                 inverse_node_order=False, log: Log = None):
        super().__init__(pipeline, task, iterations, timeout, log)
        self.inverse_node_order = inverse_node_order

    def tune_pipeline(self, input_data, loss_function, loss_params=None):
        """ Method for hyperparameters sequential tuning """

        # Train test split
        train_input, predict_input = train_test_data_setup(input_data)
        test_target = np.array(predict_input.target)

        is_need_to_maximize = _greater_is_better(target=test_target,
                                                 loss_function=loss_function,
                                                 loss_params=loss_params)
        self.is_need_to_maximize = is_need_to_maximize

        # Check source metrics for data
        self.init_check(train_input, predict_input, test_target,
                        loss_function, loss_params)

        # Calculate amount of iterations we can apply per node
        nodes_amount = len(self.pipeline.nodes)
        iterations_per_node = round(self.iterations / nodes_amount)
        iterations_per_node = int(iterations_per_node)
        if iterations_per_node == 0:
            iterations_per_node = 1

        # Calculate amount of seconds we can apply per node
        seconds_per_node = round(self.max_seconds / nodes_amount)
        seconds_per_node = int(seconds_per_node)

        # Tuning performed sequentially for every node - so get ids of nodes
        nodes_ids = self.get_nodes_order(nodes_amount=nodes_amount)
        for node_id in nodes_ids:
            node = self.pipeline.nodes[node_id]
            operation_name = str(node.operation)

            # Get node's parameters to optimize
            node_params = get_node_params(node_id=node_id,
                                          operation_name=operation_name)

            if node_params is None:
                print(f'"{operation_name}" operation has no parameters to optimize')
            else:
                # Apply tuning for current node
                self._optimize_node(node_id=node_id,
                                    train_input=train_input,
                                    predict_input=predict_input,
                                    test_target=test_target,
                                    node_params=node_params,
                                    iterations_per_node=iterations_per_node,
                                    seconds_per_node=seconds_per_node,
                                    loss_function=loss_function,
                                    loss_params=loss_params)

        # Validation is the optimization do well
        final_pipeline = self.final_check(train_input=train_input,
                                          predict_input=predict_input,
                                          test_target=test_target,
                                          tuned_pipeline=self.pipeline,
                                          loss_function=loss_function,
                                          loss_params=loss_params)

        return final_pipeline

    def tune_node(self, input_data, loss_function, node_index, loss_params=None):
        """ Method for hyperparameters tuning for particular node"""
        # Train test split
        train_input, predict_input = train_test_data_setup(input_data)
        test_target = np.array(predict_input.target)

        is_need_to_maximize = _greater_is_better(target=test_target,
                                                 loss_function=loss_function,
                                                 loss_params=loss_params)
        self.is_need_to_maximize = is_need_to_maximize

        # Check source metrics for data
        self.init_check(train_input, predict_input, test_target,
                        loss_function, loss_params)

        node = self.pipeline.nodes[node_index]
        operation_name = str(node.operation.operation_type)

        # Get node's parameters to optimize
        node_params = get_node_params(node_id=node_index,
                                      operation_name=operation_name)

        if node_params is None:
            print(f'"{operation_name}" operation has no parameters to optimize')
        else:
            # Apply tuning for current node
            self._optimize_node(node_id=node_index,
                                train_input=train_input,
                                predict_input=predict_input,
                                test_target=test_target,
                                node_params=node_params,
                                iterations_per_node=self.iterations,
                                seconds_per_node=self.max_seconds,
                                loss_function=loss_function,
                                loss_params=loss_params)

        # Validation is the optimization do well
        final_pipeline = self.final_check(train_input=train_input,
                                          predict_input=predict_input,
                                          test_target=test_target,
                                          tuned_pipeline=self.pipeline,
                                          loss_function=loss_function,
                                          loss_params=loss_params)

        return final_pipeline

    def get_nodes_order(self, nodes_amount):
        """ Method returns list with indices of nodes in the pipeline """

        if self.inverse_node_order is True:
            # From source data to output
            nodes_ids = range(nodes_amount - 1, -1, -1)
        else:
            # From output to source data
            nodes_ids = range(0, nodes_amount)

        return nodes_ids

    def _optimize_node(self, node_id, train_input, predict_input, test_target,
                       node_params, iterations_per_node, seconds_per_node,
                       loss_function, loss_params):
        """
        Method for node optimization

        :param node_id: id of the current node in the pipeline
        :param train_input: input for train pipeline model
        :param predict_input: input for test pipeline model
        :param test_target: target for validation
        :param node_params: dictionary with parameters for node
        :param iterations_per_node: amount of iterations to produce
        :param seconds_per_node: amount of seconds to produce
        :param loss_function: loss function to minimize

        :return : updated pipeline with tuned parameters in particular node
        """
        best_parameters = fmin(partial(self._objective,
                                       pipeline=self.pipeline,
                                       node_id=node_id,
                                       train_input=train_input,
                                       predict_input=predict_input,
                                       test_target=test_target,
                                       loss_function=loss_function,
                                       loss_params=loss_params),
                               node_params,
                               algo=tpe.suggest,
                               max_evals=iterations_per_node,
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

    def _objective(self, node_params, pipeline, node_id, train_input, predict_input,
                   test_target, loss_function, loss_params: dict):
        """
        Objective function for minimization / maximization problem

        :param node_params: dictionary with parameters for node
        :param pipeline: pipeline to optimize
        :param train_input: input for train pipeline model
        :param predict_input: input for test pipeline model
        :param test_target: target for validation
        :param loss_function: loss function to optimize
        :param loss_params: parameters for loss function

        :return metric_value: value of objective function
        """

        # Set hyperparameters for node
        pipeline = SequentialTuner.set_arg_node(pipeline=pipeline,
                                                node_id=node_id,
                                                node_params=node_params)

        try:
            metric_value = SequentialTuner.get_metric_value(train_input=train_input,
                                                            predict_input=predict_input,
                                                            test_target=test_target,
                                                            pipeline=pipeline,
                                                            loss_function=loss_function,
                                                            loss_params=loss_params)
        except Exception:
            if self.is_need_to_maximize is True:
                metric_value = -999999.0
            else:
                metric_value = 999999.0

        if self.is_need_to_maximize is True:
            return -metric_value
        else:
            return metric_value
