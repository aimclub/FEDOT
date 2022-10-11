from datetime import timedelta
from functools import partial
from typing import Callable, ClassVar

from hyperopt import fmin, space_eval, tpe

from fedot.core.optimisers.objective import PipelineObjectiveEvaluate
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.tuning.search_space import SearchSpace, convert_params
from fedot.core.pipelines.tuning.tuner_interface import HyperoptTuner


class SequentialTuner(HyperoptTuner):
    """
    Class for hyperparameters optimization for all nodes sequentially
    """

    def __init__(self, objective_evaluate: PipelineObjectiveEvaluate,
                 iterations=100, early_stopping_rounds=None,
                 timeout: timedelta = timedelta(minutes=5),
                 inverse_node_order=False,
                 search_space: ClassVar = SearchSpace(),
                 algo: Callable = tpe.suggest,
                 n_jobs: int = -1):
        super().__init__(objective_evaluate, iterations, early_stopping_rounds, timeout, search_space, algo, n_jobs)

        self.inverse_node_order = inverse_node_order

    def tune(self, pipeline: Pipeline) -> Pipeline:
        """ Function for hyperparameters tuning on the entire pipeline

        :param pipeline: Pipeline which hyperparameters will be tuned
        """
        # Check source metrics for data
        self.init_check(pipeline)

        pipeline.replace_n_jobs_in_nodes(n_jobs=self.n_jobs)

        # Calculate amount of iterations we can apply per node
        nodes_amount = pipeline.length
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
            node = pipeline.nodes[node_id]
            operation_name = node.operation.operation_type

            # Get node's parameters to optimize
            node_params = self.search_space.get_node_params(node_id=node_id,
                                                            operation_name=operation_name)

            if node_params is None:
                self.log.info(f'"{operation_name}" operation has no parameters to optimize')
            else:
                # Apply tuning for current node
                self._optimize_node(node_id=node_id,
                                    pipeline=pipeline,
                                    node_params=node_params,
                                    iterations_per_node=iterations_per_node,
                                    seconds_per_node=seconds_per_node)

        # Validation is the optimization do well
        final_pipeline = self.final_check(tuned_pipeline=pipeline)

        return final_pipeline

    def get_nodes_order(self, nodes_number: int) -> range:
        """ Method returns list with indices of nodes in the pipeline """

        if self.inverse_node_order is True:
            # From source data to output
            nodes_ids = range(nodes_number - 1, -1, -1)
        else:
            # From output to source data
            nodes_ids = range(0, nodes_number)

        return nodes_ids

    def tune_node(self, pipeline: Pipeline, node_index: int) -> Pipeline:
        """ Method for hyperparameters tuning for particular node

        :param pipeline: Pipeline which contains a node to be tuned
        :param node_index: Index of the node to tune
        """
        # Check source metrics for data
        self.init_check(pipeline)

        node = pipeline.nodes[node_index]
        operation_name = node.operation.operation_type

        # Get node's parameters to optimize
        node_params = self.search_space.get_node_params(node_id=node_index,
                                                        operation_name=operation_name)

        if node_params is None:
            self.log.info(f'"{operation_name}" operation has no parameters to optimize')
        else:
            # Apply tuning for current node
            self._optimize_node(pipeline=pipeline,
                                node_id=node_index,
                                node_params=node_params,
                                iterations_per_node=self.iterations,
                                seconds_per_node=self.max_seconds,
                                )

        # Validation is the optimization do well
        final_pipeline = self.final_check(tuned_pipeline=pipeline)
        return final_pipeline

    def _optimize_node(self, pipeline: Pipeline, node_id: int, node_params: dict, iterations_per_node: int,
                       seconds_per_node: int) -> Pipeline:
        """
        Method for node optimization

        :param pipeline: Pipeline which node is optimized
        :param node_id: id of the current node in the pipeline
        :param node_params: dictionary with parameters for node
        :param iterations_per_node: amount of iterations to produce
        :param seconds_per_node: amount of seconds to produce

        :return : updated pipeline with tuned parameters in particular node
        """
        best_parameters = fmin(partial(self._objective,
                                       pipeline=pipeline,
                                       node_id=node_id
                                       ),
                               node_params,
                               algo=self.algo,
                               max_evals=iterations_per_node,
                               early_stop_fn=self.early_stop_fn,
                               timeout=seconds_per_node)

        best_parameters = space_eval(space=node_params,
                                     hp_assignment=best_parameters)

        # Set best params for this node in the pipeline
        self.pipeline = self.set_arg_node(pipeline=pipeline,
                                          node_id=node_id,
                                          node_params=best_parameters)
        return self.pipeline

    def _objective(self, node_params: dict, pipeline: Pipeline, node_id: int) -> float:
        """
        Objective function for minimization / maximization problem

        :param node_params: dictionary with parameters for node
        :param pipeline: pipeline to evaluate
        :param node_id: id of the node to which parameters should be assigned

        :return metric_value: value of objective function
        """

        # Set hyperparameters for node
        pipeline = self.set_arg_node(pipeline=pipeline, node_id=node_id,
                                     node_params=node_params)

        metric_value = self.get_metric_value(pipeline=pipeline)
        return metric_value

    @staticmethod
    def set_arg_node(pipeline: Pipeline, node_id: int, node_params: dict) -> Pipeline:
        """ Method for parameters setting to a pipeline

        :param pipeline: pipeline which contains the node
        :param node_id: id of the node to which parameters should be assigned
        :param node_params: dictionary with labeled parameters to set

        :return pipeline: pipeline with new hyperparameters in each node
        """

        # Remove label prefixes
        node_params = convert_params(node_params)

        # Update parameters in nodes
        pipeline.nodes[node_id].parameters = node_params

        return pipeline
