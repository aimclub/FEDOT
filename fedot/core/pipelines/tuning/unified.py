from functools import partial
from typing import Tuple

from hyperopt import fmin, space_eval

from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.tuning.search_space import convert_params
from fedot.core.pipelines.tuning.tuner_interface import HyperoptTuner


class PipelineTuner(HyperoptTuner):
    """
    Class for hyperparameters optimization for all nodes simultaneously
    """

    def tune(self, pipeline: Pipeline, show_progress: bool = True) -> Pipeline:
        """ Function for hyperparameters tuning on the entire pipeline

        :param pipeline: Pipeline which hyperparameters will be tuned
        :param show_progress: shows progress of tuning if true
        """
        parameters_dict, initial_params = self._get_parameters_for_tune(pipeline)

        # Check source metrics for data
        self.init_check(pipeline)

        pipeline.replace_n_jobs_in_nodes(n_jobs=self.n_jobs)

        # trials = generate_trials_to_calculate([initial_params])

        best = fmin(partial(self._objective, pipeline=pipeline),
                    parameters_dict,
                    points_to_evaluate=[initial_params],
                    # trials=trials,
                    algo=self.algo,
                    max_evals=self.iterations,
                    show_progressbar=show_progress,
                    early_stop_fn=self.early_stop_fn,
                    timeout=self.max_seconds)

        best = space_eval(space=parameters_dict, hp_assignment=best)

        tuned_pipeline = self.set_arg_pipeline(pipeline=pipeline,
                                               parameters=best)

        # Validation is the optimization do well
        final_pipeline = self.final_check(tuned_pipeline)

        return final_pipeline

    def _get_parameters_for_tune(self, pipeline: Pipeline) -> Tuple[dict, dict]:
        """
        Function for defining the search space

        :return parameters_dict: dictionary with operation names and parameters
        """

        parameters_dict = {}
        initial_parameters = {}
        for node_id, node in enumerate(pipeline.nodes):
            operation_name = node.operation.operation_type

            # Assign unique prefix for each model hyperparameter
            # label - number of node in the pipeline
            node_params = self.search_space.get_node_params(node_id=node_id,
                                                            operation_name=operation_name)

            if node_params is not None:
                parameters_dict.update(node_params)

            tunable_node_params = self.search_space.get_operation_parameter_range(operation_name)
            tunable_initial_params = {f'{node_id} || {operation_name} | {p}':
                                          node.parameters[p] for p in node.parameters if p in tunable_node_params}
            if tunable_initial_params:
                initial_parameters.update(tunable_initial_params)

        return parameters_dict, initial_parameters

    def _objective(self, parameters_dict: dict, pipeline: Pipeline) \
            -> float:
        """
        Objective function for minimization / maximization problem

        :param parameters_dict: Dict which contains new hyperparameters of the pipeline
        :param pipeline: pipeline to optimize

        :return metric_value: value of objective function
        """

        # Set hyperparameters for every node
        pipeline = self.set_arg_pipeline(pipeline=pipeline, parameters=parameters_dict)

        metric_value = self.get_metric_value(pipeline=pipeline)
        return metric_value

    @staticmethod
    def set_arg_pipeline(pipeline: Pipeline, parameters: dict) -> Pipeline:
        """ Method for parameters setting to a pipeline

        :param pipeline: pipeline to which parameters should ba assigned
        :param parameters: dictionary with parameters to set

        :return pipeline: pipeline with new hyperparameters in each node
        """

        # Set hyperparameters for every node
        for node_id, _ in enumerate(pipeline.nodes):
            node_params = parameters.get(node_id)

            if node_params is not None:
                # Delete all prefix strings to get appropriate parameters names
                new_params = convert_params(node_params)

                # Update parameters in nodes
                pipeline.nodes[node_id].parameters = new_params

        return pipeline
