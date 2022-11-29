from functools import partial
from typing import Tuple, Optional

from hyperopt import fmin, space_eval, Trials

from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.tuning.search_space import convert_params, get_node_operation_parameter_label
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

        parameters_dict, init_parameters = self._get_parameters_for_tune(pipeline)

        self.init_check(pipeline)

        pipeline.replace_n_jobs_in_nodes(n_jobs=self.n_jobs)

        trials = Trials()

        # try searching using initial parameters (uses original search space with fixed initial parameters)
        trials, init_trials_num = self._search_near_initial_parameters(pipeline, parameters_dict, init_parameters,
                                                                       trials, show_progress)

        best = fmin(partial(self._objective, pipeline=pipeline),
                    parameters_dict,
                    trials=trials,
                    algo=self.algo,
                    max_evals=self.iterations,
                    show_progressbar=show_progress,
                    early_stop_fn=self.early_stop_fn,
                    timeout=self.max_seconds)

        best = space_eval(space=parameters_dict, hp_assignment=best)
        # check if best point was obtained using search space with fixed initial parameters
        is_best_trial_with_init_params = trials.best_trial.get('tid') in range(init_trials_num)
        if is_best_trial_with_init_params:
            best = {**best, **init_parameters}

        tuned_pipeline = self.set_arg_pipeline(pipeline=pipeline,
                                               parameters=best)

        # Validation is the optimization do well
        final_pipeline = self.final_check(tuned_pipeline)

        return final_pipeline

    def _search_near_initial_parameters(self, pipeline: Pipeline, search_space: dict, initial_parameters: dict,
                                        trials: Trials, show_progress: bool = True):
        try_initial_parameters = initial_parameters and self.iterations > 1
        if not try_initial_parameters:
            init_trials_num = 0
            return trials, init_trials_num

        is_init_params_full = len(initial_parameters) == len(search_space)
        if self.iterations < 10 or is_init_params_full:
            init_trials_num = 1
        else:
            init_trials_num = min(int(self.iterations * 0.1), 10)

        # fmin updates trials with evaluation points tried out during the call
        fmin(partial(self._objective, pipeline=pipeline, unchangeable_parameters=initial_parameters),
             search_space,
             trials=trials,
             algo=self.algo,
             max_evals=init_trials_num,
             show_progressbar=show_progress,
             early_stop_fn=self.early_stop_fn,
             timeout=self.max_seconds)
        return trials, init_trials_num

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
            tunable_initial_params = {get_node_operation_parameter_label(node_id, operation_name, p):
                                      node.parameters[p] for p in node.parameters if p in tunable_node_params}
            if tunable_initial_params:
                initial_parameters.update(tunable_initial_params)

        return parameters_dict, initial_parameters

    def _objective(self, parameters_dict: dict, pipeline: Pipeline, unchangeable_parameters: Optional[dict] = None) \
            -> float:
        """
        Objective function for minimization / maximization problem

        Args:
            parameters_dict: Dict which contains new pipeline hyperparameters
            pipeline: pipeline to optimize
            unchangeable_parameters: Dict with parameters that should not be changed

        Returns:
             metric_value: value of objective function
        """

        # replace new parameters with parameters
        if unchangeable_parameters:
            parameters_dict = {**parameters_dict, **unchangeable_parameters}
        # Set hyperparameters for every node
        pipeline = self.set_arg_pipeline(pipeline, parameters_dict)
        metric_value = self.get_metric_value(pipeline=pipeline)
        return metric_value

    @staticmethod
    def set_arg_pipeline(pipeline: Pipeline, parameters: dict) -> Pipeline:
        """ Method for parameters setting to a pipeline

        Args:
            pipeline: pipeline to which parameters should ba assigned
            parameters: dictionary with parameters to set

        Returns:
            pipeline: pipeline with new hyperparameters in each node
        """
        # Set hyperparameters for every node
        for node_id, node in enumerate(pipeline.nodes):
            node_params = {key: value for key, value in parameters.items()
                           if key.startswith(f'{str(node_id)} || {node.name}')}

            if node_params is not None:
                # Delete all prefix strings to get appropriate parameters names
                new_params = convert_params(node_params)

                # Update parameters in nodes
                pipeline.nodes[node_id].parameters = new_params

        return pipeline
