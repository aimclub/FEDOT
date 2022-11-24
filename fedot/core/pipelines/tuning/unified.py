from functools import partial
from typing import Tuple

from hyperopt import fmin, space_eval, hp, Trials

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

        parameters_dict, init_parameters, is_init_params_full = self._get_parameters_for_tune(pipeline)
        self.init_check(pipeline)

        pipeline.replace_n_jobs_in_nodes(n_jobs=self.n_jobs)

        trials = Trials()

        # try searching using initial parameters (uses original search space with fixed initial parameters)
        try_initial_parameters = init_parameters and self.iterations > 1

        if try_initial_parameters:
            trials, init_trials_num = self._search_near_initial_parameters(pipeline, init_parameters,
                                                                           is_init_params_full, trials,
                                                                           show_progress)

        best = fmin(partial(self._objective, pipeline=pipeline),
                    parameters_dict,
                    trials=trials,
                    algo=self.algo,
                    max_evals=self.iterations,
                    show_progressbar=show_progress,
                    early_stop_fn=self.early_stop_fn,
                    timeout=self.max_seconds)

        # check if best point was obtained using search space with fixed initial parameters
        if try_initial_parameters:
            is_best_trial_with_init_params = trials.best_trial.get('tid') in range(init_trials_num)
            # replace search space
            parameters_dict = init_parameters if is_best_trial_with_init_params else parameters_dict

        best = space_eval(space=parameters_dict, hp_assignment=best)

        tuned_pipeline = self.set_arg_pipeline(pipeline=pipeline,
                                               parameters=best)

        # Validation is the optimization do well
        final_pipeline = self.final_check(tuned_pipeline)

        return final_pipeline

    def _search_near_initial_parameters(self, pipeline: Pipeline, initial_parameters: dict,
                                        is_init_parameters_full: bool, trials: Trials,
                                        show_progress: bool = True):
        if self.iterations >= 10 and not is_init_parameters_full:
            init_trials_num = min(int(self.iterations * 0.1), 10)
        else:
            init_trials_num = 1

        # fmin updates trials with evaluation points tried out during the call
        fmin(partial(self._objective, pipeline=pipeline),
             initial_parameters,
             trials=trials,
             algo=self.algo,
             max_evals=init_trials_num,
             show_progressbar=show_progress,
             early_stop_fn=self.early_stop_fn,
             timeout=self.max_seconds)
        return trials, init_trials_num

    def _get_parameters_for_tune(self, pipeline: Pipeline) -> Tuple[dict, dict, bool]:
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

        # create search space with fixed initial parameters
        init_params_space = {}
        is_init_params_full = len(initial_parameters) == len(parameters_dict)
        if initial_parameters:
            for key in parameters_dict:
                if key in initial_parameters:
                    value = initial_parameters[key]
                    # fix possible value for initial parameter (the value will be chosen with probability=1)
                    init_params_space[key] = hp.pchoice(key, [(1, value)])
                else:
                    init_params_space[key] = parameters_dict[key]

        return parameters_dict, init_params_space, is_init_params_full

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
            node_params = {key: value for key, value in parameters.items() if key.startswith(str(node_id))}

            if node_params is not None:
                # Delete all prefix strings to get appropriate parameters names
                new_params = convert_params(node_params)

                # Update parameters in nodes
                pipeline.nodes[node_id].parameters = new_params

        return pipeline
