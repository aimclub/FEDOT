from datetime import timedelta
from functools import partial
from typing import Callable, ClassVar

from hyperopt import fmin, space_eval, tpe

from fedot.core.optimisers.objective import PipelineObjectiveEvaluate
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.tuning.search_space import SearchSpace, convert_params
from fedot.core.pipelines.tuning.tuner_interface import HyperoptTuner


class PipelineTuner(HyperoptTuner):
    """
    Class for hyperparameters optimization for all nodes simultaneously
    """

    def __init__(self, task,
                 iterations=100, early_stopping_rounds=None,
                 timeout: timedelta = timedelta(minutes=5),
                 search_space: ClassVar = SearchSpace(),
                 algo: Callable = tpe.suggest,
                 n_jobs: int = -1):
        super().__init__(task=task,
                         iterations=iterations, early_stopping_rounds=early_stopping_rounds,
                         timeout=timeout,
                         search_space=search_space,
                         algo=algo,
                         n_jobs=n_jobs)

    def tune(self, pipeline: Pipeline, objective_evaluate: PipelineObjectiveEvaluate,
             show_progress: bool = True):
        """ Function for hyperparameters tuning on the entire pipeline """

        parameters_dict = self._get_parameters_for_tune(pipeline)

        # is_need_to_maximize = self._greater_is_better(metric_function=objective_evaluate._objective.metrics[0])
        # self.is_need_to_maximize = is_need_to_maximize
        # TODO: refactor
        self.is_need_to_maximize = False
        # Check source metrics for data
        self.init_check(pipeline, objective_evaluate)

        self.pipeline.replace_n_jobs_in_nodes(n_jobs=self.n_jobs)

        best = fmin(partial(self._objective, pipeline=pipeline, objective_evaluate=objective_evaluate),
                    parameters_dict,
                    algo=self.algo,
                    max_evals=self.iterations,
                    show_progressbar=show_progress,
                    early_stop_fn=self.early_stop_fn,
                    timeout=self.max_seconds)

        best = space_eval(space=parameters_dict, hp_assignment=best)

        tuned_pipeline = self.set_arg_pipeline(pipeline=pipeline,
                                               parameters=best)

        # Validation is the optimization do well
        final_pipeline = self.final_check(tuned_pipeline, objective_evaluate)

        return final_pipeline

    def _get_parameters_for_tune(self, pipeline: Pipeline) -> dict:
        """
        Function for defining the search space

        :return parameters_dict: dictionary with operation names and parameters
        """

        parameters_dict = {}
        for node_id, node in enumerate(pipeline.nodes):
            operation_name = node.operation.operation_type

            # Assign unique prefix for each model hyperparameter
            # label - number of node in the pipeline
            node_params = self.search_space.get_node_params(node_id=node_id,
                                                            operation_name=operation_name)

            parameters_dict.update({node_id: node_params})

        return parameters_dict

    def _objective(self, parameters_dict: dict, pipeline: Pipeline, objective_evaluate: PipelineObjectiveEvaluate) \
            -> float:
        """
        Objective function for minimization / maximization problem

        :param pipeline: pipeline to optimize
        :param objective_evaluate: PipelineObjectiveEvaluate to evaluate the pipeline
        :param parameters_dict: Dict which contains new hyperparameters of the pipeline

        :return metric_value: value of objective function
        """

        # Set hyperparameters for every node
        pipeline = self.set_arg_pipeline(pipeline=pipeline, parameters=parameters_dict)

        metric_value = self.get_metric_value(pipeline=pipeline,
                                             objective_evaluate=objective_evaluate)
        return metric_value

    @staticmethod
    def set_arg_pipeline(pipeline, parameters):
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
                pipeline.nodes[node_id].custom_params = new_params

        return pipeline
