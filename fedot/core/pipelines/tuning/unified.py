from datetime import timedelta
from functools import partial
from typing import Callable, ClassVar
from hyperopt import fmin, tpe, space_eval

from fedot.core.log import Log
from fedot.core.pipelines.tuning.search_space import convert_params, SearchSpace
from fedot.core.pipelines.tuning.tuner_interface import HyperoptTuner, _greater_is_better


class PipelineTuner(HyperoptTuner):
    """
    Class for hyperparameters optimization for all nodes simultaneously
    """

    def __init__(self, pipeline, task, iterations=100,
                 timeout: timedelta = timedelta(minutes=5),
                 log: Log = None,
                 search_space: ClassVar = SearchSpace(),
                 algo: Callable = tpe.suggest):
        super().__init__(pipeline, task, iterations, timeout, log, search_space, algo)

    def tune_pipeline(self, input_data, loss_function, loss_params=None,
                      cv_folds: int = None, validation_blocks: int = None):
        """ Function for hyperparameters tuning on the entire pipeline """

        self.validation_blocks = validation_blocks
        # Define folds for cross validation
        self.cv_folds = cv_folds

        parameters_dict = self._get_parameters_for_tune()

        is_need_to_maximize = _greater_is_better(target=input_data.target,
                                                 loss_function=loss_function,
                                                 loss_params=loss_params,
                                                 data_type=input_data.data_type)
        self.is_need_to_maximize = is_need_to_maximize

        # Check source metrics for data
        self.init_check(input_data, loss_function, loss_params)

        best = fmin(partial(self._objective,
                            pipeline=self.pipeline,
                            data=input_data,
                            loss_function=loss_function,
                            loss_params=loss_params),
                    parameters_dict,
                    algo=self.algo,
                    max_evals=self.iterations,
                    timeout=self.max_seconds)

        best = space_eval(space=parameters_dict, hp_assignment=best)

        tuned_pipeline = self.set_arg_pipeline(pipeline=self.pipeline,
                                               parameters=best)

        # Validation is the optimization do well
        final_pipeline = self.final_check(data=input_data,
                                          tuned_pipeline=tuned_pipeline,
                                          loss_function=loss_function,
                                          loss_params=loss_params)

        return final_pipeline

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

    def _get_parameters_for_tune(self):
        """
        Function for defining the search space

        :return parameters_dict: dictionary with operation names and parameters
        """

        parameters_dict = {}
        for node_id, node in enumerate(self.pipeline.nodes):
            operation_name = node.operation.operation_type

            # Assign unique prefix for each model hyperparameter
            # label - number of node in the pipeline
            node_params = self.search_space.get_node_params(node_id=node_id,
                                                            operation_name=operation_name)

            parameters_dict.update({node_id: node_params})

        return parameters_dict

    def _objective(self, parameters_dict, pipeline, data, loss_function,
                   loss_params: dict):
        """
        Objective function for minimization / maximization problem

        :param parameters_dict: dictionary with operation names and parameters
        :param pipeline: pipeline to optimize
        :param data: InputData for model train and validation
        :param loss_function: loss function to optimize
        :param loss_params: parameters for loss function

        :return metric_value: value of objective function
        """

        # Set hyperparameters for every node
        pipeline = self.set_arg_pipeline(pipeline=pipeline, parameters=parameters_dict)

        metric_value = self.get_metric_value(data=data,
                                             pipeline=pipeline,
                                             loss_function=loss_function,
                                             loss_params=loss_params)
        return metric_value
