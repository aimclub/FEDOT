from datetime import timedelta
from functools import partial

import numpy as np
from hyperopt import fmin, space_eval, tpe

from fedot.core.data.data_split import train_test_data_setup
from fedot.core.log import Log
from fedot.core.pipelines.tuning.hyperparams import convert_params, get_node_params
from fedot.core.pipelines.tuning.tuner_interface import HyperoptTuner, _greater_is_better

MAX_METRIC_VALUE = 10e6


class PipelineTuner(HyperoptTuner):
    """
    Class for hyperparameters optimization for all nodes simultaneously
    """

    def __init__(self, pipeline, task, iterations=100,
                 timeout: timedelta = timedelta(minutes=5),
                 log: Log = None):
        super().__init__(pipeline, task, iterations, timeout, log)

    def tune_pipeline(self, input_data, loss_function, loss_params=None):
        """ Function for hyperparameters tuning on the entire pipeline """

        parameters_dict = self._get_parameters_for_tune(self.pipeline)

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

        best = fmin(partial(self._objective,
                            pipeline=self.pipeline,
                            train_input=train_input,
                            predict_input=predict_input,
                            test_target=test_target,
                            loss_function=loss_function,
                            loss_params=loss_params),
                    parameters_dict,
                    algo=tpe.suggest,
                    max_evals=self.iterations,
                    timeout=self.max_seconds)

        best = space_eval(space=parameters_dict, hp_assignment=best)

        tuned_pipeline = self.set_arg_pipeline(pipeline=self.pipeline,
                                               parameters=best)

        # Validation is the optimization do well
        final_pipeline = self.final_check(train_input=train_input,
                                          predict_input=predict_input,
                                          test_target=test_target,
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

    @staticmethod
    def _get_parameters_for_tune(pipeline):
        """
        Function for defining the search space

        :param pipeline: pipeline to optimize
        :return parameters_dict: dictionary with operation names and parameters
        """

        parameters_dict = {}
        for node_id, node in enumerate(pipeline.nodes):
            operation_name = str(node.operation)

            # Assign unique prefix for each model hyperparameter
            # label - number of node in the pipeline
            node_params = get_node_params(node_id=node_id,
                                          operation_name=operation_name)

            parameters_dict.update({node_id: node_params})

        return parameters_dict

    def _objective(self, parameters_dict, pipeline, train_input, predict_input,
                   test_target, loss_function, loss_params: dict):
        """
        Objective function for minimization / maximization problem

        :param parameters_dict: dictionary with operation names and parameters
        :param pipeline: pipeline to optimize
        :param train_input: input for train pipeline model
        :param predict_input: input for test pipeline model
        :param test_target: target for validation
        :param loss_function: loss function to optimize
        :param loss_params: parameters for loss function

        :return metric_value: value of objective function
        """

        # Set hyperparameters for every node
        pipeline = PipelineTuner.set_arg_pipeline(pipeline=pipeline, parameters=parameters_dict)

        try:
            metric_value = PipelineTuner.get_metric_value(train_input=train_input,
                                                          predict_input=predict_input,
                                                          test_target=test_target,
                                                          pipeline=pipeline,
                                                          loss_function=loss_function,
                                                          loss_params=loss_params)
        except Exception:
            if self.is_need_to_maximize is True:
                metric_value = -MAX_METRIC_VALUE
            else:
                metric_value = MAX_METRIC_VALUE

        if self.is_need_to_maximize is True:
            return -metric_value
        else:
            return metric_value
