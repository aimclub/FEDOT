from datetime import timedelta
from functools import partial

import numpy as np
from hyperopt import fmin, tpe, space_eval

from fedot.core.chains.tuning.hyperparams import get_node_params, convert_params
from fedot.core.chains.tuning.tuner_interface import HyperoptTuner, _greater_is_better
from fedot.core.data.data import train_test_data_setup
from fedot.core.log import Log

MAX_METRIC_VALUE = 10e6


class ChainTuner(HyperoptTuner):
    """
    Class for hyperparameters optimization for all nodes simultaneously
    """

    def __init__(self, chain, task, iterations=100,
                 max_lead_time: timedelta = timedelta(minutes=5),
                 log: Log = None):
        super().__init__(chain, task, iterations, max_lead_time, log)

    def tune_chain(self, input_data, loss_function, loss_params=None):
        """ Function for hyperparameters tuning on the entire chain """

        parameters_dict = self._get_parameters_for_tune(self.chain)

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
                            chain=self.chain,
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

        tuned_chain = self.set_arg_chain(chain=self.chain,
                                         parameters=best)

        # Validation is the optimization do well
        final_chain = self.final_check(train_input=train_input,
                                       predict_input=predict_input,
                                       test_target=test_target,
                                       tuned_chain=tuned_chain,
                                       loss_function=loss_function,
                                       loss_params=loss_params)

        return final_chain

    @staticmethod
    def set_arg_chain(chain, parameters):
        """ Method for parameters setting to a chain

        :param chain: chain to which parameters should ba assigned
        :param parameters: dictionary with parameters to set
        :return chain: chain with new hyperparameters in each node
        """

        # Set hyperparameters for every node
        for node_id, _ in enumerate(chain.nodes):
            node_params = parameters.get(node_id)

            if node_params is not None:
                # Delete all prefix strings to get appropriate parameters names
                new_params = convert_params(node_params)

                # Update parameters in nodes
                chain.nodes[node_id].custom_params = new_params

        return chain

    @staticmethod
    def _get_parameters_for_tune(chain):
        """
        Function for defining the search space

        :param chain: chain to optimize
        :return parameters_dict: dictionary with operation names and parameters
        """

        parameters_dict = {}
        for node_id, node in enumerate(chain.nodes):
            operation_name = str(node.operation)

            # Assign unique prefix for each model hyperparameter
            # label - number of node in the chain
            node_params = get_node_params(node_id=node_id,
                                          operation_name=operation_name)

            parameters_dict.update({node_id: node_params})

        return parameters_dict

    def _objective(self, parameters_dict, chain, train_input, predict_input,
                   test_target, loss_function, loss_params: dict):
        """
        Objective function for minimization / maximization problem

        :param parameters_dict: dictionary with operation names and parameters
        :param chain: chain to optimize
        :param train_input: input for train chain model
        :param predict_input: input for test chain model
        :param test_target: target for validation
        :param loss_function: loss function to optimize
        :param loss_params: parameters for loss function

        :return metric_value: value of objective function
        """

        # Set hyperparameters for every node
        chain = ChainTuner.set_arg_chain(chain=chain, parameters=parameters_dict)

        try:
            metric_value = ChainTuner.get_metric_value(train_input=train_input,
                                                       predict_input=predict_input,
                                                       test_target=test_target,
                                                       chain=chain,
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
