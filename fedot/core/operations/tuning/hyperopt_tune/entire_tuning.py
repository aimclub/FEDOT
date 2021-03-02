import datetime

import numpy as np

from functools import partial

from fedot.core.operations.tuning.hyperopt_tune.hp_hyperparams import get_node_params, convert_params
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum

from hyperopt import fmin, tpe, space_eval
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


class ChainTuner:
    """
    Class for hyperparameters optimization for all nodes simultaneously

    :param chain: chain to optimize
    :param max_lead_time: max time for hyperparameters searching
    :param iterations: max number of iterations
    """

    def __init__(self,
                 chain,
                 task,
                 max_lead_time=datetime.timedelta(minutes=2),
                 iterations=100):
        self.chain = chain
        self.task = task
        self.max_lead_time = max_lead_time
        self.iterations = iterations

    def tune(self, input_data):
        """
        Function for hyperparameters tuning on the entire chain

        :param input_data: data used for hyperparameter searching
        :return fitted_chain: chain with optimized hyperparameters
        """
        parameters_dict = self._get_parameters_for_tune(self.chain)

        # Train test split
        train_input, predict_input, test_target = self._validation_split(input_data)

        best = fmin(partial(objective,
                            chain=self.chain,
                            train_input=train_input,
                            predict_input=predict_input,
                            test_target=test_target),
                    parameters_dict,
                    algo=tpe.suggest,
                    max_evals=self.iterations)

        best = space_eval(space=parameters_dict, hp_assignment=best)

        tuned_chain = self.set_arg_chain(chain=self.chain,
                                         parameters=best)

        return tuned_chain

    def _validation_split(self, input_data):
        """
        Function for applying train test split for cross validation

        :param input_data: data used for splitting

        :return train_input: part of data for training
        :return predict_input: part of data for predicting
        :return y_data_test: actual values for validation
        """

        input_features = input_data.features
        input_target = input_data.target

        # TODO refactor it
        try:
            trigger = self.task.task_params.forecast_length
        except Exception:
            trigger = None
        if trigger is not None:
            # Time series forecasting task
            forecast_length = self.task.task_params.forecast_length
            x_data_train = input_features[:-forecast_length]
            x_data_test = input_features[:-forecast_length]

            y_data_train = x_data_train
            y_data_test = input_target[-forecast_length:]

            idx_for_train = np.arange(0, len(x_data_train))
            idx_for_predict = idx_for_train
        else:
            x_data_train, x_data_test, \
            y_data_train, y_data_test = train_test_split(input_features,
                                                         input_target,
                                                         test_size=0.33)
            idx_for_train = np.arange(0, len(x_data_train))
            idx_for_predict = np.arange(0, len(x_data_test))


        # Prepare data to train the model
        train_input = InputData(idx=idx_for_train,
                                features=x_data_train,
                                target=y_data_train,
                                task=self.task,
                                data_type=DataTypesEnum.table)

        predict_input = InputData(idx=idx_for_predict,
                                  features=x_data_test,
                                  target=None,
                                  task=self.task,
                                  data_type=DataTypesEnum.table)

        return train_input, predict_input, y_data_test

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


def objective(parameters_dict, chain, train_input, predict_input, test_target):
    """
    Objective function for minimization

    :param parameters_dict: dictionary with operation names and parameters
    :param chain: chain to optimize
    :param train_input: input for train chain model
    :param predict_input: input for test chain model
    :param test_target: target for validation

    :return min_function: value of objective function
    """

    # Set hyperparameters for every node
    chain = ChainTuner.set_arg_chain(chain=chain, parameters=parameters_dict)

    # Fit it
    chain.fit_from_scratch(train_input)

    predicted_values = chain.predict(predict_input)
    preds = np.ravel(np.array(predicted_values.predict))

    min_function = mean_absolute_error(test_target, preds)
    return min_function