import datetime

import numpy as np

from functools import partial

from fedot.core.operations.tuning.hyperopt_tune.hp_hyperparams import get_node_params, convert_params
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum

from hyperopt import fmin, tpe, space_eval
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


class NodesTuner:
    """
    Class for hyperparameters optimization for all nodes recursively

    :param chain: chain to optimize
    :param max_lead_time: max time for hyperparameters searching
    :param iterations: max number of iterations
    """

    def __init__(self,
                 chain,
                 task,
                 max_lead_time=datetime.timedelta(minutes=2),
                 iterations=100,
                 inverse_node_order=False):
        self.chain = chain
        self.task = task
        self.max_lead_time = max_lead_time
        self.iterations = iterations
        self.inverse_node_order = inverse_node_order

    def tune(self, input_data):
        """
        Function for hyperparameters tuning on the entire chain

        :param input_data: data used for hyperparameter searching
        :return fitted_chain: chain with optimized hyperparameters
        """

        # Train test split
        train_input, predict_input, test_target = self._validation_split(input_data)

        # Calculate amount of iterations we can apply per node
        nodes_amount = len(self.chain.nodes)
        iterations_per_node = round(self.iterations/nodes_amount)
        iterations_per_node = int(iterations_per_node)

        # Tuning performed sequentially for every node
        if self.inverse_node_order is True:
            # From source data to output
            nodes_ids = range(nodes_amount-1, -1, -1)
        else:
            # From output to source data
            nodes_ids = range(0, nodes_amount)

        for node_id in nodes_ids:
            node = self.chain.nodes[node_id]
            operation_name = str(node.operation)

            # Get node's parameters to optimize
            node_params = get_node_params(node_id=node_id,
                                          operation_name=operation_name)

            if node_params is None:
                print(f'"{operation_name}" operation has no parameters to optimize')
            else:

                # Apply tuning for current node
                best_parameters = fmin(partial(objective,
                                               chain=self.chain,
                                               node_id=node_id,
                                               train_input=train_input,
                                               predict_input=predict_input,
                                               test_target=test_target),
                                       node_params,
                                       algo=tpe.suggest,
                                       max_evals=iterations_per_node)

                best_parameters = space_eval(space=node_params,
                                             hp_assignment=best_parameters)

                # Set best params for this node in the chain
                self.chain = self.set_arg_node(chain=self.chain,
                                               node_id=node_id,
                                               node_params=best_parameters)

        return self.chain

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
    def set_arg_node(chain, node_id, node_params):
        """ Method for parameters setting to a chain

        :param chain: chain with nodes
        :param node_id: id of the node to which parameters should ba assigned
        :param node_params: dictionary with labeled parameters to set
        :return chain: chain with new hyperparameters in each node
        """

        # Remove label prefixes
        node_params = convert_params(node_params)

        # Update parameters in nodes
        chain.nodes[node_id].custom_params = node_params

        return chain


def objective(node_params, chain, node_id,
              train_input, predict_input, test_target):
    """
    Objective function for minimization

    :param node_params: dictionary with operation names and parameters
    :param chain: chain to optimize
    :param node_id: id of the node to which parameters should ba assigned
    :param train_input: input for train chain model
    :param predict_input: input for test chain model
    :param test_target: target for validation

    :return min_function: value of objective function
    """

    # Set hyperparameters for node
    chain = NodesTuner.set_arg_node(chain=chain,
                                    node_id=node_id,
                                    node_params=node_params)

    # Fit it
    chain.fit_from_scratch(train_input)

    predicted_values = chain.predict(predict_input)
    preds = np.ravel(np.array(predicted_values.predict))

    min_function = mean_absolute_error(test_target, preds)
    return min_function