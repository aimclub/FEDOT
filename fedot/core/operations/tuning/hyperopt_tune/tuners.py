import datetime
from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np

from functools import partial

from fedot.core.operations.tuning.hyperopt_tune.hp_hyperparams import get_node_params, convert_params
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TaskTypesEnum

from hyperopt import fmin, tpe, space_eval
from sklearn.model_selection import train_test_split


class HyperoptTuner(ABC):
    """
    Base class for hyperparameters optimization based on hyperopt library

    :param chain: chain to optimize
    :param task: task (classification, regression, ts_forecasting, clustering)
    :param iterations: max number of iterations
    """

    def __init__(self, chain, task, iterations=100):
        self.chain = chain
        self.task = task
        self.iterations = iterations
        self.init_chain = None
        self.init_metric = None
        self.is_need_to_maximize = None

    @abstractmethod
    def tune_chain(self, input_data, loss_function):
        """
        Function for hyperparameters tuning on the chain

        :param input_data: data used for hyperparameter searching
        :param loss_function: function to minimize (or maximize)
        :return fitted_chain: chain with optimized hyperparameters
        """
        raise NotImplementedError()

    @staticmethod
    def get_metric_value(train_input, predict_input, test_target,
                         chain, loss_function):
        """
        Method calculates metric for algorithm validation

        :param train_input: data for train chain
        :param predict_input: data for prediction
        :param test_target: target array for validation
        :param chain: chain to process
        :param loss_function: function to minimize (or maximize)

        :return : value of loss function
        """

        chain.fit_from_scratch(train_input)

        # Make prediction
        predicted_values = chain.predict(predict_input)
        preds = np.ravel(np.array(predicted_values.predict))

        return loss_function(test_target, preds)

    def init_check(self, train_input, predict_input,
                   test_target, loss_function) -> None:
        """
        Method get metric on validation set before start optimization

        :param train_input: data for train chain
        :param predict_input: data for prediction
        :param test_target: target array for validation
        :param loss_function: function to minimize (or maximize)
        """

        # Train chain
        self.init_chain = deepcopy(self.chain)

        self.init_metric = self.get_metric_value(train_input=train_input,
                                                 predict_input=predict_input,
                                                 test_target=test_target,
                                                 chain=self.init_chain,
                                                 loss_function=loss_function)

    def final_check(self, train_input, predict_input, test_target,
                    tuned_chain, loss_function):

        obtained_metric = self.get_metric_value(train_input=train_input,
                                                predict_input=predict_input,
                                                test_target=test_target,
                                                chain=tuned_chain,
                                                loss_function=loss_function)

        prefix_tuned_phrase = '\nReturn tuned chain due to the fact that obtained metric'
        prefix_init_phrase = '\nReturn init chain due to the fact that obtained metric'

        # 5% deviation is acceptable
        deviation = (self.init_metric / 100.0) * 5

        if self.is_need_to_maximize is True:
            # Maximization
            init_metric = self.init_metric - deviation
            if obtained_metric >= init_metric:
                print(f'{prefix_tuned_phrase} {obtained_metric:.3f} equal or '
                      f'bigger than initial (- 5% deviation) {init_metric:.3f}')
                return tuned_chain
            else:
                print(f'{prefix_init_phrase} {obtained_metric:.3f} '
                      f'smaller than initial (- 5% deviation) {init_metric:.3f}')
                return self.init_chain
        else:
            # Minimization
            init_metric = self.init_metric + deviation
            if obtained_metric <= init_metric:
                print(f'{prefix_tuned_phrase} {obtained_metric:.3f} equal or '
                      f'smaller than initial (+ 5% deviation) {init_metric:.3f}')
                return tuned_chain
            else:
                print(f'{prefix_init_phrase} {obtained_metric:.3f} '
                      f'bigger than initial (+ 5% deviation) {init_metric:.3f}')
                return self.init_chain

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

        if self.task.task_type == TaskTypesEnum.ts_forecasting:
            # Time series forecasting task - TODO not optimal split -> vital!
            forecast_length = self.task.task_params.forecast_length
            x_train = input_features[:-forecast_length]
            x_test = input_features[:-forecast_length]

            y_train = x_train
            y_test = input_target[-forecast_length:]

            idx_for_train = np.arange(0, len(x_train))

            start_forecast = len(x_train)
            end_forecast = start_forecast + forecast_length
            idx_for_predict = np.arange(start_forecast, end_forecast)
        else:
            x_train, x_test, y_train, y_test = train_test_split(input_features,
                                                                input_target,
                                                                test_size=0.6)
            idx_for_train = np.arange(0, len(x_train))
            idx_for_predict = np.arange(0, len(x_test))

        # Prepare data to train the model
        train_input = InputData(idx=idx_for_train,
                                features=x_train,
                                target=y_train,
                                task=self.task,
                                data_type=DataTypesEnum.table)

        predict_input = InputData(idx=idx_for_predict,
                                  features=x_test,
                                  target=None,
                                  task=self.task,
                                  data_type=DataTypesEnum.table)

        return train_input, predict_input, y_test


def _greater_is_better(target, loss_function) -> bool:
    """ Function checks is metric (loss function) need to be minimized or
    maximized

    :param target: array with target
    :param loss_function: loss function

    :return : bool value is it good to maximize metric or not
    """
    metric = loss_function(target, target)
    if int(round(metric)) == 0:
        return False
    else:
        return True


class ChainTuner(HyperoptTuner):
    """
    Class for hyperparameters optimization for all nodes simultaneously
    """

    def __init__(self, chain, task, iterations=100):
        super().__init__(chain, task, iterations)

    def tune_chain(self, input_data, loss_function):
        """ Function for hyperparameters tuning on the entire chain """

        parameters_dict = self._get_parameters_for_tune(self.chain)

        # Train test split
        train_input, predict_input, test_target = self._validation_split(input_data)

        is_need_to_maximize = _greater_is_better(target=test_target,
                                                 loss_function=loss_function)
        self.is_need_to_maximize = is_need_to_maximize

        # Check source metrics for data
        self.init_check(train_input, predict_input, test_target, loss_function)

        best = fmin(partial(self._objective,
                            chain=self.chain,
                            train_input=train_input,
                            predict_input=predict_input,
                            test_target=test_target,
                            loss_function=loss_function,
                            is_need_to_maximize=is_need_to_maximize),
                    parameters_dict,
                    algo=tpe.suggest,
                    max_evals=self.iterations)

        best = space_eval(space=parameters_dict, hp_assignment=best)

        tuned_chain = self.set_arg_chain(chain=self.chain,
                                         parameters=best)

        # Validation is the optimization do well
        final_chain = self.final_check(train_input=train_input,
                                       predict_input=predict_input,
                                       test_target=test_target,
                                       tuned_chain=tuned_chain,
                                       loss_function=loss_function)

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

    @staticmethod
    def _objective(parameters_dict, chain, train_input, predict_input,
                   test_target, loss_function, is_need_to_maximize):
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

        try:
            min_function = ChainTuner.get_metric_value(train_input=train_input,
                                                       predict_input=predict_input,
                                                       test_target=test_target,
                                                       chain=chain,
                                                       loss_function=loss_function)
        except Exception:
            if is_need_to_maximize is True:
                min_function = -999999.0
            else:
                min_function = 999999.0

        if is_need_to_maximize is True:
            return -min_function
        else:
            return min_function


class SequentialTuner(HyperoptTuner):
    """
    Class for hyperparameters optimization for all nodes sequentially
    """

    def __init__(self, chain, task, iterations=100, inverse_node_order=False):
        super().__init__(chain, task, iterations)
        self.inverse_node_order = inverse_node_order

    def tune_chain(self, input_data, loss_function):
        """ Method for hyperparameters sequential tuning """

        # Train test split
        train_input, predict_input, test_target = self._validation_split(input_data)

        is_need_to_maximize = _greater_is_better(target=test_target,
                                                 loss_function=loss_function)
        self.is_need_to_maximize = is_need_to_maximize

        # Check source metrics for data
        self.init_check(train_input, predict_input, test_target, loss_function)

        # Calculate amount of iterations we can apply per node
        nodes_amount = len(self.chain.nodes)
        iterations_per_node = round(self.iterations/nodes_amount)
        iterations_per_node = int(iterations_per_node)

        # Tuning performed sequentially for every node - so get ids of nodes
        nodes_ids = self.get_nodes_order(nodes_amount=nodes_amount)
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
                self._optimize_node(node_id=node_id,
                                    train_input=train_input,
                                    predict_input=predict_input,
                                    test_target=test_target,
                                    node_params=node_params,
                                    iterations_per_node=iterations_per_node,
                                    loss_function=loss_function)

        # Validation is the optimization do well
        final_chain = self.final_check(train_input=train_input,
                                       predict_input=predict_input,
                                       test_target=test_target,
                                       tuned_chain=self.chain,
                                       loss_function=loss_function)

        return final_chain

    def tune_node(self, input_data, loss_function, node_id):
        """ Method for hyperparameters tuning for particular node"""
        # Train test split
        train_input, predict_input, test_target = self._validation_split(input_data)

        is_need_to_maximize = _greater_is_better(target=test_target,
                                                 loss_function=loss_function)
        self.is_need_to_maximize = is_need_to_maximize

        # Check source metrics for data
        self.init_check(train_input, predict_input, test_target, loss_function)

        node = self.chain.nodes[node_id]
        operation_name = str(node.operation)

        # Get node's parameters to optimize
        node_params = get_node_params(node_id=node_id,
                                      operation_name=operation_name)

        if node_params is None:
            print(f'"{operation_name}" operation has no parameters to optimize')
        else:
            # Apply tuning for current node
            self._optimize_node(node_id=node_id,
                                train_input=train_input,
                                predict_input=predict_input,
                                test_target=test_target,
                                node_params=node_params,
                                iterations_per_node=self.iterations,
                                loss_function=loss_function)

        # Validation is the optimization do well
        final_chain = self.final_check(train_input=train_input,
                                       predict_input=predict_input,
                                       test_target=test_target,
                                       tuned_chain=self.chain,
                                       loss_function=loss_function)

        return final_chain

    def get_nodes_order(self, nodes_amount):
        """ Method returns list with indices of nodes in the chain """

        if self.inverse_node_order is True:
            # From source data to output
            nodes_ids = range(nodes_amount-1, -1, -1)
        else:
            # From output to source data
            nodes_ids = range(0, nodes_amount)

        return nodes_ids

    def _optimize_node(self, node_id, train_input, predict_input, test_target,
                       node_params, iterations_per_node, loss_function):
        """
        Method for node optimization

        :param node_id: id of the current node in the chain
        :param train_input: input for train chain model
        :param predict_input: input for test chain model
        :param test_target: target for validation
        :param node_params: dictionary with parameters for node
        :param iterations_per_node: amount of iterations to produce
        :param loss_function: loss function to minimize

        :return : updated chain with tuned parameters in particular node
        """
        best_parameters = fmin(partial(self._objective,
                                       chain=self.chain,
                                       node_id=node_id,
                                       train_input=train_input,
                                       predict_input=predict_input,
                                       test_target=test_target,
                                       loss_function=loss_function,
                                       is_need_to_maximize=self.is_need_to_maximize),
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

    @staticmethod
    def _objective(node_params, chain, node_id, train_input, predict_input,
                   test_target, loss_function, is_need_to_maximize):
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
        chain = SequentialTuner.set_arg_node(chain=chain,
                                             node_id=node_id,
                                             node_params=node_params)

        try:
            min_function = SequentialTuner.get_metric_value(train_input=train_input,
                                                            predict_input=predict_input,
                                                            test_target=test_target,
                                                            chain=chain,
                                                            loss_function=loss_function)
        except Exception:
            if is_need_to_maximize is True:
                min_function = -999999.0
            else:
                min_function = 999999.0

        if is_need_to_maximize is True:
            return -min_function
        else:
            return min_function
