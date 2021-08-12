import random
import numpy as np
from hyperopt import hp
from hyperopt.pyll.stochastic import sample as hp_sample


class ParametersChanger:
    """
    Class for the hyperparameters changing in the operation

    :attribute operation_name: name of operation to get hyperparameters for
    :attribute current_params: current parameters value
    """

    def __init__(self, operation_name, current_params):
        self.operation_name = operation_name
        self.current_params = current_params

    def get_new_operation_params(self):
        """ Function return a dictionary with new parameters values """

        # Get available parameters for operation
        params_list = get_operation_parameter_range(self.operation_name)

        if params_list is None:
            params_dict = None
        else:
            # Get new values for all parameters
            params_dict = self.new_params_dict(params_list)

        return params_dict

    def new_params_dict(self, params_list):
        """ Change values of hyperparameters by different ways

        :param params_list: list with hyperparameters names
        """

        _change_by_name = {'lagged': {'window_size': self._incremental_change},
                           'sparse_lagged': {'window_size': self._incremental_change}}

        params_dict = {}
        for parameter_name in params_list:
            # Get current value of the parameter
            current_value = self._get_current_parameter_value(parameter_name)

            # Perform parameter value change using appropriate function
            operation_dict = _change_by_name.get(self.operation_name)
            if operation_dict is not None:
                func = operation_dict.get(parameter_name)
            else:
                # Default changes perform with random choice
                func = self._random_change

            parameters = {'operation_name': self.operation_name,
                          'current_value': current_value}
            param_value = func(parameter_name, **parameters)
            params_dict.update(param_value)

        return params_dict

    def _get_current_parameter_value(self, parameter_name):
        if isinstance(self.current_params, str):
            # TODO 'default_params' - need to process
            current_value = None
        else:
            # Dictionary with parameters
            current_value = self.current_params[parameter_name]

        return current_value

    @staticmethod
    def _random_change(parameter_name, **kwargs):
        """ Randomly selects a parameter value from a specified range """

        space = get_operation_parameter_range(operation_name=kwargs['operation_name'],
                                              parameter_name=parameter_name,
                                              label=parameter_name)
        # Randomly choose new value
        new_value = hp_sample(space)
        return {parameter_name: new_value}

    @staticmethod
    def _incremental_change(parameter_name, **kwargs):
        """ Next to the current value, the normally distributed new value is set aside """
        # TODO add the ability to limit the boundaries of the params ranges
        sigma = kwargs['current_value'] * 0.3
        new_value = random.normalvariate(kwargs['current_value'], sigma)
        return {parameter_name: new_value}


def get_operation_parameter_range(operation_name: str, parameter_name: str = None,
                                  label: str = 'default'):
    """
    Function prepares appropriate labeled dictionary for desired operation.
    If parameter name is not defined - return all available operation

    :param operation_name: name of the operation
    :param parameter_name: name of hyperparameter of particular operation
    :param label: label to assign in hyperopt pyll

    :return : dictionary with appropriate range
    """

    parameters_per_operation = {
        'kmeans': {
            'n_clusters': hp.choice(label, [2, 3, 4, 5, 6])
        },
        'adareg': {
            'n_estimators': hp.choice(label, [100]),
            'learning_rate': hp.uniform(label, 1e-3, 1.0),
            'loss': hp.choice(label, ["linear", "square", "exponential"])
        },
        'gbr': {
            'n_estimators': hp.choice(label, [100]),
            'loss': hp.choice(label, ["ls", "lad", "huber", "quantile"]),
            'learning_rate': hp.uniform(label, 1e-3, 1.0),
            'max_depth': hp.choice(label, range(1, 11)),
            'min_samples_split': hp.choice(label, range(2, 21)),
            'min_samples_leaf': hp.choice(label, range(1, 21)),
            'subsample': hp.uniform(label, 0.05, 1.0),
            'max_features': hp.uniform(label, 0.05, 1.0),
            'alpha': hp.uniform(label, 0.75, 0.99)
        },
        'logit': {
            'C': hp.uniform(label, 1e-2, 10.0)
        },
        'rf': {
            'n_estimators': hp.choice(label, [100]),
            'criterion': hp.choice(label, ["gini", "entropy"]),
            'max_features': hp.uniform(label, 0.05, 1.01),
            'min_samples_split': hp.choice(label, range(2, 10)),
            'min_samples_leaf': hp.choice(label, range(1, 15)),
            'bootstrap': hp.choice(label, [True, False])
        },
        'lasso': {
            'alpha': hp.uniform(label, 0.01, 10.0)
        },
        'ridge': {
            'alpha': hp.uniform(label, 0.01, 10.0)
        },
        'rfr': {
            'n_estimators': hp.choice(label, [100]),
            'max_features': hp.uniform(label, 0.05, 1.01),
            'min_samples_split': hp.choice(label, range(2, 21)),
            'min_samples_leaf': hp.choice(label, range(1, 21)),
            'bootstrap': hp.choice(label, [True, False])
        },
        'xgbreg': {
            'n_estimators': hp.choice(label, [100]),
            'max_depth': hp.choice(label, range(1, 11)),
            'learning_rate': hp.choice(label, [1e-3, 1e-2, 1e-1, 0.5, 1.]),
            'subsample': hp.choice(label, np.arange(0.05, 1.01, 0.05)),
            'min_child_weight': hp.choice(label, range(1, 21)),
            'objective': hp.choice(label, ['reg:squarederror'])
        },
        'xgboost': {
            'n_estimators': hp.choice(label, [100]),
            'max_depth': hp.choice(label, range(1, 7)),
            'learning_rate': hp.uniform(label, 0.1, 0.9),
            'subsample': hp.uniform(label, 0.05, 0.99),
            'min_child_weight': hp.choice(label, range(1, 21)),
            'nthread': hp.choice(label, [1])
        },
        'svr': {
            'loss': hp.choice(label, ["epsilon_insensitive", "squared_epsilon_insensitive"]),
            'tol': hp.choice(label, [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]),
            'C': hp.uniform(label, 1e-4, 25.0),
            'epsilon': hp.uniform(label, 1e-4, 1.0)
        },
        'dtreg': {
            'max_depth': hp.choice(label, range(1, 11)),
            'min_samples_split': hp.choice(label, range(2, 21)),
            'min_samples_leaf': hp.choice(label, range(1, 21))
        },
        'treg': {
            'n_estimators': hp.choice(label, [100]),
            'max_features': hp.uniform(label, 0.05, 1.0),
            'min_samples_split': hp.choice(label, range(2, 21)),
            'min_samples_leaf': hp.choice(label, range(1, 21)),
            'bootstrap': hp.choice(label, [True, False])
        },
        'dt': {
            'max_depth': hp.choice(label, range(1, 11)),
            'min_samples_split': hp.choice(label, range(2, 21)),
            'min_samples_leaf': hp.choice(label, range(1, 21))
        },
        'knnreg': {
            'n_neighbors': hp.choice(label, range(1, 50)),
            'weights': hp.choice(label, ["uniform", "distance"]),
            'p': hp.choice(label, [1, 2])
        },
        'knn': {
            'n_neighbors': hp.choice(label, range(1, 50)),
            'weights': hp.choice(label, ["uniform", "distance"]),
            'p': hp.choice(label, [1, 2])
        },
        'arima': {
            'p': hp.choice(label, [1, 2, 3, 4, 5, 6]),
            'd': hp.choice(label, [0, 1]),
            'q': hp.choice(label, [1, 2, 3, 4])
        },
        'stl_arima': {
            'p': hp.choice(label, [1, 2, 3, 4, 5, 6]),
            'd': hp.choice(label, [0, 1]),
            'q': hp.choice(label, [1, 2, 3, 4]),
            'period': hp.choice(label, range(1, 365)),
        },
        'ar': {
            'lag_1': hp.uniform(label, 2, 200),
            'lag_2': hp.uniform(label, 2, 800)
        },
        'pca': {
            'n_components': hp.uniform(label, 0.1, 0.99),
            'svd_solver': hp.choice(label, ['full'])
        },
        'kernel_pca': {
            'n_components': hp.choice(label, range(1, 20))
        },
        'ransac_lin_reg': {
            'min_samples': hp.uniform(label, 0.1, 0.9),
            'residual_threshold': hp.choice(label,
                                            [0.1, 1.0, 100.0, 500.0, 1000.0]),
            'max_trials': hp.uniform(label, 50, 500),
            'max_skips': hp.uniform(label, 50, 500000)
        },
        'ransac_non_lin_reg': {
            'min_samples': hp.uniform(label, 0.1, 0.9),
            'residual_threshold': hp.choice(label,
                                            [0.1, 1.0, 100.0, 500.0, 1000.0]),
            'max_trials': hp.uniform(label, 50, 500),
            'max_skips': hp.uniform(label, 50, 500000)
        },
        'rfe_lin_reg': {
            'n_features_to_select': hp.choice(label, [0.5, 0.7, 0.9]),
            'step': hp.choice(label, [0.1, 0.15, 0.2])
        },
        'rfe_non_lin_reg': {
            'n_features_to_select': hp.choice(label, [0.5, 0.7, 0.9]),
            'step': hp.choice(label, [0.1, 0.15, 0.2])
        },
        'poly_features': {
            'degree': hp.choice(label, [2, 3, 4]),
            'interaction_only': hp.choice(label, [True, False])
        },
        'lagged': {
            'window_size': hp.uniform(label, 5, 500)
        },
        'sparse_lagged': {
            'window_size': hp.uniform(label, 5, 500),
            'n_components': hp.uniform(label, 0, 0.5),
            'use_svd': hp.choice(label, [True, False])
        },
        'smoothing': {
            'window_size': hp.uniform(label, 2, 20)
        },
        'gaussian_filter': {
            'sigma': hp.uniform(label, 1, 5)
        },
        'lgbm': {
            'class_weight': hp.choice(label, [None, 'balanced']),
            'num_leaves': hp.choice(label, np.arange(2, 256, 8, dtype=int)),
            'learning_rate': hp.loguniform(label, np.log(0.01), np.log(0.2)),
            'colsample_bytree': hp.uniform(label, 0.4, 1),
            'subsample': hp.uniform(label, 0.4, 1),
            'lambda_l1': hp.uniform(label, 1e-8, 10.0),
            'lambda_l2': hp.uniform(label, 1e-8, 10.0),
        },
        'lgbmreg': {
            'num_leaves': hp.choice(label, np.arange(2, 256, 8, dtype=int)),
            'learning_rate': hp.loguniform(label, np.log(0.01), np.log(0.2)),
            'colsample_bytree': hp.uniform(label, 0.4, 1),
            'subsample': hp.uniform(label, 0.4, 1),
            'lambda_l1': hp.uniform(label, 1e-8, 10.0),
            'lambda_l2': hp.uniform(label, 1e-8, 10.0),
        },
        'catboost': {
            'max_depth': hp.choice(label, range(1, 11)),
            'learning_rate': hp.loguniform(label, np.log(0.01), np.log(0.2)),
            'min_data_in_leaf': hp.qloguniform(label, 0, 6, 1),
            'border_count': hp.randint(label, 2, 255),
            'l2_leaf_reg': hp.uniform(label, 1e-8, 10.0),
            'loss_function': hp.choice(label, ['Logloss', 'CrossEntropy']),
        },
        'catboostreg': {
            'max_depth': hp.choice(label, range(1, 11)),
            'learning_rate': hp.loguniform(label, np.log(0.01), np.log(0.2)),
            'min_data_in_leaf': hp.qloguniform(label, 0, 6, 1),
            'border_count': hp.randint(label, 2, 255),
            'l2_leaf_reg': hp.uniform(label, 1e-8, 10.0),
        }
    }

    # Get available parameters for current operation
    operation_parameters = parameters_per_operation.get(operation_name)

    if operation_parameters is None:
        return None
    else:
        # If there are not parameter_name - return list with all parameters
        if parameter_name is None:
            return list(operation_parameters.keys())
        else:
            return operation_parameters.get(parameter_name)


def get_node_params(node_id, operation_name):
    """
    Function for forming dictionary with hyperparameters for considering
    operation as a part of the whole pipeline

    :param node_id: number of node in pipeline.nodes list
    :param operation_name: name of operation in the node

    :return params_dict: dictionary-like structure with labeled hyperparameters
    and their range per operation
    """

    # Get available parameters for operation
    params_list = get_operation_parameter_range(operation_name)

    if params_list is None:
        params_dict = None
    else:
        params_dict = {}
        for parameter_name in params_list:
            # Name with operation and parameter
            op_parameter_name = ''.join((operation_name, ' | ', parameter_name))

            # Name with node id || operation | parameter
            node_op_parameter_name = ''.join((str(node_id), ' || ', op_parameter_name))

            # For operation get range where search can be done
            space = get_operation_parameter_range(operation_name=operation_name,
                                                  parameter_name=parameter_name,
                                                  label=node_op_parameter_name)

            params_dict.update({node_op_parameter_name: space})

    return params_dict


def convert_params(params):
    """
    Function removes labels from dictionary with operations

    :param params: labeled parameters
    :return new_params: dictionary without labels of node_id and operation_name
    """

    new_params = {}
    for operation_parameter, value in params.items():
        # Remove right part of the parameter name
        parameter_name = operation_parameter.split(' | ')[-1]

        if value is not None:
            new_params.update({parameter_name: value})

    return new_params
