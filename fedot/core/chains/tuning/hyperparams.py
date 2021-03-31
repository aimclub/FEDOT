import numpy as np
from hyperopt import hp, fmin, tpe, space_eval


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
            'window_size': hp.uniform(label, 10, 500)
        },
        'smoothing': {
            'window_size': hp.uniform(label, 2, 20)
        },
        'gaussian_filter': {
            'sigma': hp.uniform(label, 1, 5)
        },
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
    operation as a part of the whole chain

    :param node_id: number of node in chain.nodes list
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


def get_new_operation_params(operation_name):
    """ Function return a dictionary with new

    :param operation_name: name of operation to get hyperparameters for
    """

    # Function to imitate objective
    def fake_objective(fake_params):
        return 0

    # Get available parameters for operation
    params_list = get_operation_parameter_range(operation_name)

    if params_list is None:
        params_dict = None
    else:
        params_dict = {}
        for parameter_name in params_list:
            # Get
            space = get_operation_parameter_range(operation_name=operation_name,
                                                  parameter_name=parameter_name,
                                                  label=parameter_name)
            # Get parameters values for chosen parameter
            small_dict = {parameter_name: space}
            best = fmin(fake_objective,
                        small_dict,
                        algo=tpe.suggest,
                        max_evals=1,
                        show_progressbar=False)
            best = space_eval(space=small_dict, hp_assignment=best)
            params_dict.update(best)

    return params_dict
