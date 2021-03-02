import numpy as np
from hyperopt import hp


params_by_operation = {
    'lasso': ['alpha'],
    'ridge': ['alpha'],
    'dtreg': ['max_depth', 'min_samples_split', 'min_samples_leaf'],
    'knnreg': ['n_neighbors', 'weights', 'p'],
    'rfr': ['n_estimators', 'max_features', 'min_samples_split',
            'min_samples_leaf', 'bootstrap'],
    'rfe_lin_reg': ['n_features_to_select', 'step'],
    'ransac_lin_reg': ['min_samples', 'residual_threshold',
                       'max_trials', 'max_skips'],
    'lagged': ['window_size'],
}


def __get_range_by_parameter(label, parameter_name):
    """
    Function prepares appropriate labeled dictionary for desired operation

    :param label: label to assign in hyperopt pyll
    :param parameter_name: name of hyperparameter of particular operation

    :return : dictionary with appropriate range
    """

    range_by_parameter = {
        'lasso | alpha': hp.uniform(label, 0.01, 1.0),
        'ridge | alpha': hp.uniform(label, 0.01, 1.0),

        'ransac_lin_reg | min_samples': hp.uniform(label, 0.1, 0.9),
        'ransac_lin_reg | residual_threshold': hp.choice(label, [0.1, 1.0, 100.0, 500.0, 1000.0]),
        'ransac_lin_reg | max_trials': hp.uniform(label, 50, 500),
        'ransac_lin_reg | max_skips': hp.uniform(label, 50, 500000),

        'rfr | n_estimators': hp.choice(label, [100]),
        'rfr | max_features': hp.uniform(label, 0.05, 1.01),
        'rfr | min_samples_split': hp.choice(label, range(2, 21)),
        'rfr | min_samples_leaf': hp.choice(label, range(1, 21)),
        'rfr | bootstrap': hp.choice(label, [True, False]),

        'dtreg | max_depth': hp.choice(label, range(1, 11)),
        'dtreg | min_samples_split': hp.choice(label, range(2, 21)),
        'dtreg | min_samples_leaf': hp.choice(label, range(1, 21)),

        'knnreg | n_neighbors': hp.choice(label, range(1, 101)),
        'knnreg | weights': hp.choice(label, ["uniform", "distance"]),
        'knnreg | p': hp.choice(label, [True, False]),

        'lagged | window_size': hp.uniform(label, 10, 1000),
    }

    return range_by_parameter.get(parameter_name)


def get_node_params(node_id, operation_name):
    """
    Function for forming dictionary with hyperparameters for considering
    operation

    :param node_id: number of node in chain.nodes list
    :param operation_name: name of operation in the node

    :return params_dict: dictionary-like structure with labeled hyperparameters
    and their range per operation
    """

    # Get available parameters for operation
    params_list = params_by_operation.get(operation_name)

    if params_list is None:
        params_dict = None
    else:
        params_dict = {}
        for parameter_name in params_list:
            # Name with operation and parameter
            parameter_name = ''.join((operation_name, ' | ', parameter_name))

            # Name with node id || operation | parameter
            labeled_parameter_name = ''.join((str(node_id), ' || ', parameter_name))

            # For operation get range where search can be done
            space = __get_range_by_parameter(label=labeled_parameter_name,
                                             parameter_name=parameter_name)

            params_dict.update({labeled_parameter_name: space})

    return params_dict


def convert_params(params):
    """
    Function removes labels from dictionary with operations

    :param params: labeled parameters
    :return new_params: dictionary without labels of node_id and operation_name
    """
    operation_parameters = list(params.keys())

    new_params = {}
    for operation_parameter in operation_parameters:
        value = params.get(operation_parameter)

        # Remove right part of the parameter name
        parameter_name = operation_parameter.split(' | ')[-1]

        if value is None:
            pass
        else:
            new_params.update({parameter_name: value})

    return new_params
