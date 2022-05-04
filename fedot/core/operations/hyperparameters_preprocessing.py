import numpy as np
from typing import Optional
from fedot.core.data.data import InputData


class HyperparametersPreprocessor():
    def __init__(self,
                 operation_type: Optional[str],
                 train_data: Optional[InputData]):
        self.operation_type = operation_type
        print(operation_type)
        self.train_data = train_data

    def correct(self,
                params: Optional[dict]):
        integer_params = {
            'n_clusters', 'n_neighbors',
            'n_estimators', 'num_iterations', 'num_iteration', 'n_iter', 'max_iter',
            'num_tree', 'num_trees', 'num_round', 'num_rounds', 'nrounds', 'num_boost_round',
            'num_leaves', 'num_leaf', 'max_leaves', 'max_leaf', 'max_leaf_nodes',
            'max_depth',
            'max_bin', 'max_bins',
            'min_data_in_leaf', 'min_data_per_leaf', 'min_data', 'min_child_samples', 'min_samples_leaf',
        }
        data_relative_to_absolute_params = {
            'min_data_in_leaf', 'min_data_per_leaf', 'min_data', 'min_child_samples', 'min_samples_leaf'
        }

        for param in params:
            if param in data_relative_to_absolute_params and 0 <= params[param] < 1:
                # Adding option of using share of total samples in data besides an absolute number of samples
                params[param] = np.ceil(params[param] * self.train_data.target.shape[0])
            if param in integer_params:
                # Round parameter values to avoid errors
                params[param] = int(params[param])

        return params


def preprocess_params(params: dict, train_data: InputData) -> dict:
    """
    The function handles incorrect parameters values and returns corresponding correct parameters

    :param params: dictionary of model parameters
    :param train_data: data used for model training

    :return : processed parameters dictionary
    """
    integer_params = {
        'n_clusters', 'n_neighbors',
        'n_estimators', 'num_iterations', 'num_iteration', 'n_iter', 'max_iter',
        'num_tree', 'num_trees', 'num_round', 'num_rounds', 'nrounds', 'num_boost_round',
        'num_leaves', 'num_leaf', 'max_leaves', 'max_leaf', 'max_leaf_nodes',
        'max_depth',
        'max_bin', 'max_bins',
        'min_data_in_leaf', 'min_data_per_leaf', 'min_data', 'min_child_samples', 'min_samples_leaf',
    }
    data_relative_to_absolute_params = {
        'min_data_in_leaf', 'min_data_per_leaf', 'min_data', 'min_child_samples', 'min_samples_leaf'
    }

    for param in params:
        if param in data_relative_to_absolute_params and 0 <= params[param] < 1:
            # Adding option of using share of total samples in data besides an absolute number of samples
            params[param] = np.ceil(params[param] * train_data.target.shape[0])
        if param in integer_params:
            # Round parameter values to avoid errors
            params[param] = int(params[param])

    return params
