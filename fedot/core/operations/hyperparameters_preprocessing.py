import numpy as np
from typing import Optional


class HyperparametersPreprocessor():
    """
    Class for hyperparameters preprocessing before operation fitting
    :param operation_type: name of the operation
    :param n_samples_data: number of rows in data
    """

    def __init__(self,
                 operation_type: Optional[str],
                 n_samples_data: Optional[int]):
        self.preprocessing_rules = self._get_preprocessing_rules(operation_type)
        self.n_samples_data = n_samples_data

    def _get_preprocessing_rules(self, operation_type):
        all_preprocessing_rules = {
            'knnreg': {
                'n_neighbors': ['integer']
            },
            'dtreg': {
                'max_leaf_nodes': ['integer'],
                'max_depth': ['integer']
            },
            'treg': {
                'n_estimators': ['integer'],
                'max_leaf_nodes': ['integer'],
                'max_depth': ['integer']
            },
            'rfr': {
                'n_estimators': ['integer'],
                'max_leaf_nodes': ['integer'],
                'max_depth': ['integer']
            },
            'adareg': {
                'n_estimators': ['integer'],
                'max_leaf_nodes': ['integer'],
                'max_depth': ['integer']
            },
            'gbr': {
                'n_estimators': ['integer'],
                'max_leaf_nodes': ['integer'],
                'max_depth': ['integer']
            },
            'xgbreg': {
                'nthread': ['integer'],
                'n_estimators': ['integer'],
                'max_depth': ['integer'],
                'max_leaves': ['integer'],
                'max_bin': ['integer']
            },
            'lgbmreg': {
                'num_iterations': ['integer'], 'num_iteration': ['integer'], 'n_iter': ['integer'],
                'num_tree': ['integer'], 'num_trees': ['integer'],
                'num_round': ['integer'], 'num_rounds': ['integer'], 'nrounds': ['integer'],
                'num_boost_round': ['integer'], 'n_estimators': ['integer'], 'max_iter': ['integer'],
                'num_leaves': ['integer'], 'num_leaf': ['integer'],
                'max_leaves': ['integer'], 'max_leaf': ['integer'], 'max_leaf_nodes': ['integer'],
                'max_depth': ['integer'],
                'min_data_in_leaf': ['absolute', 'integer'], 'min_data_per_leaf': ['absolute', 'integer'],
                'min_data': ['absolute', 'integer'], 'min_child_samples': ['absolute', 'integer'],
                'min_samples_leaf': ['absolute', 'integer'],
                'bagging_freq': ['integer'], 'subsample_freq': ['integer'],
                'max_bin': ['integer'], 'max_bins': ['integer'],
            },
            'catboostreg': {
                'iterations': ['integer'], 'num_boost_round': ['integer'],
                'n_estimators': ['integer'], 'num_trees': ['integer'],
                'depth': ['integer'], 'max_depth': ['integer'],
                'min_data_in_leaf': ['absolute', 'integer'], 'min_child_samples': ['absolute', 'integer'],
                'max_leaves': ['integer'], 'num_leaves': ['integer'],
            },

            'knn': {
                'n_neighbors': ['integer']
            },
            'dt': {
                'max_leaf_nodes': ['integer'],
                'max_depth': ['integer']
            },
            'rf': {
                'n_estimators': ['integer'],
                'max_leaf_nodes': ['integer'],
                'max_depth': ['integer']
            },
            'xgboost': {
                'nthread': ['integer'],
                'n_estimators': ['integer'],
                'max_depth': ['integer'],
                'max_leaves': ['integer'],
                'max_bin': ['integer']
            },
            'lgbm': {
                'num_iterations': ['integer'], 'num_iteration': ['integer'], 'n_iter': ['integer'],
                'num_tree': ['integer'], 'num_trees': ['integer'],
                'num_round': ['integer'], 'num_rounds': ['integer'], 'nrounds': ['integer'],
                'num_boost_round': ['integer'], 'n_estimators': ['integer'], 'max_iter': ['integer'],
                'num_leaves': ['integer'], 'num_leaf': ['integer'],
                'max_leaves': ['integer'], 'max_leaf': ['integer'], 'max_leaf_nodes': ['integer'],
                'max_depth': ['integer'],
                'min_data_in_leaf': ['absolute', 'integer'], 'min_data_per_leaf': ['absolute', 'integer'],
                'min_data': ['absolute', 'integer'], 'min_child_samples': ['absolute', 'integer'],
                'min_samples_leaf': ['absolute', 'integer'],
                'bagging_freq': ['integer'], 'subsample_freq': ['integer'],
                'max_bin': ['integer'], 'max_bins': ['integer'],
            },
            'catboost': {
                'iterations': ['integer'], 'num_boost_round': ['integer'],
                'n_estimators': ['integer'], 'num_trees': ['integer'],
                'depth': ['integer'], 'max_depth': ['integer'],
                'min_data_in_leaf': ['absolute', 'integer'], 'min_child_samples': ['absolute', 'integer'],
                'max_leaves': ['integer'], 'num_leaves': ['integer'],
            },

            'kmeans': {
                'n_clusters': ['integer']
            },

            'kernel_pca': {
                'n_components': ['integer']
            },
            'fast_ica': {
                'n_components': ['integer']
            },
        }

        return all_preprocessing_rules.get(operation_type, {})

    def correct(self,
                params: Optional[dict]):
        if params is None:
            return params

        for param in params:
            if param in self.preprocessing_rules:
                for preprocess in self.preprocessing_rules[param]:
                    params[param] = self._correct(param_value=params[param],
                                                  preprocess_type=preprocess)

        return params

    def _correct(self, param_value, preprocess_type):
        """
        Method calls preprocessing methods basing on preprocess type
        :param param_value : initial value of the parameter
        :param preprocess_type : type of the preprocessing transformation
        :return : param_value after preprocessing
        """

        if preprocess_type == 'integer':
            param_value = self._correct_integer(param_value=param_value)
        elif preprocess_type == 'absolute':
            param_value = self._correct_absolute(param_value=param_value)
        return param_value

    def _correct_absolute(self,
                          param_value):
        """
        Method adds option of using share of total samples in data besides an absolute number of samples
        :param param_value: initial value of the parameter
        :return: param_value after transformation
        """

        if 0 <= param_value < 1:
            return np.ceil(param_value * self.n_samples_data)
        return param_value

    def _correct_integer(self,
                         param_value):
        """
        Method rounds parameter value to avoid errors
        :param param_value: initial value of the parameter
        :return: param_value after rounding
        """

        return round(param_value)
