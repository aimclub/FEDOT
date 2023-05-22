from typing import Optional, Dict, Tuple, Callable, List

import numpy as np
from golem.core.tuning.search_space import SearchSpace
from hyperopt import hp


class PipelineSearchSpace(SearchSpace):
    """
    Class for extracting searching space

    :param custom_search_space: dictionary of dictionaries of tuples (hyperopt expression (e.g. hp.choice), *params)
     for applying custom hyperparameters search space
    :param replace_default_search_space: whether replace default dictionary (False) or append it (True)
    """

    def __init__(self,
                 custom_search_space: Optional[Dict[str, Dict[str, Tuple[Callable, List]]]] = None,
                 replace_default_search_space: bool = False):
        self.custom_search_space = custom_search_space
        self.replace_default_search_space = replace_default_search_space
        parameters_per_operation = self.get_parameters_dict()
        super().__init__(parameters_per_operation)

    def get_parameters_dict(self):
        parameters_per_operation = {
            'kmeans': {
                'n_clusters': (hp.uniformint, [2, 7])
            },
            'adareg': {

                'learning_rate': (hp.loguniform, [np.log(1e-3), np.log(1)]),
                'loss': (hp.choice, [["linear", "square", "exponential"]])
            },
            'gbr': {

                'loss': (hp.choice, [["ls", "lad", "huber", "quantile"]]),
                'learning_rate': (hp.loguniform, [np.log(1e-3), np.log(1)]),
                'max_depth': (hp.uniformint, [1, 11]),
                'min_samples_split': (hp.uniformint, [2, 21]),
                'min_samples_leaf': (hp.uniformint, [1, 21]),
                'subsample': (hp.uniform, [0.05, 1.0]),
                'max_features': (hp.uniform, [0.05, 1.0]),
                'alpha': (hp.uniform, [0.75, 0.99])
            },
            'logit': {
                'C': (hp.uniform, [1e-2, 10.0])
            },
            'rf': {
                'criterion': (hp.choice, [["gini", "entropy"]]),
                'max_features': (hp.uniform, [0.05, 1.0]),
                'min_samples_split': (hp.uniformint, [2, 10]),
                'min_samples_leaf': (hp.uniformint, [1, 15]),
                'bootstrap': (hp.choice, [[True, False]])
            },
            'lasso': {
                'alpha': (hp.uniform, [0.01, 10.0])
            },
            'ridge': {
                'alpha': (hp.uniform, [0.01, 10.0])
            },
            'rfr': {

                'max_features': (hp.uniform, [0.05, 1.0]),
                'min_samples_split': (hp.uniformint, [2, 21]),
                'min_samples_leaf': (hp.uniformint, [1, 21]),
                'bootstrap': (hp.choice, [[True, False]])
            },
            'xgbreg': {

                'max_depth': (hp.uniformint, [1, 11]),
                'learning_rate': (hp.loguniform, [np.log(1e-3), np.log(1)]),
                'subsample': (hp.uniform, [0.05, 1.0]),
                'min_child_weight': (hp.uniformint, [1, 21]),
                'objective': (hp.choice, [['reg:squarederror']])
            },
            'xgboost': {

                'max_depth': (hp.uniformint, [1, 7]),
                'learning_rate': (hp.loguniform, [np.log(1e-3), np.log(1)]),
                'subsample': (hp.uniform, [0.05, 0.99]),
                'min_child_weight': (hp.uniform, [1, 21])
            },
            'svr': {
                'loss': (hp.choice, [["epsilon_insensitive", "squared_epsilon_insensitive"]]),
                'tol': (hp.loguniform, [np.log(1e-5), np.log(1e-1)]),
                'C': (hp.uniform, [1e-4, 25.0]),
                'epsilon': (hp.uniform, [1e-4, 1.0])
            },
            'dtreg': {
                'max_depth': (hp.uniformint, [1, 11]),
                'min_samples_split': (hp.uniformint, [2, 21]),
                'min_samples_leaf': (hp.uniformint, [1, 21])
            },
            'treg': {

                'max_features': (hp.uniform, [0.05, 1.0]),
                'min_samples_split': (hp.uniformint, [2, 21]),
                'min_samples_leaf': (hp.uniformint, [1, 21]),
                'bootstrap': (hp.choice, [[True, False]])
            },
            'dt': {
                'max_depth': (hp.uniformint, [1, 11]),
                'min_samples_split': (hp.uniformint, [2, 21]),
                'min_samples_leaf': (hp.uniformint, [1, 21])
            },
            'knnreg': {
                'n_neighbors': (hp.uniformint, [1, 50]),
                'weights': (hp.choice, [["uniform", "distance"]]),
                'p': (hp.choice, [[1, 2]])
            },
            'knn': {
                'n_neighbors': (hp.uniformint, [1, 50]),
                'weights': (hp.choice, [["uniform", "distance"]]),
                'p': (hp.choice, [[1, 2]])
            },
            'arima': {
                'p': (hp.uniformint, [1, 7]),
                'd': (hp.uniformint, [0, 2]),
                'q': (hp.uniformint, [1, 5])
            },
            'stl_arima': {
                'p': (hp.uniformint, [1, 7]),
                'd': (hp.uniformint, [0, 2]),
                'q': (hp.uniformint, [1, 5]),
                'period': (hp.uniformint, [1, 365])
            },
            'ar': {
                'lag_1': (hp.uniform, [2, 200]),
                'lag_2': (hp.uniform, [2, 800])
            },
            'ets': {
                'error': (hp.choice, [['add', 'mul']]),
                'trend': (hp.choice, [[None, 'add', 'mul']]),
                'seasonal': (hp.choice, [[None, 'add', 'mul']]),
                'damped_trend': (hp.choice, [[True, False]]),
                'seasonal_periods': (hp.uniform, [1, 100])
            },
            'glm': {'nested_space': (hp.choice, [[
                {
                    'family': 'gaussian',
                    'link': hp.choice('link_gaussian', ['identity',
                                                        'inverse_power',
                                                        'log'])
                },
                {
                    'family': 'gamma',
                    'link': hp.choice('link_gamma', ['identity',
                                                     'inverse_power',
                                                     'log'])
                },
                {
                    'family': 'inverse_gaussian',
                    'link': hp.choice('link_inv_gaussian', ['identity',
                                                            'inverse_power'])
                }

            ]])},
            'cgru': {
                'hidden_size': (hp.uniform, [20, 200]),
                'learning_rate': (hp.uniform, [0.0005, 0.005]),
                'cnn1_kernel_size': (hp.uniformint, [3, 8]),
                'cnn1_output_size': (hp.choice, [[8, 16, 32, 64]]),
                'cnn2_kernel_size': (hp.uniformint, [3, 8]),
                'cnn2_output_size': (hp.choice, [[8, 16, 32, 64]]),
                'batch_size': (hp.choice, [[64, 128]]),
                'num_epochs': (hp.choice, [[10, 20, 50, 100]]),
                'optimizer': (hp.choice, [['adamw', 'sgd']]),
                'loss': (hp.choice, [['mae', 'mse']])
            },
            'pca': {
                'n_components': (hp.uniform, [0.1, 0.99]),
                'svd_solver': (hp.choice, [['full']])
            },
            'kernel_pca': {
                'n_components': (hp.uniformint, [1, 20]),
                'kernel': (hp.choice, [['linear', 'poly', 'rbf', 'sigmoid', 'cosine', 'precomputed']])
            },
            'fast_ica': {
                'n_components': (hp.uniformint, [1, 20]),
                'fun': (hp.choice, [['logcosh', 'exp', 'cube']])
            },
            'ransac_lin_reg': {
                'min_samples': (hp.uniform, [0.1, 0.9]),
                'residual_threshold': (hp.loguniform, [np.log(0.1), np.log(1000)]),
                'max_trials': (hp.uniform, [50, 500]),
                'max_skips': (hp.uniform, [50, 500000])
            },
            'ransac_non_lin_reg': {
                'min_samples': (hp.uniform, [0.1, 0.9]),
                'residual_threshold': (hp.loguniform, [np.log(0.1), np.log(1000)]),
                'max_trials': (hp.uniform, [50, 500]),
                'max_skips': (hp.uniform, [50, 500000])
            },
            'isolation_forest_reg': {
                'max_samples': (hp.uniform, [0.05, 0.99]),
                'max_features': (hp.uniform, [0.05, 0.99]),
                'bootstrap': (hp.choice, [[True, False]])
            },
            'isolation_forest_class': {
                'max_samples': (hp.uniform, [0.05, 0.99]),
                'max_features': (hp.uniform, [0.05, 0.99]),
                'bootstrap': (hp.choice, [[True, False]])
            },
            'rfe_lin_reg': {
                'n_features_to_select': (hp.uniform, [0.5, 0.9]),
                'step': (hp.uniform, [0.1, 0.2])
            },
            'rfe_non_lin_reg': {
                'n_features_to_select': (hp.uniform, [0.5, 0.9]),
                'step': (hp.uniform, [0.1, 0.2])
            },
            'poly_features': {
                'degree': (hp.uniformint, [2, 5]),
                'interaction_only': (hp.choice, [[True, False]])
            },
            'polyfit': {
                'degree': (hp.uniformint, [1, 6])
            },
            'lagged': {
                'window_size': (hp.uniformint, [5, 500])
            },
            'sparse_lagged': {
                'window_size': (hp.uniformint, [5, 500]),
                'n_components': (hp.uniform, [0, 0.5]),
                'use_svd': (hp.choice, [[True, False]])
            },
            'smoothing': {
                'window_size': (hp.uniformint, [2, 20])
            },
            'gaussian_filter': {
                'sigma': (hp.uniform, [1, 5])
            },
            'diff_filter': {
                'poly_degree': (hp.uniformint, [1, 5]),
                'order': (hp.uniform, [1, 3]),
                'window_size': (hp.uniform, [3, 20])
            },
            'cut': {
                'cut_part': (hp.uniform, [0, 0.9])
            },
            'lgbm': {
                'class_weight': (hp.choice, [[None, 'balanced']]),
                'num_leaves': (hp.uniformint, [2, 256]),
                'learning_rate': (hp.loguniform, [np.log(0.01), np.log(0.2)]),
                'colsample_bytree': (hp.uniform, [0.4, 1]),
                'subsample': (hp.uniform, [0.4, 1]),
                'reg_alpha': (hp.loguniform, [np.log(1e-8), np.log(10)]),
                'reg_lambda': (hp.loguniform, [np.log(1e-8), np.log(10)])
            },
            'lgbmreg': {
                'num_leaves': (hp.uniformint, [2, 256]),
                'learning_rate': (hp.loguniform, [np.log(0.01), np.log(0.2)]),
                'colsample_bytree': (hp.uniform, [0.4, 1]),
                'subsample': (hp.uniform, [0.4, 1]),
                'reg_alpha': (hp.loguniform, [np.log(1e-8), np.log(10)]),
                'reg_lambda': (hp.loguniform, [np.log(1e-8), np.log(10)])
            },
            'catboost': {
                'max_depth': (hp.uniformint, [1, 11]),
                'learning_rate': (hp.loguniform, [np.log(0.01), np.log(0.2)]),
                'min_data_in_leaf': (hp.qloguniform, [0, 6, 1]),
                'border_count': (hp.uniformint, [2, 255]),
                'l2_leaf_reg': (hp.loguniform, [np.log(1e-8), np.log(10)])
            },
            'catboostreg': {
                'max_depth': (hp.uniformint, [1, 11]),
                'learning_rate': (hp.loguniform, [np.log(0.01), np.log(0.2)]),
                'min_data_in_leaf': (hp.qloguniform, [0, 6, 1]),
                'border_count': (hp.uniformint, [2, 255]),
                'l2_leaf_reg': (hp.loguniform, [np.log(1e-8), np.log(10)])
            },
            'resample': {
                'balance': (hp.choice, [['expand_minority', 'reduce_majority']]),
                'replace': (hp.choice, [[True, False]]),
                'balance_ratio': (hp.uniform, [0.3, 1])
            },
            'lda': {
                'solver': (hp.choice, [['svd', 'lsqr', 'eigen']]),
                'shrinkage': (hp.uniform, [0.1, 0.9])
            },
            'ts_naive_average': {'part_for_averaging': (hp.uniform, [0.1, 1])},
            'locf': {'part_for_repeat': (hp.uniform, [0.01, 0.5])},
            'word2vec_pretrained': {
                'model_name': (hp.choice, [['glove-twitter-25', 'glove-twitter-50',
                                            'glove-wiki-gigaword-100', 'word2vec-ruscorpora-300']])
            },
            'tfidf': {
                'ngram_range': (hp.choice, [[(1, 1), (1, 2), (1, 3)]]),
                'min_df': (hp.uniform, [0.0001, 0.1]),
                'max_df': (hp.uniform, [0.9, 0.99])
            },
        }

        if self.custom_search_space is not None:
            if self.replace_default_search_space:
                parameters_per_operation.update(self.custom_search_space)
            else:
                for operation_name, operation_dct in self.custom_search_space.items():
                    parameters_per_operation[operation_name].update(operation_dct)

        return parameters_per_operation
