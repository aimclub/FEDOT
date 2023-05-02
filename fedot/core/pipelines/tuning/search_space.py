from functools import partial
from typing import Optional

from golem.core.tuning.search_space import SearchSpace, OperationParametersMapping
from hyperopt import hp

from fedot.core.utils import NESTED_PARAMS_LABEL


class PipelineSearchSpace(SearchSpace):
    """
    Class for extracting searching space

    :param custom_search_space: dictionary of dictionaries of tuples (hyperopt expression (e.g. hp.choice), *params)
     for applying custom hyperparameters search space
    :param replace_default_search_space: whether replace default dictionary (False) or append it (True)
    """

    def __init__(self,
                 custom_search_space: Optional[OperationParametersMapping] = None,
                 replace_default_search_space: bool = False):
        self.custom_search_space = custom_search_space
        self.replace_default_search_space = replace_default_search_space
        parameters_per_operation = self.get_parameters_dict()
        super().__init__(parameters_per_operation)

    def get_parameters_dict(self):
        parameters_per_operation = {
            'kmeans': {
                'n_clusters': {
                    'hyperopt-dist': hp.uniformint,
                    'sampling-scope': [2, 7],
                    'type': 'discrete'}
            },
            'adareg': {
                'learning_rate': {
                    'hyperopt-dist': hp.loguniform,
                    'sampling-scope': [1e-3, 1],
                    'type': 'continuous'},
                'loss': {
                    'hyperopt-dist': hp.choice,
                    'sampling-scope': [["linear", "square", "exponential"]],
                    'type': 'categorical'}
            },
            'gbr': {
                'loss': {
                    'hyperopt-dist': hp.choice,
                    'sampling-scope': [["ls", "lad", "huber", "quantile"]],
                    'type': 'categorical'},
                'learning_rate': {
                    'hyperopt-dist': hp.loguniform,
                    'sampling-scope': [1e-3, 1],
                    'type': 'continuous'},
                'max_depth': {
                    'hyperopt-dist': hp.uniformint,
                    'sampling-scope': [1, 11],
                    'type': 'discrete'},
                'min_samples_split': {
                    'hyperopt-dist': hp.uniformint,
                    'sampling-scope': [2, 21],
                    'type': 'discrete'},
                'min_samples_leaf': {
                    'hyperopt-dist': hp.uniformint,
                    'sampling-scope': [1, 21],
                    'type': 'discrete'},
                'subsample': {
                    'hyperopt-dist': hp.uniform,
                    'sampling-scope': [0.05, 1.0],
                    'type': 'continuous'},
                'max_features': {
                    'hyperopt-dist': hp.uniform,
                    'sampling-scope': [0.05, 1.0],
                    'type': 'continuous'},
                'alpha': {
                    'hyperopt-dist': hp.uniform,
                    'sampling-scope': [0.75, 0.99],
                    'type': 'continuous'}
            },
            'logit': {
                'C': {
                    'hyperopt-dist': hp.uniform,
                    'sampling-scope': [1e-2, 10.0],
                    'type': 'continuous'}
            },
            'rf': {
                'criterion': {
                    'hyperopt-dist': hp.choice,
                    'sampling-scope': [["gini", "entropy"]],
                    'type': 'categorical'},
                'max_features': {
                    'hyperopt-dist': hp.uniform,
                    'sampling-scope': [0.05, 1.0],
                    'type': 'continuous'},
                'min_samples_split': {
                    'hyperopt-dist': hp.uniformint,
                    'sampling-scope': [2, 10],
                    'type': 'discrete'},
                'min_samples_leaf': {
                    'hyperopt-dist': hp.uniformint,
                    'sampling-scope': [1, 15],
                    'type': 'discrete'},
                'bootstrap': {
                    'hyperopt-dist': hp.choice,
                    'sampling-scope': [[True, False]],
                    'type': 'categorical'}
            },
            'ridge': {
                'alpha': {
                    'hyperopt-dist': hp.uniform,
                    'sampling-scope': [0.01, 10.0],
                    'type': 'continuous'}
            },
            'lasso': {
                'alpha': {
                    'hyperopt-dist': hp.uniform,
                    'sampling-scope': [0.01, 10.0],
                    'type': 'continuous'}
            },
            'rfr': {
                'max_features': {
                    'hyperopt-dist': hp.uniform,
                    'sampling-scope': [0.05, 1.0],
                    'type': 'continuous'},
                'min_samples_split': {
                    'hyperopt-dist': hp.uniformint,
                    'sampling-scope': [2, 21],
                    'type': 'discrete'},
                'min_samples_leaf': {
                    'hyperopt-dist': hp.uniformint,
                    'sampling-scope': [1, 15],
                    'type': 'discrete'},
                'bootstrap': {
                    'hyperopt-dist': hp.choice,
                    'sampling-scope': [[True, False]],
                    'type': 'categorical'}
            },
            'xgbreg': {
                'max_depth': {
                    'hyperopt-dist': hp.uniformint,
                    'sampling-scope': [1, 11],
                    'type': 'discrete'},
                'learning_rate': {
                    'hyperopt-dist': hp.loguniform,
                    'sampling-scope': [1e-3, 1],
                    'type': 'continuous'},
                'subsample': {
                    'hyperopt-dist': hp.uniform,
                    'sampling-scope': [0.05, 1.0],
                    'type': 'continuous'},
                'min_child_weight': {
                    'hyperopt-dist': hp.uniformint,
                    'sampling-scope': [1, 21],
                    'type': 'discrete'},
            },
            'xgboost': {
                'max_depth': {
                    'hyperopt-dist': hp.uniformint,
                    'sampling-scope': [1, 7],
                    'type': 'discrete'},
                'learning_rate': {
                    'hyperopt-dist': hp.loguniform,
                    'sampling-scope': [1e-3, 1],
                    'type': 'continuous'},
                'subsample': {
                    'hyperopt-dist': hp.uniform,
                    'sampling-scope': [0.05, 0.99],
                    'type': 'continuous'},
                'min_child_weight': {
                    'hyperopt-dist': hp.uniformint,
                    'sampling-scope': [1, 21],
                    'type': 'discrete'}
            },
            'svr': {
                'C': {
                    'hyperopt-dist': hp.uniform,
                    'sampling-scope': [1e-4, 25.0],
                    'type': 'continuous'},
                'epsilon': {
                    'hyperopt-dist': hp.uniform,
                    'sampling-scope': [1e-4, 1],
                    'type': 'continuous'},
                'tol': {
                    'hyperopt-dist': hp.loguniform,
                    'sampling-scope': [1e-5, 1e-1],
                    'type': 'continuous'},
                'loss': {
                    'hyperopt-dist': hp.choice,
                    'sampling-scope': [["epsilon_insensitive", "squared_epsilon_insensitive"]],
                    'type': 'categorical'}
            },
            'dtreg': {
                'max_depth': {
                    'hyperopt-dist': hp.uniformint,
                    'sampling-scope': [1, 11],
                    'type': 'discrete'},
                'min_samples_split': {
                    'hyperopt-dist': hp.uniformint,
                    'sampling-scope': [2, 21],
                    'type': 'discrete'},
                'min_samples_leaf': {
                    'hyperopt-dist': hp.uniformint,
                    'sampling-scope': [1, 21],
                    'type': 'discrete'}
            },
            'treg': {
                'max_features': {
                    'hyperopt-dist': hp.uniform,
                    'sampling-scope': [0.05, 1.0],
                    'type': 'continuous'},
                'min_samples_split': {
                    'hyperopt-dist': hp.uniformint,
                    'sampling-scope': [2, 21],
                    'type': 'discrete'},
                'min_samples_leaf': {
                    'hyperopt-dist': hp.uniformint,
                    'sampling-scope': [1, 21],
                    'type': 'discrete'},
                'bootstrap': {
                    'hyperopt-dist': hp.choice,
                    'sampling-scope': [[True, False]],
                    'type': 'categorical'}
            },
            'dt': {
                'max_depth': {
                    'hyperopt-dist': hp.uniformint,
                    'sampling-scope': [1, 11],
                    'type': 'discrete'},
                'min_samples_split': {
                    'hyperopt-dist': hp.uniformint,
                    'sampling-scope': [2, 21],
                    'type': 'discrete'},
                'min_samples_leaf': {
                    'hyperopt-dist': hp.uniformint,
                    'sampling-scope': [1, 21],
                    'type': 'discrete'}
            },
            'knnreg': {
                'n_neighbors': {
                    'hyperopt-dist': hp.uniformint,
                    'sampling-scope': [1, 50],
                    'type': 'discrete'},
                'weights': {
                    'hyperopt-dist': hp.choice,
                    'sampling-scope': [["uniform", "distance"]],
                    'type': 'categorical'},
                'p': {
                    'hyperopt-dist': hp.choice,
                    'sampling-scope': [[1, 2]],
                    'type': 'categorical'}
            },
            'knn': {
                'n_neighbors': {
                    'hyperopt-dist': hp.uniformint,
                    'sampling-scope': [1, 50],
                    'type': 'discrete'},
                'weights': {
                    'hyperopt-dist': hp.choice,
                    'sampling-scope': [["uniform", "distance"]],
                    'type': 'categorical'},
                'p': {
                    'hyperopt-dist': hp.choice,
                    'sampling-scope': [[1, 2]],
                    'type': 'categorical'}
            },
            'arima': {
                'p': {
                    'hyperopt-dist': hp.uniformint,
                    'sampling-scope': [1, 7],
                    'type': 'discrete'},
                'd': {
                    'hyperopt-dist': hp.uniformint,
                    'sampling-scope': [0, 2],
                    'type': 'discrete'},
                'q': {
                    'hyperopt-dist': hp.uniformint,
                    'sampling-scope': [1, 5],
                    'type': 'discrete'}
            },
            'stl_arima': {
                'p': {
                    'hyperopt-dist': hp.uniformint,
                    'sampling-scope': [1, 7],
                    'type': 'discrete'},
                'd': {
                    'hyperopt-dist': hp.uniformint,
                    'sampling-scope': [0, 2],
                    'type': 'discrete'},
                'q': {
                    'hyperopt-dist': hp.uniformint,
                    'sampling-scope': [1, 5],
                    'type': 'discrete'},
                'period': {
                    'hyperopt-dist': hp.uniformint,
                    'sampling-scope': [1, 365],
                    'type': 'discrete'}
            },
            'ar': {
                'lag_1': {
                    'hyperopt-dist': hp.uniform,
                    'sampling-scope': [2, 200],
                    'type': 'continuous'},
                'lag_2': {
                    'hyperopt-dist': hp.uniform,
                    'sampling-scope': [2, 800],
                    'type': 'continuous'}
            },
            'ets': {
                'error': {
                    'hyperopt-dist': hp.choice,
                    'sampling-scope': [["add", "mul"]],
                    'type': 'categorical'},
                'trend': {
                    'hyperopt-dist': hp.choice,
                    'sampling-scope': [[None, "add", "mul"]],
                    'type': 'categorical'},
                'seasonal': {
                    'hyperopt-dist': hp.choice,
                    'sampling-scope': [[None, "add", "mul"]],
                    'type': 'categorical'},
                'damped_trend': {
                    'hyperopt-dist': hp.choice,
                    'sampling-scope': [[True, False]],
                    'type': 'categorical'},
                'seasonal_periods': {
                    'hyperopt-dist': hp.uniform,
                    'sampling-scope': [1, 100],
                    'type': 'continuous'}
            },
            'glm': {
                NESTED_PARAMS_LABEL: {
                    'hyperopt-dist': hp.choice,
                    'sampling-scope': [[
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

                    ]],
                    'type': 'categorical'}
            },
            'cgru': {
                'hidden_size': {
                    'hyperopt-dist': hp.uniform,
                    'sampling-scope': [20, 200],
                    'type': 'continuous'},
                'learning_rate': {
                    'hyperopt-dist': hp.uniform,
                    'sampling-scope': [0.0005, 0.005],
                    'type': 'continuous'},
                'cnn1_kernel_size': {
                    'hyperopt-dist': hp.uniformint,
                    'sampling-scope': [3, 8],
                    'type': 'discrete'},
                'cnn1_output_size': {
                    'hyperopt-dist': hp.choice,
                    'sampling-scope': [[8, 16, 32, 64]],
                    'type': 'categorical'},
                'cnn2_kernel_size': {
                    'hyperopt-dist': hp.uniformint,
                    'sampling-scope': [3, 8],
                    'type': 'discrete'},
                'cnn2_output_size': {
                    'hyperopt-dist': hp.choice,
                    'sampling-scope': [[8, 16, 32, 64]],
                    'type': 'categorical'},
                'batch_size': {
                    'hyperopt-dist': hp.choice,
                    'sampling-scope': [[64, 128]],
                    'type': 'categorical'},
                'num_epochs': {
                    'hyperopt-dist': hp.choice,
                    'sampling-scope': [[10, 20, 50, 100]],
                    'type': 'categorical'},
                'optimizer': {
                    'hyperopt-dist': hp.choice,
                    'sampling-scope': [['adamw', 'sgd']],
                    'type': 'categorical'},
                'loss': {
                    'hyperopt-dist': hp.choice,
                    'sampling-scope': [['mae', 'mse']],
                    'type': 'categorical'},
            },
            'pca': {
                'n_components': {
                    'hyperopt-dist': hp.uniform,
                    'sampling-scope': [0.1, 0.99],
                    'type': 'continuous'}
            },
            'kernel_pca': {
                'n_components': {
                    'hyperopt-dist': hp.uniformint,
                    'sampling-scope': [1, 20],
                    'type': 'discrete'},
                'kernel': {
                    'hyperopt-dist': hp.choice,
                    'sampling-scope': [['linear', 'poly', 'rbf', 'sigmoid', 'cosine', 'precomputed']],
                    'type': 'categorical'}
            },
            'fast_ica': {
                'n_components': {
                    'hyperopt-dist': hp.uniformint,
                    'sampling-scope': [1, 20],
                    'type': 'discrete'},
                'fun': {
                    'hyperopt-dist': hp.choice,
                    'sampling-scope': [['logcosh', 'exp', 'cube']],
                    'type': 'categorical'}
            },
            'ransac_lin_reg': {
                'min_samples': {
                    'hyperopt-dist': hp.uniform,
                    'sampling-scope': [0.1, 0.9],
                    'type': 'continuous'},
                'residual_threshold': {
                    'hyperopt-dist': hp.loguniform,
                    'sampling-scope': [0.1, 1000],
                    'type': 'continuous'},
                'max_trials': {
                    'hyperopt-dist': hp.uniform,
                    'sampling-scope': [50, 500],
                    'type': 'continuous'},
                'max_skips': {
                    'hyperopt-dist': hp.uniform,
                    'sampling-scope': [50, 500000],
                    'type': 'continuous'}
            },
            'ransac_non_lin_reg': {
                'min_samples': {
                    'hyperopt-dist': hp.uniform,
                    'sampling-scope': [0.1, 0.9],
                    'type': 'continuous'},
                'residual_threshold': {
                    'hyperopt-dist': hp.loguniform,
                    'sampling-scope': [0.1, 1000],
                    'type': 'continuous'},
                'max_trials': {
                    'hyperopt-dist': hp.uniform,
                    'sampling-scope': [50, 500],
                    'type': 'continuous'},
                'max_skips': {
                    'hyperopt-dist': hp.uniform,
                    'sampling-scope': [50, 500000],
                    'type': 'continuous'}
            },
            'isolation_forest_reg': {
                'max_samples': {
                    'hyperopt-dist': hp.uniform,
                    'sampling-scope': [0.05, 0.99],
                    'type': 'continuous'},
                'max_features': {
                    'hyperopt-dist': hp.uniform,
                    'sampling-scope': [0.05, 0.99],
                    'type': 'continuous'},
                'bootstrap': {
                    'hyperopt-dist': hp.choice,
                    'sampling-scope': [[True, False]],
                    'type': 'categorical'}
            },
            'isolation_forest_class': {
                'max_samples': {
                    'hyperopt-dist': hp.uniform,
                    'sampling-scope': [0.05, 0.99],
                    'type': 'continuous'},
                'max_features': {
                    'hyperopt-dist': hp.uniform,
                    'sampling-scope': [0.05, 0.99],
                    'type': 'continuous'},
                'bootstrap': {
                    'hyperopt-dist': hp.choice,
                    'sampling-scope': [[True, False]],
                    'type': 'categorical'}
            },
            'rfe_lin_reg': {
                'n_features_to_select': {
                    'hyperopt-dist': hp.uniform,
                    'sampling-scope': [0.5, 0.9],
                    'type': 'continuous'},
                'step': {
                    'hyperopt-dist': hp.uniform,
                    'sampling-scope': [0.1, 0.2],
                    'type': 'continuous'}
            },
            'rfe_non_lin_reg': {
                'n_features_to_select': {
                    'hyperopt-dist': hp.uniform,
                    'sampling-scope': [0.5, 0.9],
                    'type': 'continuous'},
                'step': {
                    'hyperopt-dist': hp.uniform,
                    'sampling-scope': [0.1, 0.2],
                    'type': 'continuous'}
            },
            'poly_features': {
                'degree': {
                    'hyperopt-dist': hp.uniformint,
                    'sampling-scope': [2, 5],
                    'type': 'discrete'},
                'interaction_only': {
                    'hyperopt-dist': hp.choice,
                    'sampling-scope': [[True, False]],
                    'type': 'categorical'}
            },
            'polyfit': {
                'degree': {
                    'hyperopt-dist': hp.uniformint,
                    'sampling-scope': [1, 6],
                    'type': 'discrete'}
            },
            'lagged': {
                'window_size': {
                    'hyperopt-dist': hp.uniformint,
                    'sampling-scope': [5, 500],
                    'type': 'discrete'}
            },
            'sparse_lagged': {
                'window_size': {
                    'hyperopt-dist': hp.uniformint,
                    'sampling-scope': [5, 500],
                    'type': 'discrete'},
                'n_components': {
                    'hyperopt-dist': hp.uniform,
                    'sampling-scope': [0, 0.5],
                    'type': 'continuous'},
                'use_svd': {
                    'hyperopt-dist': hp.choice,
                    'sampling-scope': [[True, False]],
                    'type': 'categorical'}
            },
            'smoothing': {
                'window_size': {
                    'hyperopt-dist': hp.uniformint,
                    'sampling-scope': [2, 20],
                    'type': 'discrete'}
            },
            'gaussian_filter': {
                'sigma': {
                    'hyperopt-dist': hp.uniform,
                    'sampling-scope': [1, 5],
                    'type': 'continuous'}
            },
            'diff_filter': {
                'poly_degree': {
                    'hyperopt-dist': hp.uniformint,
                    'sampling-scope': [1, 5],
                    'type': 'discrete'},
                'order': {
                    'hyperopt-dist': hp.uniform,
                    'sampling-scope': [1, 3],
                    'type': 'continuous'},
                'window_size': {
                    'hyperopt-dist': hp.uniform,
                    'sampling-scope': [3, 20],
                    'type': 'continuous'}
            },
            'cut': {
                'cut_part': {
                    'hyperopt-dist': hp.uniform,
                    'sampling-scope': [0, 0.9],
                    'type': 'continuous'}
            },
            'lgbm': {
                'class_weight': {
                    'hyperopt-dist': hp.choice,
                    'sampling-scope': [[None, 'balanced']],
                    'type': 'categorical'},
                'num_leaves': {
                    'hyperopt-dist': hp.uniformint,
                    'sampling-scope': [2, 256],
                    'type': 'discrete'},
                'learning_rate': {
                    'hyperopt-dist': hp.loguniform,
                    'sampling-scope': [0.01, 0.2],
                    'type': 'continuous'},
                'colsample_bytree': {
                    'hyperopt-dist': hp.uniform,
                    'sampling-scope': [0.4, 1],
                    'type': 'continuous'},
                'subsample': {
                    'hyperopt-dist': hp.uniform,
                    'sampling-scope': [0.4, 1],
                    'type': 'continuous'},
                'reg_alpha': {
                    'hyperopt-dist': hp.loguniform,
                    'sampling-scope': [1e-8, 10],
                    'type': 'continuous'},
                'reg_lambda': {
                    'hyperopt-dist': hp.loguniform,
                    'sampling-scope': [1e-8, 10],
                    'type': 'continuous'}
            },
            'lgbmxt': {
                'class_weight': (hp.choice, [[None, 'balanced']]),
                'num_leaves': (hp.uniformint, [2, 256]),
                'learning_rate': (hp.loguniform, [np.log(0.01), np.log(0.2)]),
                'colsample_bytree': (hp.uniform, [0.4, 1]),
                'subsample': (hp.uniform, [0.4, 1]),
                'reg_alpha': (hp.loguniform, [np.log(1e-8), np.log(10)]),
                'reg_lambda': (hp.loguniform, [np.log(1e-8), np.log(10)])
            },
            'lgbmreg': {
                'num_leaves': {
                    'hyperopt-dist': hp.uniformint,
                    'sampling-scope': [2, 256],
                    'type': 'discrete'},
                'learning_rate': {
                    'hyperopt-dist': hp.loguniform,
                    'sampling-scope': [0.01, 0.2],
                    'type': 'continuous'},
                'colsample_bytree': {
                    'hyperopt-dist': hp.uniform,
                    'sampling-scope': [0.4, 1],
                    'type': 'continuous'},
                'subsample': {
                    'hyperopt-dist': hp.uniform,
                    'sampling-scope': [0.4, 1],
                    'type': 'continuous'},
                'reg_alpha': {
                    'hyperopt-dist': hp.loguniform,
                    'sampling-scope': [1e-8, 10],
                    'type': 'continuous'},
                'reg_lambda': {
                    'hyperopt-dist': hp.loguniform,
                    'sampling-scope': [1e-8, 10],
                    'type': 'continuous'}
            },
            'lgbmxtreg': {
                'num_leaves': (hp.uniformint, [2, 256]),
                'learning_rate': (hp.loguniform, [np.log(0.01), np.log(0.2)]),
                'colsample_bytree': (hp.uniform, [0.4, 1]),
                'subsample': (hp.uniform, [0.4, 1]),
                'reg_alpha': (hp.loguniform, [np.log(1e-8), np.log(10)]),
                'reg_lambda': (hp.loguniform, [np.log(1e-8), np.log(10)])
            },
            'catboost': {
                'iterations': {
                    'hyperopt-dist': hp.randint,
                    'sampling-scope': [500, 10000],
                    'type': 'discrete'
                },
                'learning_rate': {
                    'hyperopt-dist': hp.loguniform,
                    'sampling-scope': [0.01, 1.0],
                    'type': 'continuous'
                },
                'max_depth': {
                    'hyperopt-dist': hp.uniformint,
                    'sampling-scope': [4, 10],
                    'type': 'discrete'
                },
                'max_leaves': {
                    'hyperopt-dist': hp.uniformint,
                    'sampling-scope': [1, 100],
                    'type': 'discrete'
                },
                'min_data_in_leaf': {
                    'hyperopt-dist': partial(hp.qloguniform, q=1),
                    'sampling-scope': [0, 25],
                    'type': 'discrete'
                },
                'border_count': {
                    'hyperopt-dist': hp.uniformint,
                    'sampling-scope': [1, 65535],
                    'type': 'discrete'
                },
                'l2_leaf_reg': {
                    'hyperopt-dist': hp.uniform,
                    'sampling-scope': [1e-8, 10],
                    'type': 'continuous'
                },
                'colsample_bylevel': {
                    'hyperopt-dist': hp.uniform,
                    'sampling-scope': [0.01, 1.0],
                    'type': 'continuous'
                }
            },
            'catboostreg': {
                'iterations': {
                    'hyperopt-dist': hp.randint,
                    'sampling-scope': [500, 10000],
                    'type': 'discrete'
                },
                'learning_rate': {
                    'hyperopt-dist': hp.uniform,
                    'sampling-scope': [0.2, 1.0],
                    'type': 'continuous'
                },
                'max_depth': {
                    'hyperopt-dist': hp.uniformint,
                    'sampling-scope': [4, 10],
                    'type': 'discrete'
                },
                'max_leaves': {
                    'hyperopt-dist': hp.uniformint,
                    'sampling-scope': [1, 100],
                    'type': 'discrete'
                },
                'min_data_in_leaf': {
                    'hyperopt-dist': partial(hp.qloguniform, q=1),
                    'sampling-scope': [0, 25],
                    'type': 'discrete'
                },
                'border_count': {
                    'hyperopt-dist': hp.uniformint,
                    'sampling-scope': [1, 65535],
                    'type': 'discrete'
                },
                'l2_leaf_reg': {
                    'hyperopt-dist': hp.uniform,
                    'sampling-scope': [1e-8, 10],
                    'type': 'continuous'
                },
                'colsample_bylevel': {
                    'hyperopt-dist': hp.uniform,
                    'sampling-scope': [0.01, 1.0],
                    'type': 'continuous'
                }
            },
            'resample': {
                'balance': {
                    'hyperopt-dist': hp.choice,
                    'sampling-scope': [['expand_minority', 'reduce_majority']],
                    'type': 'categorical'},
                'replace': {
                    'hyperopt-dist': hp.choice,
                    'sampling-scope': [[True, False]],
                    'type': 'categorical'},
                'balance_ratio': {
                    'hyperopt-dist': hp.uniform,
                    'sampling-scope': [0.3, 1],
                    'type': 'continuous'}
            },
            'lda': {
                'solver': {
                    'hyperopt-dist': hp.choice,
                    'sampling-scope': [['svd', 'lsqr', 'eigen']],
                    'type': 'categorical'},
                'shrinkage': {
                    'hyperopt-dist': hp.uniform,
                    'sampling-scope': [0.1, 0.9],
                    'type': 'continuous'}
            },
            'ts_naive_average': {
                'part_for_averaging': {
                    'hyperopt-dist': hp.uniform,
                    'sampling-scope': [0.1, 1],
                    'type': 'continuous'}
            },
            'locf': {
                'part_for_repeat': {
                    'hyperopt-dist': hp.uniform,
                    'sampling-scope': [0.01, 0.5],
                    'type': 'continuous'}
            },
            'word2vec_pretrained': {
                'model_name': {
                    'hyperopt-dist': hp.choice,
                    'sampling-scope': [['glove-twitter-25', 'glove-twitter-50',
                                        'glove-wiki-gigaword-100', 'word2vec-ruscorpora-300']],
                    'type': 'categorical'}
            },
            'tfidf': {
                'ngram_range': {
                    'hyperopt-dist': hp.choice,
                    'sampling-scope': [[(1, 1), (1, 2), (1, 3)]],
                    'type': 'categorical'},
                'min_df': {
                    'hyperopt-dist': hp.uniform,
                    'sampling-scope': [0.0001, 0.1],
                    'type': 'continuous'},
                'max_df': {
                    'hyperopt-dist': hp.uniform,
                    'sampling-scope': [0.9, 0.99],
                    'type': 'continuous'}
            },
            'bag_dt': {
                'n_estimators': (hp.uniformint, [3, 50]),
                'bootstrap': (hp.choice, [[True, False]]),
                'oob_score': (hp.choice, [[True, False]]),
                'model_params': {
                    'max_depth': (hp.uniformint, [1, 11]),
                    'min_samples_split': (hp.uniformint, [2, 21]),
                    'min_samples_leaf': (hp.uniformint, [1, 21])
                }
            },
            'bag_dtreg': {
                'n_estimators': (hp.uniformint, [3, 50]),
                'bootstrap': (hp.choice, [[True, False]]),
                'oob_score': (hp.choice, [[True, False]]),
                'model_params': {
                    'max_depth': (hp.uniformint, [1, 11]),
                    'min_samples_split': (hp.uniformint, [2, 21]),
                    'min_samples_leaf': (hp.uniformint, [1, 21])
                }
            },
            'bag_adareg': {
                'n_estimators': (hp.uniformint, [3, 50]),
                'bootstrap': (hp.choice, [[True, False]]),
                'oob_score': (hp.choice, [[True, False]]),
                'model_params': {
                    'learning_rate': (hp.loguniform, [np.log(1e-3), np.log(1)]),
                    'loss': (hp.choice, [["linear", "square", "exponential"]])
                }
            },
            'bag_catboost': {
                'n_estimators': (hp.uniformint, [3, 50]),
                'bootstrap': (hp.choice, [[True, False]]),
                'oob_score': (hp.choice, [[True, False]]),
                'model_params': {
                    'max_depth': (hp.uniformint, [1, 11]),
                    'learning_rate': (hp.loguniform, [np.log(0.01), np.log(0.2)]),
                    'min_data_in_leaf': (hp.qloguniform, [0, 6, 1]),
                    'border_count': (hp.uniformint, [2, 255]),
                    'l2_leaf_reg': (hp.loguniform, [np.log(1e-8), np.log(10)]),
                    'loss_function': (hp.choice, [['Logloss', 'CrossEntropy']])
                }
            },
            'bag_xgboost': {
                'n_estimators': (hp.uniformint, [3, 50]),
                'bootstrap': (hp.choice, [[True, False]]),
                'oob_score': (hp.choice, [[True, False]]),
                'model_params': {
                    'max_depth': (hp.uniformint, [1, 7]),
                    'learning_rate': (hp.loguniform, [np.log(1e-3), np.log(1)]),
                    'subsample': (hp.uniform, [0.05, 0.99]),
                    'min_child_weight': (hp.uniform, [1, 21])
                }
            },
            'bag_lgbm': {
                'n_estimators': (hp.uniformint, [3, 50]),
                'bootstrap': (hp.choice, [[True, False]]),
                'oob_score': (hp.choice, [[True, False]]),
                'model_params': {
                    'class_weight': (hp.choice, [[None, 'balanced']]),
                    'num_leaves': (hp.uniformint, [2, 256]),
                    'learning_rate': (hp.loguniform, [np.log(0.01), np.log(0.2)]),
                    'colsample_bytree': (hp.uniform, [0.4, 1]),
                    'subsample': (hp.uniform, [0.4, 1]),
                    'reg_alpha': (hp.loguniform, [np.log(1e-8), np.log(10)]),
                    'reg_lambda': (hp.loguniform, [np.log(1e-8), np.log(10)])
                }
            },
            'bag_lgbmxt': {
                'n_estimators': (hp.uniformint, [3, 50]),
                'bootstrap': (hp.choice, [[True, False]]),
                'oob_score': (hp.choice, [[True, False]]),
                'model_params': {
                    'class_weight': (hp.choice, [[None, 'balanced']]),
                    'num_leaves': (hp.uniformint, [2, 256]),
                    'learning_rate': (hp.loguniform, [np.log(0.01), np.log(0.2)]),
                    'colsample_bytree': (hp.uniform, [0.4, 1]),
                    'subsample': (hp.uniform, [0.4, 1]),
                    'reg_alpha': (hp.loguniform, [np.log(1e-8), np.log(10)]),
                    'reg_lambda': (hp.loguniform, [np.log(1e-8), np.log(10)])
                }
            },
            'bag_catboostreg': {
                'n_estimators': (hp.uniformint, [3, 50]),
                'bootstrap': (hp.choice, [[True, False]]),
                'oob_score': (hp.choice, [[True, False]]),
                'model_params': {
                    'max_depth': (hp.uniformint, [1, 11]),
                    'learning_rate': (hp.loguniform, [np.log(0.01), np.log(0.2)]),
                    'min_data_in_leaf': (hp.qloguniform, [0, 6, 1]),
                    'border_count': (hp.uniformint, [2, 255]),
                    'l2_leaf_reg': (hp.loguniform, [np.log(1e-8), np.log(10)])
                }
            },
            'bag_xgboostreg': {
                'n_estimators': (hp.uniformint, [3, 50]),
                'bootstrap': (hp.choice, [[True, False]]),
                'oob_score': (hp.choice, [[True, False]]),
                'model_params': {
                    'max_depth': (hp.uniformint, [1, 7]),
                    'learning_rate': (hp.loguniform, [np.log(1e-3), np.log(1)]),
                    'subsample': (hp.uniform, [0.05, 0.99]),
                    'min_child_weight': (hp.uniform, [1, 21])
                }
            },
            'bag_lgbmreg': {
                'n_estimators': (hp.uniformint, [3, 50]),
                'bootstrap': (hp.choice, [[True, False]]),
                'oob_score': (hp.choice, [[True, False]]),
                'model_params': {
                    'num_leaves': (hp.uniformint, [2, 256]),
                    'learning_rate': (hp.loguniform, [np.log(0.01), np.log(0.2)]),
                    'colsample_bytree': (hp.uniform, [0.4, 1]),
                    'subsample': (hp.uniform, [0.4, 1]),
                    'reg_alpha': (hp.loguniform, [np.log(1e-8), np.log(10)]),
                    'reg_lambda': (hp.loguniform, [np.log(1e-8), np.log(10)])
                }
            },
            'bag_lgbmxtreg': {
                'n_estimators': (hp.uniformint, [3, 50]),
                'bootstrap': (hp.choice, [[True, False]]),
                'oob_score': (hp.choice, [[True, False]]),
                'model_params': {
                    'num_leaves': (hp.uniformint, [2, 256]),
                    'learning_rate': (hp.loguniform, [np.log(0.01), np.log(0.2)]),
                    'colsample_bytree': (hp.uniform, [0.4, 1]),
                    'subsample': (hp.uniform, [0.4, 1]),
                    'reg_alpha': (hp.loguniform, [np.log(1e-8), np.log(10)]),
                    'reg_lambda': (hp.loguniform, [np.log(1e-8), np.log(10)])
                }
            },
        }

        if self.custom_search_space is not None:
            if self.replace_default_search_space:
                parameters_per_operation.update(self.custom_search_space)
            else:
                for operation_name, operation_dct in self.custom_search_space.items():
                    parameters_per_operation[operation_name].update(operation_dct)

        return parameters_per_operation
