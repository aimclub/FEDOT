from typing import List, Union
from abc import abstractmethod


class Problem:
    operation_params_with_bounds_by_operation_name = {
        'kmeans': {
            'n_clusters': [2, 10]
        },
        'xgboost': {
            'n_estimators': [10, 100],
            'max_depth': [1, 7],
            'learning_rate': [0.1, 0.9],
            'subsample': [0.05, 1.0],
            'min_child_weight': [1, 21]
        },
        'logit': {
            'C': [1e-2, 10.]
        },
        'knn': {
            'n_neighbors': [1, 50],
            'p': [1, 2]},
        'qda': {
            'reg_param': [0.1, 0.5]
        },
        'adareg': {
            'n_estimators': [80, 110],
            'learning_rate': [1e-3, 1.0],
        },
        'gbr': {
            'n_estimators': [80, 110],
            'learning_rate': [1e-3, 1.0],
            'max_depth': [1, 11],
            'min_samples_split': [2, 21],
            'min_samples_leaf': [1, 21],
            'subsample': [0.05, 1.0],
            'max_features': [0.05, 1.0],
            'alpha': [0.75, 0.99]
        },
        'rf': {
            'n_estimators': [80, 110],
            'max_features': [0.05, 1.01],
            'min_samples_split': [2, 10],
            'min_samples_leaf': [1, 15],
        },
        'lasso': {
            'alpha': [0.01, 10.0]
        },
        'ridge': {
            'alpha': [0.01, 10.0]
        },
        'rfr': {
            'n_estimators': [80, 110],
            'max_features': [0.05, 1.01],
            'min_samples_split': [2, 21],
            'min_samples_leaf': [1, 21],
        },
        'xgbreg': {
            'n_estimators': [80, 110],
            'max_depth': [1, 11],
            'learning_rate': [1e-3, 1.],
            'subsample': [0.05, 1.01],
            'min_child_weight': [1, 21],
        },
        'svr': {
            'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
            'C': [1e-4, 25.],
            'epsilon': [1e-4, 1.0]
        },
        'dtreg': {
            'max_depth': [1, 11],
            'min_samples_split': [2, 21],
            'min_samples_leaf': [1, 21]
        },
        'treg': {
            'n_estimators': [80, 110],
            'max_features': [0.05, 1.0],
            'min_samples_split': [2, 21],
            'min_samples_leaf': [1, 21],
        },
        'dt': {
            'max_depth': [1, 11],
            'min_samples_split': [2, 21],
            'min_samples_leaf': [1, 21]
        },
        'knnreg': {
            'n_neighbors': [1, 50],
            'p': [1, 2]
        },
        'arima': {
            'p': [1, 6],
            'd': [0, 1],
            'q': [1, 4]
        },
        'ar': {
            'lag_1': [2, 200],
            'lag_2': [2, 800]
        },
        'pca': {
            'n_components': [0.1, 0.99],
        },
        'kernel_pca': {
            'n_components': [1, 20]
        },
        'ransac_lin_reg': {
            'min_samples': [0.1, 0.9],
            'residual_threshold': [0.1, 1000.0],
            'max_trials': [50, 500],
            'max_skips': [50, 50000]
        },
        'ransac_non_lin_reg': {
            'min_samples': [0.1, 0.9],
            'residual_threshold': [0.1, 1000.0],
            'max_trials': [50, 500],
            'max_skips': [50, 50000]
        },
        'rfe_lin_reg': {
            'n_features_to_select': [0.5, 0.7, 0.9],
            'step': [0.1, 0.2]
        },
        'rfe_non_lin_reg': {
            'n_features_to_select': [0.5, 0.9],
            'step': [0.1, 0.2]
        },
        'poly_features': {
            'degree': [2, 3, 4],
        },
        'lagged': {
            'window_size': [10, 500]
        },
        'smoothing': {
            'window_size': [2, 20]
        },
        'gaussian_filter': {
            'sigma': [1, 5]
        },
    }

    INTEGER_PARAMS = ['n_estimators', 'n_neighbors', 'p', 'min_child_weight', 'max_depth']

    def __init__(self, operation_types: Union[List[str], str]):
        self.operation_types = operation_types
        self.num_vars = None
        self.names = None
        self.bounds = None

    @abstractmethod
    def convert_sample_to_dict(self, samples) -> Union[List[dict], List[List[dict]]]:
        raise NotImplementedError

    def clean_sample_variables(self, samples: List[dict]):
        """Make integer values for params if necessary"""
        for sample in samples:
            for key, value in sample.items():
                if key in self.INTEGER_PARAMS:
                    sample[key] = int(value)

        return samples

    @property
    def dictionary(self):
        problem = {
            'num_vars': self.num_vars,
            'names': self.names,
            'bounds': self.bounds
        }

        return problem


class OneOperationProblem(Problem):

    def __init__(self, operation_types: Union[List[str], str]):
        super().__init__(operation_types)
        self.params: dict = self.operation_params_with_bounds_by_operation_name.get(self.operation_types[0])
        self.num_vars: int = len(self.params)
        self.names: List[str] = list(self.params.keys())
        self.bounds: List[list] = list(self.params.values())

    def convert_sample_to_dict(self, samples) -> List[dict]:
        converted_samples = []
        for sample in samples:
            new_params = {}
            for index, value in enumerate(sample):
                new_params[self.names[index]] = value
            converted_samples.append(new_params)

        converted_and_cleaned_samples = self.clean_sample_variables(converted_samples)

        return converted_and_cleaned_samples

    def convert_for_dispersion_analysis(self, transposed_samples):
        """
        convert samples into dicts per param as follows:
        [
            [{'p1': 'v1'},{'p1': 'v2'},...{'p1': 'vn'},],
            [{'p2': 'v1'},{'p2': 'v2'},...{'p2': 'vn'},],
            ...
            [{'pm': 'v1'},{'pm': 'v2'},...{'pm': 'vn'},]]

        :param transposed_samples:
        :return:
        """

        converted_samples: List[list] = []
        for index, param in enumerate(self.names):
            samples_per_param = [{param: value} for value in transposed_samples[index]]
            cleaned = self.clean_sample_variables(samples_per_param)
            converted_samples.append(cleaned)

        return converted_samples


class MultiOperationsProblem(Problem):
    def __init__(self, operation_types: List[str]):
        super().__init__(operation_types)
        self.params: List[dict] = [self.operation_params_with_bounds_by_operation_name.get(operation_type)
                                   for operation_type in self.operation_types]
        self.num_vars: int = sum([len(operation_params) for operation_params in self.params])
        self.names: List[str] = list()
        self.names_per_node: List[list] = list()
        self.bounds: List[list] = list()
        for operation_params in self.params:
            self.names.extend(list(operation_params.keys()))
            self.names_per_node.append(list(operation_params.keys()))
            self.bounds.extend(list(operation_params.values()))

    def convert_sample_to_dict(self, samples) -> Union[List[dict], List[List[dict]]]:
        all_operations_new_params: List[List[dict]] = list()
        for sample in samples:
            new_params_per_operation: List[dict] = list()
            border_index = 0
            for node_id, params_names in enumerate(self.names_per_node):
                new_params_values_per_operation = sample[border_index:len(params_names)]
                node_params = dict(zip(params_names, new_params_values_per_operation))
                new_params_per_operation.append(node_params)
                border_index = len(params_names) + 1
            cleaned_new_params_per_params = self.clean_sample_variables(new_params_per_operation)
            all_operations_new_params.append(cleaned_new_params_per_params)

        return all_operations_new_params

        # Apply new params to nodes in chain in operation sensitivity
