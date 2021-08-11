import json
from abc import abstractmethod
from os.path import join
from typing import List, Union

from fedot.core.utils import fedot_project_root


def load_params_bounds_file():
    params_bounds_file = join(fedot_project_root(), 'fedot', 'sensitivity',
                              'operations_hp_sensitivity', 'params_bounds.json')
    with open(params_bounds_file, 'r') as file:
        data = json.load(file)

    return data


class Problem:
    operation_params_with_bounds_by_operation_name = load_params_bounds_file()

    INTEGER_PARAMS = ['n_estimators', 'n_neighbors', 'p', 'min_child_weight', 'max_depth', 'n_clusters',
                      'min_samples_split', 'min_samples_leaf', 'd',
                      'q', 'lag_1', 'lag_2', 'max_trials', 'max_skips',
                      'degree', 'window_size', 'sigma']

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
            if sample is not None:
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
        self.params: dict = \
            self.operation_params_with_bounds_by_operation_name.get(self.operation_types[0])
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

    def convert_for_dispersion_analysis(self, transposed_samples) -> List[List[dict]]:
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
        self.params: List[dict] = \
            [self.operation_params_with_bounds_by_operation_name.get(operation_type)
             for operation_type in self.operation_types]
        self.num_vars: int = sum([len(operation_params) for operation_params in self.params
                                  if operation_params is not None])
        self.names: List[str] = list()
        self.names_per_node: dict = dict()
        self.bounds: List[list] = list()
        for index, operation_params in enumerate(self.params):
            if operation_params is not None:
                self.names.extend(list(operation_params.keys()))
                self.names_per_node[str(index)] = list(operation_params.keys())
                self.bounds.extend(list(operation_params.values()))
            else:
                self.names_per_node[str(index)] = None

    def convert_sample_to_dict(self, samples) -> Union[List[dict], List[List[dict]]]:
        all_operations_new_params: List[List[dict]] = list()
        for sample in samples:
            new_params_per_operation: List[Union[dict, None]] = list()
            border_index = 0
            for _, params_names in self.names_per_node.items():
                if params_names is not None:
                    new_params_values_per_operation = \
                        sample[border_index:border_index + len(params_names)]
                    node_params = dict(zip(params_names, new_params_values_per_operation))
                    new_params_per_operation.append(node_params)
                    border_index += len(params_names)
                else:
                    new_params_per_operation.append(None)

            cleaned_new_params_per_params = self.clean_sample_variables(new_params_per_operation)
            all_operations_new_params.append(cleaned_new_params_per_params)

        return all_operations_new_params

        # Apply new params to nodes in pipeline in operation sensitivity
