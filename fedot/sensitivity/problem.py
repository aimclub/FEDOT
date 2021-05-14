from typing import List


class Problem:
    operation_params_with_bounds_by_operation_name = {
        'xgboost': {
            'n_estimators': [10, 100],
            'max_depth': [1, 7],
            'learning_rate': [0.1, 0.9],
            'subsample': [0.05, 1.0],
            'min_child_weight': [1, 21]},
        'logit': {
            'C': [1e-2, 10.]},
        'knn': {
            'n_neighbors': [1, 50],
            'p': [1, 2]},
        'qda': {
            'reg_param': [0.1, 0.5]},
    }

    INTEGER_PARAMS = ['n_estimators', 'n_neighbors', 'p', 'min_child_weight', 'max_depth']

    def __init__(self, operation_type: str):
        self.operation_type = operation_type
        self.params = self.operation_params_with_bounds_by_operation_name.get(self.operation_type)
        self.num_vars = len(self.params)
        self.names = list(self.params.keys())
        self.bounds = list()

        for key, bounds in self.params.items():
            if bounds[0] is not str:
                bounds = list(bounds)
                self.bounds.append([bounds[0], bounds[-1]])
            else:
                self.bounds.append(bounds)

    @property
    def dictionary(self):
        problem = {
            'num_vars': len(self.params),
            'names': list(self.params.keys()),
            'bounds': self.bounds
        }

        return problem

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

    def clean_sample_variables(self, samples: List[dict]):
        """Make integer values for params if necessary"""
        for sample in samples:
            for key, value in sample.items():
                if key in self.INTEGER_PARAMS:
                    sample[key] = int(value)

        return samples
