from copy import deepcopy
from os.path import join
from typing import List, Union

import numpy as np
from sklearn.metrics import mean_squared_error

from fedot.core.chains.chain import Chain
from fedot.core.data.data import InputData
from fedot.core.utils import default_fedot_data_dir
from fedot.sensitivity.operations_sensitivity.problem import MultiOperationsProblem
from fedot.sensitivity.operations_sensitivity.sa_and_sample_methods import sample_method_by_name, \
    analyze_method_by_name


class MultiOperationsAnalyze:
    def __init__(self, chain: Chain, train_data: InputData, test_data: InputData,
                 path_to_save=None):
        self.chain = chain
        self.train_data = train_data
        self.test_data = test_data
        self.problem = None
        self.analyze_method = None
        self.sample_method = None
        self.operation_types = None

        if not path_to_save:
            self.path_to_save = join(default_fedot_data_dir(), 'sensitivity')
        else:
            self.path_to_save = path_to_save

    def analyze(self,
                sa_method: str = 'sobol',
                sample_method: str = 'saltelli',
                sample_size: int = 100, ) -> List[dict]:
        if not self.chain.fitted_on_data:
            self.chain.fit(self.train_data)

        # define methods
        self.analyze_method = analyze_method_by_name.get(sa_method)
        self.sample_method = sample_method_by_name.get(sample_method)

        # create problem
        self.operation_types = [node.operation.operation_type for node in self.chain.nodes]
        self.problem = MultiOperationsProblem(self.operation_types)

        # sample
        samples = self.sample(sample_size)
        converted_samples = self.problem.convert_sample_to_dict(samples)

        response_matrix = self.get_operation_response_matrix(converted_samples)

        indices = self.analyze_method(self.problem.dictionary, samples, response_matrix)
        converted_to_json_indices = self.convert_results_to_json(problem=self.problem,
                                                                 si=indices)

        return [converted_to_json_indices]

    def sample(self, *args) -> Union[Union[List[Chain], Chain], np.array]:
        """

        :param args:
        :return:
        """
        sample_size = args[0]
        samples = self.sample_method(self.problem.dictionary, num_of_samples=sample_size)

        return samples

    def get_operation_response_matrix(self, samples: List[List[dict]]):
        operation_response_matrix = []
        for sample in samples:
            chain = deepcopy(self.chain)
            for node_id, params_per_node in enumerate(sample):
                chain.nodes[node_id].custom_params = sample

            chain.fit(self.train_data)
            prediction = chain.predict(self.test_data)
            mse_metric = mean_squared_error(y_true=self.test_data.target,
                                            y_pred=prediction.predict)
            operation_response_matrix.append(mse_metric)

        return np.array(operation_response_matrix)

    @staticmethod
    def convert_results_to_json(problem: MultiOperationsProblem, si: dict):
        sobol_indices = []
        for index in range(problem.num_vars):
            var_indices = {f"{problem.names_per_node[index]}": {
                'S1': list(si['S1'])[index],
                'S1_conf': list(si['S1_conf'])[index],
                'ST': list(si['ST'])[index],
                'ST_conf': list(si['ST_conf'])[index],
            }}
            sobol_indices.append(var_indices)

        data = {
            'problem': problem.dictionary,
            'sobol_indices': sobol_indices
        }

        return data
