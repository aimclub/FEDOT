from copy import deepcopy
from os.path import join
from threading import Thread, Lock
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error

from fedot.core.chains.chain import Chain
from fedot.core.data.data import InputData
from fedot.core.operations.operation_template import extract_operation_params
from fedot.sensitivity.node_sensitivity import NodeAnalyzeApproach
from fedot.sensitivity.operations_sensitivity.problem import Problem, OneOperationProblem
from fedot.sensitivity.operations_sensitivity.sa_and_sample_methods import sample_method_by_name, analyze_method_by_name


class OneOperationAnalyze(NodeAnalyzeApproach):
    lock = Lock()

    def __init__(self, chain: Chain, train_data, test_data: InputData, path_to_save=None):
        super().__init__(chain, train_data, test_data, path_to_save)
        self.operation_params = None
        self.operation_type = None
        self.problem = None
        self.analyze_method = None
        self.sample_method = None
        self.manager_dict = {}

    def analyze(self, node_id: int,
                sa_method: str = 'sobol',
                sample_method: str = 'saltelli',
                sample_size: int = 1,
                is_dispersion_analysis: bool = False) -> Union[List[dict], float]:
        """

        :param node_id:
        :param sa_method:
        :param sample_method:
        :param sample_size:
        :param is_dispersion_analysis:
        :return:
        """

        # check whether the chain is fitted
        if not self._chain.fitted_on_data:
            self._chain.fit(self._train_data)

        # define methods
        self.analyze_method = analyze_method_by_name.get(sa_method)
        self.sample_method = sample_method_by_name.get(sample_method)

        # create problem
        self.operation_type: str = self._chain.nodes[node_id].operation.operation_type
        self.problem: Problem = OneOperationProblem(operation_types=[self.operation_type])

        # sample
        samples = self.sample(sample_size)
        converted_samples = self.problem.convert_sample_to_dict(samples)

        response_matrix = self.get_operation_response_matrix(converted_samples,
                                                             node_id)
        indices = self.analyze_method(self.problem.dictionary, samples, response_matrix)
        converted_to_json_indices = self.convert_results_to_json(problem=self.problem,
                                                                 si=indices)

        # Analyze the dispersion of params
        if is_dispersion_analysis:
            self.dispersion_analysis(node_id=node_id,
                                     samples=samples)

        return [converted_to_json_indices]

    def sample(self, *args) -> Union[Union[List[Chain], Chain], np.array]:
        """

        :param args:
        :return:
        """
        sample_size = args[0]
        samples = self.sample_method(self.problem.dictionary, num_of_samples=sample_size)

        return samples

    def get_operation_response_matrix(self, samples: List[dict], node_id: int):
        operation_response_matrix = []
        for sample in samples:
            chain = deepcopy(self._chain)
            chain.nodes[node_id].custom_params = sample

            chain.fit(self._train_data)
            prediction = chain.predict(self._test_data)
            mse_metric = mean_squared_error(y_true=self._test_data.target,
                                            y_pred=prediction.predict)
            operation_response_matrix.append(mse_metric)

        return np.array(operation_response_matrix)

    def evaluate_variance(self, params: List[dict], samples, node_id):
        # default values of param & loss
        param_name = list(params[0].keys())[0]
        default_param_value = extract_operation_params(self._chain.nodes[node_id]).get(param_name)

        # percentage ratio
        samples = (samples - default_param_value) / default_param_value
        response_matrix = self.get_operation_response_matrix(params, node_id)
        response_matrix = (response_matrix - np.mean(response_matrix)) / \
                          (max(response_matrix) - min(response_matrix))

        OneOperationAnalyze.lock.acquire()
        self.manager_dict[f'{param_name}'] = [samples.reshape(1, -1)[0], response_matrix]
        OneOperationAnalyze.lock.release()

    def _visualize_variance(self, data: dict):
        x_ticks_param = list()
        x_ticks_loss = list()
        for param in data.keys():
            x_ticks_param.append(param)
            x_ticks_loss.append(f'{param}_loss')
        param_values_data = list()
        losses_data = list()
        for value in data.values():
            param_values_data.append(value[0])
            losses_data.append(value[1])

        fig, (ax1, ax2) = plt.subplots(2, figsize=(20, 10))
        ax1.boxplot(param_values_data)
        ax2.boxplot(losses_data)
        ax1.set_title('param')
        ax1.set_xticks(range(1, len(x_ticks_param) + 1))
        ax1.set_xticklabels(x_ticks_param)
        ax2.set_title('loss')
        ax2.set_xticks(range(1, len(x_ticks_loss) + 1))
        ax2.set_xticklabels(x_ticks_loss)

        plt.savefig(join(self._path_to_save, f'{self.operation_type}_hp_sa.jpg'))

    def dispersion_analysis(self, node_id, samples: np.array):
        transposed_samples = samples.T
        converted_samples = self.problem.convert_for_dispersion_analysis(transposed_samples)

        jobs = [Thread(target=self.evaluate_variance,
                       args=(params, transposed_samples[index], node_id))
                for index, params in enumerate(converted_samples)]

        for job in jobs:
            job.start()

        for job in jobs:
            job.join()

        self._visualize_variance(data=self.manager_dict)

    @staticmethod
    def convert_results_to_json(problem: Problem, si: dict):
        sobol_indices = []
        for index in range(problem.num_vars):
            var_indices = {f"{problem.names[index]}": {
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
