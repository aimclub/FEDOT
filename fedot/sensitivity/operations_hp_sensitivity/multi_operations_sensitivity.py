from copy import deepcopy
from os.path import join
from typing import List, Union, Optional

import numpy as np

from fedot.core.chains.chain import Chain
from fedot.core.composer.metrics import MSE
from fedot.core.data.data import InputData
from fedot.core.log import default_log, Log
from fedot.core.utils import default_fedot_data_dir
from fedot.sensitivity.operations_hp_sensitivity.problem import MultiOperationsProblem, Problem
from fedot.sensitivity.operations_hp_sensitivity.sa_and_sample_methods import analyze_method_by_name, \
    sample_method_by_name
from fedot.sensitivity.sa_requirements import HyperparamsAnalysisMetaParams
from fedot.sensitivity.sa_requirements import SensitivityAnalysisRequirements


class MultiOperationsHPAnalyze:
    """
    Provides with analysis of all the Chain's operations hyperparameters
    using sample and analyze methods from SALib.

    :param chain: chain object to analyze
    :param train_data: data used for Chain training
    :param test_data: data used for Chain validation
    :param requirements: extra requirements to define specific details for different approaches.\
    See SensitivityAnalysisRequirements class documentation.
    :param path_to_save: path to save results to. Default: ~home/Fedot/sensitivity/
    Default: False
    :param log: log: Log object to record messages
    """

    def __init__(self, chain: Chain, train_data: InputData, test_data: InputData,
                 requirements: SensitivityAnalysisRequirements = None,
                 path_to_save=None, log: Log = None):
        self._chain = chain
        self._train_data = train_data
        self._test_data = test_data
        self.problem: Optional[Problem] = None
        requirements = SensitivityAnalysisRequirements() if requirements is None else requirements
        self.requirements: HyperparamsAnalysisMetaParams = requirements.hp_analysis_meta
        self.analyze_method = analyze_method_by_name.get(self.requirements.analyze_method)
        self.sample_method = sample_method_by_name.get(self.requirements.sample_method)

        self.operation_types = None
        self.path_to_save = \
            join(default_fedot_data_dir(), 'sensitivity', 'chain_sensitivity') if path_to_save is None else path_to_save
        self.log = default_log(__name__) if log is None else log

    def analyze(self) -> dict:
        """
        Analyze all the hyperparameters af all Chain operations using SA methods.
        Default: Sobol method with Saltelli sample algorithm

        :param sample_method: string name of sampling method from SALib
        :param analyze_method: string name of analyzing method from SALib
        :param sample_size: number of hyperparameters samples needed for analysis.\
        Default: 100.
        :return: Main and total Sobol indices for every parameter per node.
        """
        if not self._chain.fitted_on_data:
            self._chain.fit(self._train_data)

        # create problem
        self.operation_types = [node.operation.operation_type for node in self._chain.nodes]
        self.problem = MultiOperationsProblem(self.operation_types)

        # sample
        self.log.message('Making hyperparameters samples')
        samples = self.sample(self.requirements.sample_size)
        response_matrix = self._get_response_matrix(samples)

        self.log.message('Start hyperparameters sensitivity analysis')
        indices = self.analyze_method(self.problem.dictionary, samples, response_matrix)
        converted_to_json_indices = self.convert_results_to_json(problem=self.problem,
                                                                 si=indices)

        self.log.message('Finish hyperparameters sensitivity analysis')

        return converted_to_json_indices

    def sample(self, *args) -> Union[List[Chain], Chain]:
        """
        Makes hyperparameters samples

        :param args: i.e. sample_size
        :return List[Chain]: List of Chains with new sampled hyperparameters
        """
        sample_size = args[0]
        samples = self.sample_method(self.problem.dictionary, num_of_samples=sample_size)
        converted_samples: List[List[dict]] = self.problem.convert_sample_to_dict(samples)
        sampled_chains: List[Chain] = self._apply_params_to_node(params=converted_samples)

        return sampled_chains

    def _apply_params_to_node(self, params: List[List[dict]]) -> List[Chain]:
        sampled_chains: List[Chain] = list()
        for sample in params:
            copied_chain = deepcopy(self._chain)
            for node_id, params_per_node in enumerate(sample):
                copied_chain.nodes[node_id].custom_params = params_per_node
                sampled_chains.append(copied_chain)

        return sampled_chains

    def _get_response_matrix(self, samples: List[Chain]):
        operation_response_matrix = []
        for sampled_chain in samples:
            sampled_chain.fit(self._train_data)
            prediction = sampled_chain.predict(self._test_data)
            mse_metric = MSE().metric(reference=self._test_data,
                                      predicted=prediction)
            operation_response_matrix.append(mse_metric)

        return np.array(operation_response_matrix)

    @staticmethod
    def convert_results_to_json(problem: MultiOperationsProblem, si: dict) -> dict:
        sobol_indices = []
        for index, param_name in enumerate(problem.names):
            var_indices = {f"{param_name}": {
                'S1': list(si['S1'])[index],
                'S1_conf': list(si['S1_conf'])[index],
                'ST': list(si['ST'])[index],
                'ST_conf': list(si['ST_conf'])[index],
            }}
            sobol_indices.append(var_indices)

        indices_per_operation = dict()
        border_index = 0
        for operation_name, params_names in problem.names_per_node.items():
            if params_names is not None:
                indices_per_operation[operation_name] = \
                    sobol_indices[border_index:border_index + len(params_names)]
                border_index += len(params_names)
            else:
                indices_per_operation[operation_name] = 'None'

        data = {
            'problem': problem.dictionary,
            'sobol_indices_per_operation': indices_per_operation
        }

        return data
