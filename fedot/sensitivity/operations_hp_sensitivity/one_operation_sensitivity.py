from copy import deepcopy
from os.path import join
from threading import Lock, Thread
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error

from fedot.core.data.data import InputData
from fedot.core.log import default_log
from fedot.core.operations.operation_template import extract_operation_params
from fedot.core.pipelines.node import Node
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.utils import default_fedot_data_dir
from fedot.sensitivity.node_sa_approaches import NodeAnalyzeApproach
from fedot.sensitivity.operations_hp_sensitivity.problem import OneOperationProblem
from fedot.sensitivity.operations_hp_sensitivity.sa_and_sample_methods import (
    analyze_method_by_name,
    sample_method_by_name
)
from fedot.sensitivity.sa_requirements import HyperparamsAnalysisMetaParams, SensitivityAnalysisRequirements


class OneOperationHPAnalyze(NodeAnalyzeApproach):
    lock = Lock()

    def __init__(self, pipeline: Pipeline, train_data, test_data: InputData,
                 requirements: SensitivityAnalysisRequirements = None, path_to_save=None):
        super().__init__(pipeline, train_data, test_data, path_to_save)

        requirements = SensitivityAnalysisRequirements() if requirements is None else requirements
        self.requirements: HyperparamsAnalysisMetaParams = requirements.hp_analysis_meta

        self.analyze_method = analyze_method_by_name.get(self.requirements.analyze_method)
        self.sample_method = sample_method_by_name.get(self.requirements.sample_method)
        self.problem = None
        self.operation_type = None
        self.data_under_lock: dict = {}

        self.path_to_save = \
            join(default_fedot_data_dir(), 'sensitivity', 'nodes_sensitivity') if path_to_save is None else path_to_save
        self.log = default_log(self)

    def analyze(self, node: Node,
                is_dispersion_analysis: bool = False) -> Union[dict, float]:

        # check whether the pipeline is fitted
        if not self._pipeline.is_fitted:
            self._pipeline.fit(self._train_data)

        # create problem
        self.operation_type = node.operation.operation_type
        self.problem = OneOperationProblem(operation_types=[self.operation_type])

        # sample
        samples = self.sample(self.requirements.sample_size, node)

        response_matrix = self._get_response_matrix(samples)
        indices = self.analyze_method(self.problem.dictionary, samples, response_matrix)
        converted_to_json_indices = self._convert_indices_to_json(problem=self.problem,
                                                                  si=indices)

        # Analyze the dispersion of params
        if is_dispersion_analysis:
            self._dispersion_analysis(node=node,
                                      sample_size=self.requirements.sample_size)

        return converted_to_json_indices

    def sample(self, *args) -> Union[List[Pipeline], Pipeline]:
        """
        """

        sample_size, node = args
        samples: List[np.array] = self.sample_method(self.problem.dictionary, num_of_samples=sample_size)

        converted_samples: List[dict] = self.problem.convert_sample_to_dict(samples)
        sampled_pipelines: List[Pipeline] = self._apply_params_to_node(params=converted_samples,
                                                                       node=node)

        return sampled_pipelines

    def _apply_params_to_node(self, params: List[dict], node: Node) -> List[Pipeline]:
        sampled_pipelines: List[Pipeline] = list()
        for sample in params:
            copied_pipeline = deepcopy(self._pipeline)
            node_id = self._pipeline.nodes.index(node)
            copied_pipeline.nodes[node_id].parameters = sample
            sampled_pipelines.append(copied_pipeline)

        return sampled_pipelines

    def _get_response_matrix(self, samples: List[Pipeline]):
        operation_response_matrix = []
        for sampled_pipeline in samples:
            sampled_pipeline.fit(self._train_data)
            prediction = sampled_pipeline.predict(self._test_data)
            mse_metric = mean_squared_error(y_true=self._test_data.target,
                                            y_pred=prediction.predict)
            operation_response_matrix.append(mse_metric)

        return np.array(operation_response_matrix)

    def _dispersion_analysis(self, node: Node, sample_size: int):
        samples: np.array = self.sample_method(self.problem.dictionary, num_of_samples=sample_size)
        transposed_samples = samples.T
        converted_samples = self.problem.convert_for_dispersion_analysis(transposed_samples)

        jobs = [Thread(target=self._evaluate_variance,
                       args=(params, transposed_samples[index], node))
                for index, params in enumerate(converted_samples)]

        for job in jobs:
            job.start()

        for job in jobs:
            job.join()

        self._visualize_variance()

    def _evaluate_variance(self, params: List[dict], samples, node: Node):
        # default values of param & loss
        param_name = list(params[0].keys())[0]
        default_param_value = extract_operation_params(node).get(param_name)
        pipelines_with_applied_params = self._apply_params_to_node(params, node)

        # percentage ratio
        samples = (samples - default_param_value) / default_param_value
        response_matrix = self._get_response_matrix(pipelines_with_applied_params)
        response_matrix = (response_matrix - np.mean(response_matrix)) / \
                          (max(response_matrix) - min(response_matrix))

        OneOperationHPAnalyze.lock.acquire()
        self.data_under_lock[f'{param_name}'] = [samples.reshape(1, -1)[0], response_matrix]
        OneOperationHPAnalyze.lock.release()

    def _visualize_variance(self):
        x_ticks_param = list()
        x_ticks_loss = list()
        for param in self.data_under_lock.keys():
            x_ticks_param.append(param)
            x_ticks_loss.append(f'{param}_loss')
        param_values_data = list()
        losses_data = list()
        for value in self.data_under_lock.values():
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

    @staticmethod
    def _convert_indices_to_json(problem: OneOperationProblem, si: dict) -> dict:
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
