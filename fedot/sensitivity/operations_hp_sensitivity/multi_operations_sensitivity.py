from copy import deepcopy
from os.path import join
from typing import List, Optional, Union

import numpy as np

from fedot.core.composer.metrics import MSE
from fedot.core.data.data import InputData
from fedot.core.log import default_log
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.utils import default_fedot_data_dir
from fedot.sensitivity.operations_hp_sensitivity.problem import MultiOperationsProblem, Problem
from fedot.sensitivity.operations_hp_sensitivity.sa_and_sample_methods import (
    analyze_method_by_name,
    sample_method_by_name
)
from fedot.sensitivity.sa_requirements import HyperparamsAnalysisMetaParams, SensitivityAnalysisRequirements


class MultiOperationsHPAnalyze:
    """Provides with analysis of all the :class:`Pipeline`'s operations hyperparameters
    using sample and analyze methods from ``SALib``

    Args:
        pipeline: :obj:`Pipeline` object to analyze
        train_data: data used for :obj:`Pipeline` training
        test_data: data used for :obj:`Pipeline` validation
        requirements: extra requirements to define specific details for different approaches.
            See :class:`SensitivityAnalysisRequirements` class documentation.
        path_to_save: path to save results to. Default: ``~home/Fedot/sensitivity/``
    """

    def __init__(self, pipeline: Pipeline, train_data: InputData, test_data: InputData,
                 requirements: SensitivityAnalysisRequirements = None,
                 path_to_save=None):
        self._pipeline = pipeline
        self._train_data = train_data
        self._test_data = test_data
        self.problem: Optional[Problem] = None
        requirements = SensitivityAnalysisRequirements() if requirements is None else requirements
        self.requirements: HyperparamsAnalysisMetaParams = requirements.hp_analysis_meta
        self.analyze_method = analyze_method_by_name.get(self.requirements.analyze_method)
        self.sample_method = sample_method_by_name.get(self.requirements.sample_method)

        self.operation_types = None
        self.path_to_save = \
            join(default_fedot_data_dir(), 'sensitivity', 'pipeline_sensitivity') \
                if path_to_save is None else path_to_save
        self.log = default_log(self)

    def analyze(self) -> dict:
        """Analyze all the hyperparameters af all :obj:`Pipeline` operations using ``SA`` methods.\n
        Default: Sobol method with Saltelli sample algorithm

        Returns:
            dict: ``Main`` and total ``Sobol`` indices for every parameter per node
        """

        if not self._pipeline.is_fitted:
            self._pipeline.fit(self._train_data)

        # create problem
        self.operation_types = [node.operation.operation_type for node in self._pipeline.nodes]
        self.problem = MultiOperationsProblem(self.operation_types)

        # sample
        self.log.info('Making hyperparameters samples')
        samples = self.sample(self.requirements.sample_size)
        response_matrix = self._get_response_matrix(samples)

        self.log.info('Start hyperparameters sensitivity analysis')
        indices = self.analyze_method(self.problem.dictionary, samples, response_matrix)
        converted_to_json_indices = self.convert_results_to_json(problem=self.problem,
                                                                 si=indices)

        self.log.info('Finish hyperparameters sensitivity analysis')

        return converted_to_json_indices

    def sample(self, *args) -> Union[List[Pipeline], Pipeline]:
        """Makes hyperparameters samples

        Args:
            args: i.e. ``sample_size``
        Returns:
            List[Pipeline]: new sampled hyperparameters
        """

        sample_size = args[0]
        samples = self.sample_method(self.problem.dictionary, num_of_samples=sample_size)
        converted_samples: List[List[dict]] = self.problem.convert_sample_to_dict(samples)
        sampled_pipelines: List[Pipeline] = self._apply_params_to_node(params=converted_samples)

        return sampled_pipelines

    def _apply_params_to_node(self, params: List[List[dict]]) -> List[Pipeline]:
        sampled_pipelines: List[Pipeline] = list()
        for sample in params:
            copied_pipeline = deepcopy(self._pipeline)
            for node_id, params_per_node in enumerate(sample):
                copied_pipeline.nodes[node_id].parameters = params_per_node
                sampled_pipelines.append(copied_pipeline)

        return sampled_pipelines

    def _get_response_matrix(self, samples: List[Pipeline]):
        operation_response_matrix = []
        for sampled_pipeline in samples:
            sampled_pipeline.fit(self._train_data)
            prediction = sampled_pipeline.predict(self._test_data)
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
