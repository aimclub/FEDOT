import json
from os.path import join
from typing import Optional, List, Type

import numpy as np
from matplotlib import pyplot as plt

from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.node import Node
from fedot.core.dag.graph_operator import GraphOperator
from fedot.core.data.data import InputData
from fedot.core.log import Log, default_log
from fedot.core.utils import default_fedot_data_dir
from fedot.sensitivity.node_sa_approaches import NodeAnalyzeApproach, NodeAnalysis
from fedot.sensitivity.sa_requirements import SensitivityAnalysisRequirements


class NodesAnalysis:
    """
    This class is for nodes sensitivity analysis within a Pipeline .
    It takes nodes and approaches to be applied to chosen nodes.
    To define which nodes to analyze pass them to nodes_to_analyze filed
    or all nodes will be analyzed.

    :param pipeline: pipeline object to analyze
    :param train_data: data used for Pipeline training
    :param test_data: data used for Pipeline validation
    :param approaches: methods applied to nodes to modify the pipeline or analyze certain operations.\
    Default: [NodeDeletionAnalyze, NodeReplaceOperationAnalyze]
    :param nodes_to_analyze: nodes to analyze. Default: all nodes
    :param path_to_save: path to save results to. Default: ~home/Fedot/sensitivity
    :param log: log: Log object to record messages
    """

    def __init__(self, pipeline: Pipeline, train_data: InputData, test_data: InputData,
                 approaches: Optional[List[Type[NodeAnalyzeApproach]]] = None,
                 requirements: SensitivityAnalysisRequirements = None,
                 path_to_save=None, log: Log = None,
                 nodes_to_analyze: List[Node] = None):

        self.pipeline = pipeline
        self.train_data = train_data
        self.test_data = test_data
        self.approaches = approaches
        self.requirements = \
            SensitivityAnalysisRequirements() if requirements is None else requirements
        self.metric = self.requirements.metric
        self.log = default_log(__name__) if log is None else log
        self.path_to_save = \
            join(default_fedot_data_dir(), 'sensitivity', 'nodes_sensitivity') if path_to_save is None else path_to_save

        if not nodes_to_analyze:
            self.log.message('Nodes to analyze are not defined. All nodes will be analyzed.')
            self.nodes_to_analyze = self.pipeline.nodes
        else:
            self.nodes_to_analyze = nodes_to_analyze

    def analyze(self) -> dict:
        """
        Main method to run the analyze process for every node.

        :return nodes_results: dict with analysis result per Node
        """

        nodes_results = dict()
        operation_types = []
        for node in self.nodes_to_analyze:
            node_result = NodeAnalysis(approaches=self.approaches,
                                       approaches_requirements=self.requirements,
                                       path_to_save=self.path_to_save). \
                analyze(pipeline=self.pipeline, node=node,
                        train_data=self.train_data,
                        test_data=self.test_data)
            operation_types.append(node.operation.operation_type)

            nodes_results[f'id = {self.pipeline.nodes.index(node)}, ' \
                          f'operation = {node.content["name"].operation_type}'] = node_result

        if self.requirements.is_visualize:
            self._visualize_result_per_approach(nodes_results, operation_types)

        if len(self.nodes_to_analyze) == len(self.pipeline.nodes):
            self._visualize_degree_correlation(nodes_results)

        if self.requirements.is_save:
            self._save_results_to_json(nodes_results)
        return nodes_results

    def _save_results_to_json(self, result: dict):
        result_file = join(self.path_to_save, 'nodes_SA_results.json')
        with open(result_file, 'w', encoding='utf-8') as file:
            file.write(json.dumps(result, indent=4))

        self.log.message(f'Pipeline Sensitivity Analysis results were saved to {result_file}')

    def _visualize_result_per_approach(self, results: dict, types: list):
        gathered_results = self._extract_result_values(results)

        for index, result in enumerate(gathered_results):
            colors = ['r' if y < 0 else 'g' for y in result]
            fig, ax = plt.subplots(figsize=(15, 10))
            ax.bar(range(len(results)), result, width=1.0, edgecolor='black', bottom=1,
                   color=colors)
            ax.hlines(1, -1, len(types) + 1, linestyle='--')
            ax.set_xticks(range(len(results)))
            ax.set_xticklabels(types, rotation=45)
            plt.xlabel('iteration')
            plt.ylabel('quality (changed_pipeline_metric/original_metric) - 1')

            file_path = join(self.path_to_save,
                             f'{self.approaches[index].__name__}.jpg')

            plt.savefig(file_path)
            self.log.message(f'Pipeline Sensitivity Analysis visualized results were saved to {file_path}')

    def _visualize_degree_correlation(self, results: dict):
        nodes_degrees = GraphOperator(self.pipeline).get_nodes_degrees()
        gathered_results = self._extract_result_values(results)
        for index, result in enumerate(gathered_results):
            fig, ax = plt.subplots(figsize=(15, 10))
            ax.scatter(nodes_degrees, result)

            file_path = join(self.path_to_save,
                             f'{self.approaches[index].__name__}_cor.jpg')
            plt.savefig(file_path)
            self.log.message(f'Nodes degree correlation visualized results were saved to {file_path}')

    def _extract_result_values(self, results):
        gathered_results = []
        for approach in self.approaches:
            approach_result = [np.mean(result[f'{approach.__name__}']) - 1 for result in results.values()]
            gathered_results.append(approach_result)

        return gathered_results
