import json
from os.path import join
from typing import List, Optional, Type

import matplotlib.pyplot as plt

from fedot.core.chains.chain import Chain
from fedot.core.chains.chain_convert import chain_as_nx_graph
from fedot.core.data.data import InputData
from fedot.core.log import Log, default_log
from fedot.core.utils import default_fedot_data_dir
from fedot.sensitivity.node_sensitivity import NodeAnalysis, NodeAnalyzeApproach


def get_nodes_degrees(chain: Chain):
    """ Nodes degree as the number of edges the node has:
     k = k(in) + k(out)"""
    graph, _ = chain_as_nx_graph(chain)
    index_degree_pairs = graph.degree
    node_degrees = [node_degree[1] for node_degree in index_degree_pairs]
    return node_degrees


class ChainStructureAnalyze:
    """
    This class is for Chain Sensitivity analysis.
    It takes nodes(indices) and approaches to be applied to chosen nodes.
    To define which nodes to analyze
    pass their ids to nodes_ids_to_analyze or pass True to all_nodes flag.

    :param chain: chain object to analyze
    :param train_data: data used for Chain training
    :param test_data: data used for Chain validation
    :param approaches: methods applied to nodes to modify the chain or analyze certain operations.\
    Default: [NodeDeletionAnalyze, NodeTuneAnalyze, NodeReplaceOperationAnalyze]
    :param metric: metric used for validation. Default: see MetricByTask
    :param nodes_ids_to_analyze: numbers of nodes to analyze. Default: all nodes
    :param all_nodes: flag, used to choose all nodes to analyze.Default: False.
    :param path_to_save: path to save results to. Default: ~home/Fedot/sensitivity
    :param interactive_mode: flag for interactive visualization or saving plots to file.
    Default: False
    :param log: log: Log object to record messages
    """

    def __init__(self, chain: Chain, train_data: InputData, test_data: InputData,
                 approaches: Optional[List[Type[NodeAnalyzeApproach]]] = None,
                 metric: str = None, nodes_ids_to_analyze: List[int] = None,
                 all_nodes: bool = False, path_to_save=None, log: Log = None):

        self.chain = chain
        self.train_data = train_data
        self.test_data = test_data
        self.approaches = approaches
        self.metric = metric

        if all_nodes and nodes_ids_to_analyze:
            raise ValueError("Choose only one parameter between all_nodes and nodes_ids_to_analyze")
        elif not all_nodes and not nodes_ids_to_analyze:
            raise ValueError("Define nodes to analyze: all_nodes or nodes_ids_to_analyze")

        if all_nodes:
            self.nodes_ids_to_analyze = [i for i in range(len(self.chain.nodes))]
        else:
            self.nodes_ids_to_analyze = nodes_ids_to_analyze

        if not path_to_save:
            self.path_to_save = join(default_fedot_data_dir(), 'sensitivity')
        else:
            self.path_to_save = path_to_save

        if not log:
            self.log = default_log(__name__)
        else:
            self.log = log

    def analyze(self) -> dict:
        """
        Main method to run the analyze process for every node.

        :return nodes_results: dict with analysis result per Node
        """

        nodes_results = dict()
        operation_types = []
        for index in self.nodes_ids_to_analyze:
            node_result = NodeAnalysis(approaches=self.approaches, path_to_save=self.path_to_save). \
                analyze(chain=self.chain, node_id=index,
                        train_data=self.train_data,
                        test_data=self.test_data,
                        is_save=False)
            operation_types.append(self.chain.nodes[index].operation.operation_type)

            nodes_results[f'id = {index}, operation = {self.chain.nodes[index].operation.operation_type}'] = node_result

        self._visualize_result_per_approach(nodes_results, operation_types)
        if len(self.nodes_ids_to_analyze) == len(self.chain.nodes):
            self._visualize_degree_correlation(nodes_results)
        self._save_results_to_json(nodes_results)
        return nodes_results

    def _save_results_to_json(self, result):
        result_file = join(self.path_to_save, 'chain_SA_results.json')
        with open(result_file, 'w', encoding='utf-8') as file:
            file.write(json.dumps(result, indent=4))

        self.log.message(f'Chain Sensitivity Analysis results were saved to {result_file}')

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
            plt.ylabel('quality (changed_chain_metric/original_metric) - 1')

            file_path = join(self.path_to_save,
                             f'{self.approaches[index].__name__}.jpg')

            plt.savefig(file_path)
            self.log.message(f'Chain Sensitivity Analysis visualized results were saved to {file_path}')

    def _visualize_degree_correlation(self, results: dict):
        nodes_degrees = get_nodes_degrees(self.chain)
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
            approach_result = [result[f'{approach.__name__}'] - 1 for result in results.values()]
            gathered_results.append(approach_result)

        return gathered_results
