from os.path import join
from typing import List, Optional, Type

from fedot.core.chains.chain import Chain
from fedot.core.data.data import InputData
from fedot.core.utils import default_fedot_data_dir
from fedot.sensitivity.node_sensitivity import NodeAnalyzeApproach, \
    NodeAnalysis
import matplotlib.pyplot as plt


class ChainStructureAnalyze:
    def __init__(self, chain: Chain, train_data: InputData, test_data: InputData,
                 approaches: Optional[List[Type[NodeAnalyzeApproach]]] = None,
                 metric: str = None, nodes_ids_to_analyze: List[int] = None,
                 all_nodes: bool = False, path_to_save=None):

        if all_nodes and nodes_ids_to_analyze:
            raise ValueError("Choose only one parameter between all_nodes and nodes_ids_to_analyze")

        self.chain = chain
        self.train_data = train_data
        self.test_data = test_data
        self.approaches = approaches
        self.certain_nodes = nodes_ids_to_analyze
        self.all_nodes = all_nodes
        self.metric = metric

        if not path_to_save:
            self.path_to_save = join(default_fedot_data_dir(), 'sensitivity')
        else:
            self.path_to_save = path_to_save

    def analyze(self) -> dict:
        if self.all_nodes:
            nodes_ids_to_analyze = [i for i in range(len(self.chain.nodes))]
        elif self.certain_nodes:
            nodes_ids_to_analyze = self.certain_nodes
        else:
            raise ValueError("Define nodes to analyze: all_nodes or nodes_ids_to_analyze")

        nodes_results = dict()
        model_types = []
        for index in nodes_ids_to_analyze:
            node_result = NodeAnalysis(approaches=self.approaches, result_dir=self.path_to_save). \
                analyze(chain=self.chain, node_id=index,
                        train_data=self.train_data,
                        test_data=self.test_data)
            model_types.append(self.chain.nodes[index].model.model_type)

            nodes_results[f'id = {index}, model = {self.chain.nodes[index].model.model_type}'] = node_result

        self._visualize(nodes_results, model_types)
        return nodes_results

    def _visualize(self, results: dict, types: list):
        gathered_results = []
        for approach in self.approaches:
            approach_result = [result[f'{approach.__name__}'] - 1 for result in results.values()]
            gathered_results.append(approach_result)

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

            plt.savefig(join(self.path_to_save,
                             f'{self.approaches[index].__name__}.jpg'))
