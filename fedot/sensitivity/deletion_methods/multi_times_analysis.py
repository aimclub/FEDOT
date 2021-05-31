from os import makedirs
from os.path import join, exists
from typing import Optional, List, Type

from fedot.core.chains.chain import Chain
from fedot.core.composer.visualisation import ChainVisualiser
from fedot.core.data.data import InputData
from fedot.sensitivity.node_sensitivity import NodeAnalyzeApproach, NodeDeletionAnalyze
from fedot.sensitivity.structure_sensitivity import ChainStructureAnalyze
from fedot.utilities.define_metric_by_task import MetricByTask


class MultiTimesAnalyze:
    """
    Multi-Times-Analyze approach is used for Chain size decrease
    using node-deletion-algorithm defined in MultiTimesAnalyze.analyze method
    """

    def __init__(self, chain: Chain, train_data: InputData,
                 test_data: InputData, valid_data: InputData,
                 case_name: str, path_to_save: str,
                 approaches: Optional[List[Type[NodeAnalyzeApproach]]] = None):
        self.chain = chain
        self.train_data = train_data
        self.test_data = test_data
        self.valid_data = valid_data
        if not approaches:
            self.approaches = [NodeDeletionAnalyze]
        else:
            self.approaches = approaches
        self.case_name = case_name
        self.path_to_save = path_to_save

    def analyze(self) -> float:
        """
        Algorithm:
        1. Analyze chain
        2. Defines potential 'bad' nodes within iteration
        3. Choose the worst one and delete it
        3. Repeat 1-3 till the condition: no more 'bad' nodes(worst_node_score<=1) or len(Chain) < 2
        """

        delta = 10e-3
        worst_node_score = 1.1
        original_chain_len = self.chain.length
        total_nodes_deleted = 0
        iteration_index = 1
        while worst_node_score > 1.0 + delta and len(self.chain.nodes) > 2:
            print('new iteration of  MTA deletion analysis')
            iteration_result_path = join(self.path_to_save, f'iter_{iteration_index}')
            chain_analysis_result = self.chain_analysis(result_path=iteration_result_path)

            deletion_scores = [node['NodeDeletionAnalyze'] for node in chain_analysis_result.values()]
            worst_node_score = max(deletion_scores)

            if worst_node_score > 1.0 + delta:
                worst_node_index = deletion_scores.index(worst_node_score)
                self.chain.delete_node(self.chain.nodes[worst_node_index])
                total_nodes_deleted += 1

            iteration_index += 1

        print('finish MTA')
        return total_nodes_deleted / original_chain_len

    def chain_analysis(self, result_path):
        if not exists(result_path):
            makedirs(result_path)

        self._visualize(name=self.case_name, path=result_path)

        self.chain.fit_from_scratch(self.train_data)

        print('Start Chain Analysis')

        chain_analysis_result = ChainStructureAnalyze(chain=self.chain, train_data=self.train_data,
                                                      test_data=self.test_data,
                                                      all_nodes=True,
                                                      path_to_save=result_path,
                                                      approaches=self.approaches).analyze()
        print("End Chain Analysis")

        return chain_analysis_result

    def _visualize(self, name, path):
        visualiser = ChainVisualiser()
        image_path = join(path, f'{name}.png')
        visualiser.visualise(self.chain, save_path=image_path)

    def get_metric(self):
        self.chain.fit(self.train_data, use_cache=False)
        metric = MetricByTask(self.valid_data.task.task_type)
        predicted = self.chain.predict(self.valid_data)
        metric_value = metric.get_value(true=self.valid_data,
                                        predicted=predicted)

        return metric_value
