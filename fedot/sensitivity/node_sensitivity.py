from abc import ABC
from abc import abstractmethod
from copy import deepcopy
from datetime import timedelta
from os import makedirs
from os.path import join, exists
from typing import Optional, List, Union, Type

import matplotlib.pyplot as plt
import numpy as np

from fedot.core.chains.chain import Chain
from fedot.core.chains.chain_tune import Tune
from fedot.core.chains.node import Node, PrimaryNode, SecondaryNode
from fedot.core.data.data import InputData
from fedot.core.repository.model_types_repository import ModelTypesRepository
from fedot.core.utils import default_fedot_data_dir
from fedot.utilities.define_metric_by_task import MetricByTask


class NodeAnalysis:
    def __init__(self, approaches: Optional[List[Type['NodeAnalyzeApproach']]] = None, result_dir=None):
        if not approaches:
            self.approaches = [NodeDeletionAnalyze, NodeTuneAnalyze, NodeReplaceModelAnalyze]
        else:
            self.approaches = approaches

        if not result_dir:
            self.result_dir = join(default_fedot_data_dir(), 'sensitivity')
        else:
            self.result_dir = result_dir

    def analyze(self, chain: Chain, node_id: int,
                train_data: InputData, test_data: InputData) -> dict:

        results = dict()
        for approach in self.approaches:
            results[f'{approach.__name__}'] = approach(chain=chain,
                                                       train_data=train_data,
                                                       test_data=test_data,
                                                       path_to_save=self.result_dir).analyze(node_id=node_id)

        return results


class NodeAnalyzeApproach(ABC):
    def __init__(self, chain: Chain, train_data, test_data: InputData, path_to_save=None):
        self._chain = chain
        self._train_data = train_data
        self._test_data = test_data
        self._origin_metric = None

        if not path_to_save:
            self._path_to_save = join(default_fedot_data_dir(), 'sensitivity')
        else:
            self._path_to_save = path_to_save

        if not exists(self._path_to_save):
            makedirs(self._path_to_save)

    @abstractmethod
    def analyze(self, node_id) -> Union[List[dict], float]:
        """Creates the difference metric(scorer, index, etc) of the changed
        chain in relation to the original one"""
        pass

    @abstractmethod
    def sample(self, *args) -> Union[Union[List[Chain], Chain], 'ndarray']:
        """Changes the chain according to the approach"""
        pass

    def _compare_with_origin_by_metric(self, changed_chain: Chain) -> float:
        metric = MetricByTask(self._train_data.task.task_type)

        if not self._origin_metric:
            self._origin_metric = self._get_metric_value(chain=self._chain, metric=metric)

        changed_chain_metric = self._get_metric_value(chain=changed_chain, metric=metric)

        return changed_chain_metric / self._origin_metric

    def _get_metric_value(self, chain: Chain, metric: MetricByTask) -> float:
        chain.fit(self._train_data, use_cache=False)
        predicted = chain.predict(self._test_data)
        metric_value = metric.get_value(true=self._test_data,
                                        predicted=predicted)

        return metric_value


class NodeDeletionAnalyze(NodeAnalyzeApproach):
    def __init__(self, chain: Chain, train_data, test_data: InputData, path_to_save):
        super(NodeDeletionAnalyze, self).__init__(chain, train_data, test_data, path_to_save)

    def analyze(self, node_id: int) -> Union[List[dict], float]:
        if node_id == 0:
            # TODO or warning?
            return 0.0
        else:
            shortend_chain = self.sample(node_id)
            loss = self._compare_with_origin_by_metric(shortend_chain)

            return loss

    def sample(self, node_id: int):
        chain_sample = deepcopy(self._chain)
        node_to_delete = chain_sample.nodes[node_id]
        chain_sample.delete_node_with_redirection(node_to_delete)

        return chain_sample

    def __str__(self):
        return 'NodeDeletionAnalyze'


class NodeTuneAnalyze(NodeAnalyzeApproach):
    def __init__(self, chain: Chain, train_data, test_data: InputData, path_to_save):
        super(NodeTuneAnalyze, self).__init__(chain, train_data, test_data, path_to_save)

    def analyze(self, node_id: int) -> Union[List[dict], float]:
        tuned_chain = Tune(self._chain).fine_tune_certain_node(model_id=node_id,
                                                               input_data=self._train_data,
                                                               max_lead_time=timedelta(minutes=1),
                                                               iterations=30)
        loss = self._compare_with_origin_by_metric(tuned_chain)

        return loss

    def sample(self, *args) -> Union[List[Chain], Chain]:
        raise NotImplemented

    def __str__(self):
        return 'NodeTuneAnalyze'


class NodeReplaceModelAnalyze(NodeAnalyzeApproach):
    def __init__(self, chain: Chain, train_data, test_data: InputData, path_to_save):
        super(NodeReplaceModelAnalyze, self).__init__(chain, train_data, test_data, path_to_save)

    def analyze(self, node_id: int,
                nodes_to_replace_to: Optional[List[Node]] = None,
                number_of_random_models: Optional[int] = 5) -> Union[List[dict], float]:

        samples = self.sample(node_id=node_id,
                              nodes_to_replace_to=nodes_to_replace_to,
                              number_of_random_models=number_of_random_models)

        loss_values = []
        new_nodes_types = []
        for sample_chain in samples:
            loss_per_sample = self._compare_with_origin_by_metric(sample_chain)
            loss_values.append(loss_per_sample)

            new_node = sample_chain.nodes[node_id]
            new_nodes_types.append(new_node.model.model_type)

        avg_loss = np.mean(loss_values)
        self._visualize(x_values=new_nodes_types,
                        y_values=loss_values,
                        node_id=node_id)

        return float(avg_loss)

    def sample(self, node_id: int,
               nodes_to_replace_to: Optional[List[Node]],
               number_of_random_models: Optional[int]) -> Union[List[Chain], Chain]:

        # TODO Refactor according to future different types of Nodes
        if not nodes_to_replace_to:
            if isinstance(self._chain.nodes[node_id], PrimaryNode):
                node_type = PrimaryNode
            elif isinstance(self._chain.nodes[node_id], SecondaryNode):
                node_type = SecondaryNode
            else:
                raise ValueError('Unsupported type of Node. Expected Primary or Secondary')

            nodes_to_replace_to = self._node_generation(node_type=node_type,
                                                        number_of_models=number_of_random_models)

        samples = list()
        for replacing_node in nodes_to_replace_to:
            sample_chain = deepcopy(self._chain)
            replaced_node = sample_chain.nodes[node_id]
            sample_chain.update_node(old_node=replaced_node,
                                     new_node=replacing_node)
            samples.append(sample_chain)

        return samples

    def _node_generation(self, node_type: Union[Type[PrimaryNode], Type[SecondaryNode]],
                         number_of_models: int = 5) -> List[Node]:
        available_models, _ = ModelTypesRepository().suitable_model(task_type=self._train_data.task.task_type)
        # random_models = random.sample(available_models, number_of_models)
        random_models = available_models

        nodes = []
        for model in random_models:
            nodes.append(node_type(model_type=model))

        return nodes

    def _visualize(self, x_values, y_values: list, node_id: int):
        data = zip(x_values, [(y - 1) for y in y_values])  # 3
        model_type = self._chain.nodes[node_id].model.model_type

        sorted_data = sorted(data, key=lambda tup: tup[1])
        x_values = [x[0] for x in sorted_data]
        y_values = [y[1] for y in sorted_data]
        colors = ['r' if y < 0 else 'g' for y in y_values]
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.bar(x_values, y_values, width=1.0, edgecolor='black', bottom=1,
               color=colors)
        ax.hlines(1, -1, len(x_values) + 1, linestyle='--')
        ax.set_xticklabels(x_values, rotation=45)
        plt.xlabel('iteration')
        plt.ylabel('quality (changed_chain_metric/original_metric) - 1')
        original_model_index = x_values.index(model_type)
        plt.gca().get_xticklabels()[original_model_index].set_color('red')

        plt.savefig(join(self._path_to_save,
                         f'{self._chain.nodes[node_id].model.model_type}_id_{node_id}_replacement.jpg'))

    def __str__(self):
        return 'NodeReplaceModelAnalyze'
