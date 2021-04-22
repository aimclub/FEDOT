import json
import random
from abc import ABC, abstractmethod
from copy import deepcopy
from datetime import timedelta
from os import makedirs
from os.path import exists, join
from typing import List, Optional, Type, Union

import matplotlib.pyplot as plt
import numpy as np

from fedot.core.chains.chain import Chain
from fedot.core.chains.chain_tune import Tune
from fedot.core.chains.node import Node, PrimaryNode, SecondaryNode
from fedot.core.data.data import InputData
from fedot.core.log import Log, default_log
from fedot.core.repository.model_types_repository import ModelTypesRepository
from fedot.core.utils import default_fedot_data_dir
from fedot.utilities.define_metric_by_task import MetricByTask


class NodeAnalysis:
    """
    :param approaches: methods applied to nodes to modify the chain or analyze certain models.\
    Default: [NodeDeletionAnalyze, NodeTuneAnalyze, NodeReplaceModelAnalyze]
    :param path_to_save: path to save results to. Default: ~home/Fedot/sensitivity
    :param interactive_mode: flag for interactive visualization or saving plots to file. Default: False
    :param log: log: Log object to record messages
    """

    def __init__(self, approaches: Optional[List[Type['NodeAnalyzeApproach']]] = None,
                 path_to_save=None, log: Log = None):


        if not approaches:
            self.approaches = [NodeDeletionAnalyze, NodeTuneAnalyze, NodeReplaceModelAnalyze]
        else:
            self.approaches = approaches

        if not path_to_save:
            self.path_to_save = join(default_fedot_data_dir(), 'sensitivity')
        else:
            self.path_to_save = path_to_save

        if not log:
            self.log = default_log(__name__)
        else:
            self.log = log

    def analyze(self, chain: Chain, node_id: int,
                train_data: InputData, test_data: InputData, is_save=True) -> dict:

        """
        Method runs Node analysis within defined approaches

        :param chain: Chain containing the analyzed Node
        :param node_id: node index in Chain
        :param train_data: data used for Chain training
        :param test_data: data used for Chain validation
        :param is_save: flag to save results to json or not
        :return: dict with Node analysis result per approach
        """

        results = dict()
        for approach in self.approaches:
            results[f'{approach.__name__}'] = \
                approach(chain=chain,
                         train_data=train_data,
                         test_data=test_data,
                         path_to_save=self.path_to_save).analyze(node_id=node_id)

        if is_save:
            self._save_results_to_json(results)

        return results

    def _save_results_to_json(self, results):
        result_file = join(self.path_to_save, 'node_sa_results.json')
        with open(result_file, 'w', encoding='utf-8') as file:
            file.write(json.dumps(results, indent=4))

        self.log.message(f'Node Sensitivity analysis results were saved to {result_file}')


class NodeAnalyzeApproach(ABC):
    """
    Base class for analysis approach.

    :param chain: Chain containing the analyzed Node
    :param train_data: data used for Chain training
    :param test_data: data used for Chain validation
    :param path_to_save: path to save results to. Default: ~home/Fedot/sensitivity
    :param interactive_mode: flag for interactive visualization or saving plots to file. Default: False
    :param log: log: Log object to record messages
    """

    def __init__(self, chain: Chain, train_data, test_data: InputData,
                 path_to_save=None, log: Log = None):
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

        if not log:
            self.log = default_log(__name__)
        else:
            self.log = log

    @abstractmethod
    def analyze(self, node_id) -> Union[List[dict], float]:
        """Creates the difference metric(scorer, index, etc) of the changed
        chain in relation to the original one

        :param node_id: the sequence number of the node as in DFS result
        """
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
    def __init__(self, chain: Chain, train_data, test_data: InputData,
                 path_to_save=None):
        super().__init__(chain, train_data, test_data,
                         path_to_save)

    def analyze(self, node_id: int) -> Union[List[dict], float]:
        """
        :param node_id: the sequence number of the node as in DFS result
        :return: the ratio of modified chain score to origin score
        """
        if node_id == 0:
            # TODO or warning?
            return 1.0
        else:
            shortend_chain = self.sample(node_id)
            loss = self._compare_with_origin_by_metric(shortend_chain)
            del shortend_chain

            return loss

    def sample(self, node_id: int):
        """

        :param node_id: the sequence number of the node as in DFS result
        :return: Chain object without node with defined node_id
        """
        chain_sample = deepcopy(self._chain)
        node_to_delete = chain_sample.nodes[node_id]
        chain_sample.delete_node(node_to_delete)

        return chain_sample

    def __str__(self):
        return 'NodeDeletionAnalyze'


class NodeTuneAnalyze(NodeAnalyzeApproach):
    """
    Tune node and evaluate the score difference
    """

    def __init__(self, chain: Chain, train_data, test_data: InputData,
                 path_to_save=None):
        super().__init__(chain, train_data, test_data,
                         path_to_save)

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
    """
    Replace node with models available for the current task
    and evaluate the score difference
    """

    def __init__(self, chain: Chain, train_data, test_data: InputData,
                 path_to_save=None):
        super().__init__(chain, train_data, test_data,
                         path_to_save)

    def analyze(self, node_id: int,
                nodes_to_replace_to: Optional[List[Node]] = None,
                number_of_random_models: Optional[int] = None) -> Union[List[dict], float]:
        """

        :param node_id:the sequence number of the node as in DFS result
        :param nodes_to_replace_to: nodes provided for old_node replacement
        :param number_of_random_models: number of replacement operations, \
        if nodes_to_replace_to not provided
        :return: the ratio of modified chain score to origin score
        """

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
               number_of_random_models: Optional[int] = None) -> Union[List[Chain], Chain]:
        """

        :param node_id:the sequence number of the node as in DFS result
        :param nodes_to_replace_to: nodes provided for old_node replacement
        :param number_of_random_models: number of replacement operations, \
        if nodes_to_replace_to not provided
        :return: Sequence of Chain objects with new models instead of old one.
        """

        if not nodes_to_replace_to:
            node_type = type(self._chain.nodes[node_id])
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
                         number_of_models=None) -> List[Node]:
        available_models, _ = ModelTypesRepository().suitable_model(task_type=self._train_data.task.task_type)
        if number_of_models:
            random_models = random.sample(available_models, number_of_models)
        else:
            random_models = available_models

        nodes = []
        for model in random_models:
            nodes.append(node_type(model_type=model))

        return nodes

    def _visualize(self, x_values, y_values: list, node_id: int):
        data = zip(x_values, [(y - 1) for y in y_values])  # 3
        original_model_type = self._chain.nodes[node_id].model.model_type

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

        if original_model_type in x_values:
            original_model_index = x_values.index(original_model_type)
            plt.gca().get_xticklabels()[original_model_index].set_color('red')

        file_name = f'{self._chain.nodes[node_id].model.model_type}_id_{node_id}_replacement.jpg'
        result_file = join(self._path_to_save, file_name)
        plt.savefig(result_file)
        self.log.message(f'NodeReplacementAnalysis for '
                         f'{original_model_type}(index:{node_id}) was saved to {result_file}')

    def __str__(self):
        return 'NodeReplaceModelAnalyze'
