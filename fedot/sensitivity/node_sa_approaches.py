import json
import random
from abc import ABC, abstractmethod
from copy import deepcopy
from os import makedirs
from os.path import exists, join
from typing import List, Optional, Type, Union, Callable

import matplotlib.pyplot as plt

from fedot.core.data.data import InputData
from fedot.core.log import default_log
from fedot.core.pipelines.node import Node
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.verification import verify_pipeline
from fedot.core.repository.operation_types_repository import OperationTypesRepository
from fedot.core.repository.quality_metrics_repository import MetricsRepository
from fedot.core.utils import default_fedot_data_dir
from fedot.sensitivity.sa_requirements import ReplacementAnalysisMetaParams, SensitivityAnalysisRequirements
from fedot.utilities.define_metric_by_task import MetricByTask


class NodeAnalysis:
    """
    Args:
        approaches: methods applied to nodes to modify the pipeline or analyze certain operations.
            Default: [``NodeDeletionAnalyze``, ``NodeTuneAnalyze``, ``NodeReplaceOperationAnalyze``]
        path_to_save: path to save results to. Default: ``~home/Fedot/sensitivity``
    """

    def __init__(self, approaches: Optional[List[Type['NodeAnalyzeApproach']]] = None,
                 approaches_requirements: SensitivityAnalysisRequirements = None,
                 path_to_save=None):

        self.approaches = [NodeDeletionAnalyze, NodeReplaceOperationAnalyze] if approaches is None else approaches

        self.path_to_save = \
            join(default_fedot_data_dir(), 'sensitivity', 'nodes_sensitivity') if path_to_save is None else path_to_save
        self.log = default_log(self)

        self.approaches_requirements = \
            SensitivityAnalysisRequirements() if approaches_requirements is None else approaches_requirements

    def analyze(self, pipeline: Pipeline, node: Node,
                train_data: InputData, test_data: InputData,
                is_save: bool = False) -> dict:

        """Method runs Node analysis within defined approaches

        Args:
            is_save: whether the certain node analysis result is needed to ba saved
            pipeline: :obj:`Pipeline` containing the analyzed :obj:`Node`
            node: :obj:`Node` object to analyze in :obj:`Pipeline`
            train_data: data used for :obj:`Pipeline` training
            test_data: data used for :obj:`Pipeline` validation

        Returns:
            dict:  :obj:`Node` analysis result per approach
        """

        results = dict()
        for approach in self.approaches:
            results[f'{approach.__name__}'] = \
                approach(pipeline=pipeline,
                         train_data=train_data,
                         test_data=test_data,
                         requirements=self.approaches_requirements,
                         path_to_save=self.path_to_save).analyze(node=node)

        # TODO remove conflict with requirements.is_save
        if is_save:
            self._save_results_to_json(node, pipeline, results)

        node_sa_index = self._get_node_index(train_data, results)
        if node_sa_index is not None:
            node.rating = self._get_node_rating(node_sa_index)

        return results

    @staticmethod
    def _get_node_index(train_data: InputData, results: dict):
        total_index = None
        if NodeReplaceOperationAnalyze.__name__ in results.keys() and NodeDeletionAnalyze.__name__ in results.keys():
            task = train_data.task.task_type
            app_models, _ = OperationTypesRepository().suitable_operation(task_type=task)
            total_operations_number = len(app_models)

            replacement_candidates = results[NodeReplaceOperationAnalyze.__name__]
            candidates_for_replacement_number = len(
                [candidate for candidate in replacement_candidates if (1 - candidate) < 0])

            replacement_score = candidates_for_replacement_number / total_operations_number

            deletion_score = results[NodeDeletionAnalyze.__name__][0]

            total_index = (deletion_score / abs(deletion_score)) * replacement_score

        return total_index

    @staticmethod
    def _get_node_rating(total_index: float):
        rating = None
        if total_index <= -0.5:
            rating = 2
        elif -0.5 < total_index <= 0.0:
            rating = 1
        elif 0.0 < total_index <= 0.5:
            rating = 4
        elif 0.5 < total_index <= 1:
            rating = 3

        return rating

    def _save_results_to_json(self, node: Node, pipeline: Pipeline, results):
        node_id = pipeline.nodes.index(node)
        node_type = node.operation.operation_type
        result_file = join(self.path_to_save, f'{node_id}{node_type}_sa_results.json')
        with open(result_file, 'w', encoding='utf-8') as file:
            file.write(json.dumps(results, indent=4))

        self.log.info(f'Node Sensitivity analysis results were saved to {result_file}')


class NodeAnalyzeApproach(ABC):
    """Base class for analysis approach.

    Args:
        pipeline: :obj:`Pipeline` containing the analyzed :obj:`Node`
        train_data: data used for :obj:`Pipeline` training
        test_data: data used for :obj:`Pipeline` validation
        path_to_save: path to save results to. Default: ``~home/Fedot/sensitivity``
    """

    def __init__(self, pipeline: Pipeline, train_data, test_data: InputData,
                 requirements: SensitivityAnalysisRequirements = None,
                 path_to_save=None):
        self._pipeline = pipeline
        self._train_data = train_data
        self._test_data = test_data
        self._origin_metric = None
        self._requirements = \
            SensitivityAnalysisRequirements() if requirements is None else requirements

        self._path_to_save = \
            join(default_fedot_data_dir(), 'sensitivity', 'nodes_sensitivity') if path_to_save is None else path_to_save
        self.log = default_log(self)

        if not exists(self._path_to_save):
            makedirs(self._path_to_save)

    @abstractmethod
    def analyze(self, node: Node, **kwargs) -> Union[List[dict], List[float]]:
        """Creates the difference metric(scorer, index, etc) of the changed
        graph in relation to the original one

        Args:
            node: the sequence number of the node as in ``DFS`` result
        """
        pass

    @abstractmethod
    def sample(self, *args) -> Union[List[Pipeline], Pipeline]:
        """Changes the pipeline according to the approach"""
        pass

    def _compare_with_origin_by_metric(self, changed_pipeline: Pipeline) -> float:
        metric_id = MetricByTask.get_default_quality_metrics(self._train_data.task.task_type)[0]
        metric = MetricsRepository.metric_by_id(metric_id)

        if not self._origin_metric:
            self._origin_metric = self._get_metric_value(pipeline=self._pipeline, metric=metric)

        changed_pipeline_metric = self._get_metric_value(pipeline=changed_pipeline, metric=metric)

        return changed_pipeline_metric / self._origin_metric

    def _get_metric_value(self, pipeline: Pipeline, metric: Callable) -> float:
        pipeline.fit(self._train_data)
        predicted = pipeline.predict(self._test_data)
        metric_value = metric(true=self._test_data,
                              predicted=predicted)

        return metric_value


class NodeDeletionAnalyze(NodeAnalyzeApproach):
    def __init__(self, pipeline: Pipeline, train_data: InputData, test_data: InputData,
                 requirements: SensitivityAnalysisRequirements = None, path_to_save=None):
        super().__init__(pipeline, train_data, test_data, requirements,
                         path_to_save)

    def analyze(self, node: Node, **kwargs) -> Union[List[dict], List[float]]:
        """
        Args:
            node: :obj:`Node` object to analyze

        Returns:
            the ratio of modified pipeline score to origin score
        """

        if node is self._pipeline.root_node:
            # TODO or warning?
            return [1.0]
        else:
            shortened_pipeline = self.sample(node)
            if shortened_pipeline:
                loss = self._compare_with_origin_by_metric(shortened_pipeline)
                del shortened_pipeline
            else:
                loss = 1

            return [loss]

    def sample(self, node: Node):
        """
        Args:
            node: :obj:`Node` object to delete from :obj:`Pipeline` object
        Retuens:
            :obj:`Pipeline`: pipeline without node
        """

        pipeline_sample = deepcopy(self._pipeline)
        node_index_to_delete = self._pipeline.nodes.index(node)
        node_to_delete = pipeline_sample.nodes[node_index_to_delete]
        pipeline_sample.delete_node(node_to_delete)
        try:
            verify_pipeline(pipeline_sample)
        except ValueError as ex:
            self.log.info(f'Can not delete node. Deletion of this node leads to {ex}')
            return None

        return pipeline_sample

    def __str__(self):
        return 'NodeDeletionAnalyze'


class NodeReplaceOperationAnalyze(NodeAnalyzeApproach):
    """Replace node with operations available for the current task and evaluate the score difference
    """

    def __init__(self, pipeline: Pipeline, train_data: InputData, test_data: InputData,
                 requirements: SensitivityAnalysisRequirements = None, path_to_save=None):
        super().__init__(pipeline, train_data, test_data, requirements,
                         path_to_save)

    def analyze(self, node: Node, **kwargs) -> Union[List[dict], List[float]]:
        """
        Args:
            node: :obj:`Node` object to analyze

        Returns:
            the ratio of modified pipeline score to origin score
        """

        requirements: ReplacementAnalysisMetaParams = self._requirements.replacement_meta
        node_id = self._pipeline.nodes.index(node)
        samples = self.sample(node=node,
                              nodes_to_replace_to=requirements.nodes_to_replace_to,
                              number_of_random_operations=requirements.number_of_random_operations)

        loss_values = []
        new_nodes_types = []
        for sample_pipeline in samples:
            loss_per_sample = self._compare_with_origin_by_metric(sample_pipeline)
            loss_values.append(loss_per_sample)

            new_node = sample_pipeline.nodes[node_id]
            new_nodes_types.append(new_node.operation.operation_type)

        if self._requirements.visualization:
            self._visualize(x_values=new_nodes_types,
                            y_values=loss_values,
                            node_id=node_id)

        return loss_values

    def sample(self, node: Node,
               nodes_to_replace_to: Optional[List[Node]],
               number_of_random_operations: Optional[int] = None) -> Union[List[Pipeline], Pipeline]:
        """
        Args:
            node: :obj:`Node` object to replace
            nodes_to_replace_to: nodes provided for old_node replacement
            number_of_random_operations: number of replacement operations,
                if ``nodes_to_replace_to`` not provided
        Returns:
            Union[List[Pipeline], Pipeline]: sequence of :obj:`Pipeline` objects with new operations instead of old one
        """

        if not nodes_to_replace_to:
            node_type = type(node)
            nodes_to_replace_to = self._node_generation(node_type=node_type,
                                                        number_of_operations=number_of_random_operations)

        samples = list()
        for replacing_node in nodes_to_replace_to:
            sample_pipeline = deepcopy(self._pipeline)
            replaced_node_index = self._pipeline.nodes.index(node)
            replaced_node = sample_pipeline.nodes[replaced_node_index]
            sample_pipeline.update_node(old_node=replaced_node,
                                        new_node=replacing_node)
            samples.append(sample_pipeline)

        return samples

    def _node_generation(self, node_type: Type[Node],
                         number_of_operations=None) -> List[Node]:
        task = self._train_data.task.task_type
        # Get models
        app_models, _ = OperationTypesRepository().suitable_operation(task_type=task)
        # Get data operations for such task
        app_data_operations = OperationTypesRepository('data_operation').suitable_operation(
            task_type=task)

        # Unit two lists
        app_operations = app_models
        if number_of_operations:
            random_operations = random.sample(app_operations, number_of_operations)
        else:
            random_operations = app_operations

        nodes = []
        for operation in random_operations:
            nodes.append(node_type(operation_type=operation))

        return nodes

    def _visualize(self, x_values, y_values: list, node_id: int):
        data = zip(x_values, [(y - 1) for y in y_values])  # 3
        original_operation_type = self._pipeline.nodes[node_id].operation.operation_type

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
        plt.ylabel('quality (changed_pipeline_metric/original_metric) - 1')

        if original_operation_type in x_values:
            original_operation_index = x_values.index(original_operation_type)
            plt.gca().get_xticklabels()[original_operation_index].set_color('red')

        file_name = f'{self._pipeline.nodes[node_id].operation.operation_type}_id_{node_id}_replacement.jpg'
        result_file = join(self._path_to_save, file_name)
        plt.savefig(result_file)
        self.log.info(f'NodeReplacementAnalysis for '
                         f'{original_operation_type}(index:{node_id}) was saved to {result_file}')

    def __str__(self):
        return 'NodeReplaceOperationAnalyze'
