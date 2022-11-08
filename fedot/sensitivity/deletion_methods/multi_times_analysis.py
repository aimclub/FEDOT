from collections import namedtuple
from os import makedirs
from os.path import exists, join
from typing import List, Optional, Type

from fedot.core.data.data import InputData
from fedot.core.log import default_log
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.quality_metrics_repository import MetricsRepository
from fedot.core.utils import default_fedot_data_dir
from fedot.sensitivity.node_sa_approaches import NodeAnalyzeApproach, NodeDeletionAnalyze
from fedot.sensitivity.nodes_sensitivity import NodesAnalysis
from fedot.utilities.define_metric_by_task import MetricByTask

MTAMetaParams = namedtuple('MTAMetaParams', ['delta', 'worst_node_score'])


class MultiTimesAnalyze:
    """
    Multi-Times-Analyze approach is used for Pipeline size decrease
    using node-deletion-algorithm defined in MultiTimesAnalyze.analyze method

    :param pipeline: pipeline object to analyze
    :param train_data: data used for Pipeline training
    :param test_data: data used for getting prediction
    :param valid_data: used for modification validation
    :param case_name: used for uniq result directory name
    :param approaches: methods applied to nodes to modify
        the pipeline or analyze certain operations.\
    Defaults: NodeDeletionAnalyze.
    """

    default_mta_meta_params = MTAMetaParams(10e-3, 1.1)

    def __init__(self, pipeline: Pipeline, train_data: InputData,
                 test_data: InputData, valid_data: InputData,
                 case_name: str, path_to_save: str = None,
                 approaches: Optional[List[Type[NodeAnalyzeApproach]]] = None):
        self.pipeline = pipeline
        self.original_pipeline_len = self.pipeline.length
        self.train_data = train_data
        self.test_data = test_data
        self.valid_data = valid_data
        self.case_name = case_name
        self.path_to_save = \
            join(default_fedot_data_dir(),
                 'sensitivity', 'mta_analysis', f'{case_name}') \
                if path_to_save is None else path_to_save
        self.approaches = [NodeDeletionAnalyze] if approaches is None else approaches
        self.log = default_log(self)

    def analyze(self, visualization=False, meta_params: MTAMetaParams = None) -> float:
        """
        Algorithm:
        1. Analyze pipeline
        2. Defines potential 'bad' nodes within iteration
        3. Choose the worst one and delete it
        3. Repeat 1-3 till the condition: no more 'bad' nodes(worst_node_score<=1) or len(Pipeline) < 2

        :param meta_params: limiting params for sensitivity index:
         MTAMetaParams(delta, worst_node_score)
         (defaults: 10e-3, 1.1 correspondingly).
        :param visualization: boolean flag for pipeline structure visualization. Default: False

        :return ratio of number of deleted nodes to overall Pipeline length
        """

        if not meta_params:
            meta_params = self.default_mta_meta_params

        total_nodes_deleted = 0
        iteration_index = 1
        worst_node_score = meta_params.worst_node_score
        while worst_node_score > 1.0 + meta_params.delta and self.pipeline.length > 2:
            self.log.info('new iteration of MTA deletion analysis')
            iteration_result_path = join(self.path_to_save, f'iter_{iteration_index}')
            pipeline_analysis_result = self._pipeline_analysis(result_path=iteration_result_path,
                                                               visualization=visualization)

            deletion_scores = [node['NodeDeletionAnalyze'] for node in pipeline_analysis_result.values()]
            worst_node_score = max(deletion_scores)

            if worst_node_score > 1.0 + meta_params.delta:
                worst_node_index = deletion_scores.index(worst_node_score)
                self.pipeline.delete_node(self.pipeline.nodes[worst_node_index])
                total_nodes_deleted += 1

            iteration_index += 1

        self.log.info('finish MTA')
        return self._length_reduction_ratio(total_nodes_deleted)

    def _length_reduction_ratio(self, number_of_deleted_nodes: int):
        return number_of_deleted_nodes / self.original_pipeline_len

    def _pipeline_analysis(self, result_path, visualization=False):
        if not exists(result_path):
            makedirs(result_path)

        if visualization:
            self._visualize(name=self.case_name, path=result_path)

        self.pipeline.fit_from_scratch(self.train_data)

        self.log.info('Start Pipeline Analysis')

        pipeline_analysis_result = NodesAnalysis(pipeline=self.pipeline, train_data=self.train_data,
                                                 test_data=self.test_data,
                                                 path_to_save=result_path,
                                                 approaches=self.approaches).analyze()
        self.log.info("End Pipeline Analysis")

        return pipeline_analysis_result

    def _visualize(self, name, path):
        image_path = join(path, f'{name}.png')
        self.pipeline.show(save_path=image_path)

    def get_metric(self):
        self.pipeline.fit(self.train_data)
        predicted = self.pipeline.predict(self.valid_data)
        metric_value = MetricByTask.compute_default_metric(
            self.valid_data.task.task_type,
            true=self.valid_data,
            predicted=predicted
        )

        return metric_value
