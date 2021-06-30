from typing import List, Type, Union

from fedot.core.data.data import InputData
from fedot.core.log import Log, default_log
from fedot.core.pipelines.node import Node
from fedot.core.pipelines.pipeline import Pipeline
from fedot.sensitivity.node_sa_approaches import NodeAnalyzeApproach
from fedot.sensitivity.nodes_sensitivity import NodesAnalysis
from fedot.sensitivity.operations_hp_sensitivity.multi_operations_sensitivity \
    import MultiOperationsHPAnalyze
from fedot.sensitivity.pipeline_sensitivity import PipelineAnalysis
from fedot.sensitivity.sa_requirements import SensitivityAnalysisRequirements


class PipelineSensitivityAnalysis:
    """
    This class works as facade and allows to apply all kind of approaches
    to whole pipeline and separate nodes together.

    :param pipeline: pipeline object to analyze
    :param train_data: data used for Pipeline training
    :param test_data: data used for Pipeline validation
    :param approaches: methods applied to pipeline. Default: None
    :param nodes_to_analyze: nodes to analyze. Default: all nodes
    :param requirements: extra requirements to define specific details for different approaches.\
    See SensitivityAnalysisRequirements class documentation.
    :param path_to_save: path to save results to. Default: ~home/Fedot/sensitivity/
    Default: False
    :param log: log: Log object to record messages
    """

    def __init__(self, pipeline: Pipeline, train_data: InputData, test_data: InputData,
                 approaches: List[Union[Type[NodeAnalyzeApproach],
                                        Type[MultiOperationsHPAnalyze]]] = None,
                 nodes_to_analyze: List[Node] = None,
                 requirements: SensitivityAnalysisRequirements = None,
                 path_to_save=None,
                 log: Log = None):

        self.log = default_log(__name__) if log is None else log

        if approaches:
            nodes_analyze_approaches = [approach for approach in approaches
                                        if issubclass(approach, NodeAnalyzeApproach)]
            pipeline_analyze_approaches = [approach for approach in approaches
                                           if not issubclass(approach, NodeAnalyzeApproach)]
        else:
            self.log.message('Approaches for analysis are not given, thus will be set to defaults.')
            nodes_analyze_approaches = None
            pipeline_analyze_approaches = None

        self._nodes_analyze = NodesAnalysis(pipeline=pipeline,
                                            train_data=train_data,
                                            test_data=test_data,
                                            approaches=nodes_analyze_approaches,
                                            requirements=requirements,
                                            nodes_to_analyze=nodes_to_analyze,
                                            path_to_save=path_to_save, log=log)

        self._pipeline_analyze = PipelineAnalysis(pipeline=pipeline,
                                                  train_data=train_data,
                                                  test_data=test_data,
                                                  approaches=pipeline_analyze_approaches,
                                                  requirements=requirements,
                                                  path_to_save=path_to_save,
                                                  log=log)

    def analyze(self):
        """
        Applies defined sensitivity analysis approaches
        """
        if self._nodes_analyze:
            self._nodes_analyze.analyze()

        if self._pipeline_analyze:
            self._pipeline_analyze.analyze()
