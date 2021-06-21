from typing import List, Union, Type

from fedot.core.chains.chain import Chain
from fedot.core.chains.node import Node
from fedot.core.data.data import InputData
from fedot.core.log import Log, default_log
from fedot.sensitivity.node_sa_approaches import NodeAnalyzeApproach
from fedot.sensitivity.chain_sensitivity import ChainAnalysis
from fedot.sensitivity.operations_hp_sensitivity.multi_operations_sensitivity import MultiOperationsHPAnalyze
from fedot.sensitivity.nodes_sensitivity import NodesAnalysis
from fedot.sensitivity.sa_requirements import SensitivityAnalysisRequirements


class ChainSensitivityAnalysis:
    """
    This class works as facade and allows to apply all kind of approaches
    to whole chain and separate nodes together.

    :param chain: chain object to analyze
    :param train_data: data used for Chain training
    :param test_data: data used for Chain validation
    :param approaches: methods applied to chain. Default: None
    :param nodes_to_analyze: nodes to analyze. Default: all nodes
    :param requirements: extra requirements to define specific details for different approaches.\
    See SensitivityAnalysisRequirements class documentation.
    :param path_to_save: path to save results to. Default: ~home/Fedot/sensitivity/
    Default: False
    :param log: log: Log object to record messages
    """

    def __init__(self, chain: Chain, train_data: InputData, test_data: InputData,
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
            chain_analyze_approaches = [approach for approach in approaches
                                        if not issubclass(approach, NodeAnalyzeApproach)]
        else:
            self.log.message('Approaches for analysis are not given, thus will be set to defaults.')
            nodes_analyze_approaches = None
            chain_analyze_approaches = None

        self._nodes_analyze = NodesAnalysis(chain=chain,
                                            train_data=train_data,
                                            test_data=test_data,
                                            approaches=nodes_analyze_approaches,
                                            requirements=requirements,
                                            nodes_to_analyze=nodes_to_analyze,
                                            path_to_save=path_to_save, log=log)

        self._chain_analyze = ChainAnalysis(chain=chain,
                                            train_data=train_data,
                                            test_data=test_data,
                                            approaches=chain_analyze_approaches,
                                            requirements=requirements,
                                            path_to_save=path_to_save,
                                            log=log)

    def analyze(self):
        """
        Applies defined sensitivity analysis approaches
        """
        if self._nodes_analyze:
            self._nodes_analyze.analyze()

        if self._chain_analyze:
            self._chain_analyze.analyze()
