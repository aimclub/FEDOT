from typing import List

from fedot.core.chains.chain import Chain
from fedot.core.data.data import InputData
from fedot.core.log import Log
from fedot.sensitivity.node_sensitivity import NodeAnalyzeApproach
from fedot.sensitivity.non_structure_sensitivity import ChainNonStructureAnalyze
from fedot.sensitivity.structure_sensitivity import ChainStructureAnalyze


class ChainSensitivityAnalysis:

    def __init__(self, chain: Chain, train_data: InputData, test_data: InputData,
                 approaches=None,
                 nodes_ids_to_analyze: List[int] = None, path_to_save=None,
                 log: Log = None):

        if approaches:
            structural_analyze_approaches = [approach for approach in approaches
                                             if type(approach) is NodeAnalyzeApproach]
            non_structural_approaches = [approach for approach in approaches
                                         if type(approach) is not NodeAnalyzeApproach]
        else:
            structural_analyze_approaches = None
            non_structural_approaches = None

        self._structure_analyze = ChainStructureAnalyze(chain=chain,
                                                        train_data=train_data,
                                                        test_data=test_data,
                                                        approaches=structural_analyze_approaches,
                                                        nodes_ids_to_analyze=nodes_ids_to_analyze,
                                                        path_to_save=path_to_save, log=log)

        self._non_structure_analyze = ChainNonStructureAnalyze(chain=chain,
                                                               train_data=train_data,
                                                               test_data=test_data,
                                                               approaches=non_structural_approaches,
                                                               path_to_save=path_to_save,
                                                               log=log)

    def analyze(self):
        if self._structure_analyze:
            self._structure_analyze.analyze()

        if self._non_structure_analyze:
            self._non_structure_analyze.analyze()
