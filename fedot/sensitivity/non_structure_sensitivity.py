from os.path import join
from typing import List, Type, Optional

from fedot.core.chains.chain import Chain
from fedot.core.data.data import InputData
from fedot.core.log import Log, default_log
from fedot.core.utils import default_fedot_data_dir
from fedot.sensitivity.operations_sensitivity.multi_operations_sensitivity import MultiOperationsAnalyze


class ChainNonStructureAnalyze:
    def __init__(self, chain: Chain, train_data: InputData, test_data: InputData,
                 approaches: Optional[List[Type[MultiOperationsAnalyze]]] = None,
                 path_to_save=None, log: Log = None):
        self.chain = chain
        self.train_data = train_data
        self.test_data = test_data

        if not approaches:
            self.approaches = [MultiOperationsAnalyze]
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

    def analyze(self):
        all_approaches_results = []
        for approach in self.approaches:
            analyze_result = approach(chain=self.chain,
                                      train_data=self.train_data,
                                      test_data=self.test_data).analyze()
            all_approaches_results.append(analyze_result)

        return all_approaches_results
