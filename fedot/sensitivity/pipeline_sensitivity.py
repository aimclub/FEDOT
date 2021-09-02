import json
from os.path import join
from typing import List, Type, Optional

from fedot.core.data.data import InputData
from fedot.core.log import Log, default_log
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.utils import default_fedot_data_dir
from fedot.sensitivity.operations_hp_sensitivity.multi_operations_sensitivity import MultiOperationsHPAnalyze
from fedot.sensitivity.sa_requirements import SensitivityAnalysisRequirements


class PipelineAnalysis:
    """
    This class is for analyzing the Pipeline as the black-box model,
    using analysis approaches defined for whole pipeline perturbation,
    i.e. MultiOperationsHPAnalyze.

    :param pipeline: pipeline object to analyze
    :param train_data: data used for Pipeline training
    :param test_data: data used for Pipeline validation
    :param approaches: methods applied to pipeline \
    Default: [MultiOperationsHPAnalyze]
    :param requirements: extra requirements to define specific details for different approaches.\
    See SensitivityAnalysisRequirements class documentation.
    :param path_to_save: path to save results to. Default: ~home/Fedot/sensitivity/pipeline_sa
    Default: False
    :param log: log: Log object to record messages
    """

    def __init__(self, pipeline: Pipeline, train_data: InputData, test_data: InputData,
                 approaches: Optional[List[Type[MultiOperationsHPAnalyze]]] = None,
                 requirements: SensitivityAnalysisRequirements = None,
                 path_to_save=None, log: Log = None):
        self.pipeline = pipeline
        self.train_data = train_data
        self.test_data = test_data
        self.requirements = \
            SensitivityAnalysisRequirements() if requirements is None else requirements
        self.approaches = [MultiOperationsHPAnalyze] if approaches is None else approaches
        self.path_to_save = \
            join(default_fedot_data_dir(), 'sensitivity', 'pipeline_sa') if path_to_save is None else path_to_save

        self.log = default_log(__name__) if log is None else log

    def analyze(self) -> dict:
        """
        Apply defined approaches for the black-box pipeline analysis
        :return:
        """
        all_approaches_results = dict()
        for approach in self.approaches:
            analyze_result = approach(pipeline=self.pipeline,
                                      train_data=self.train_data,
                                      test_data=self.test_data,
                                      requirements=self.requirements).analyze()
            all_approaches_results[f'{approach.__name__}'] = analyze_result

        if self.requirements.is_save:
            self._save_results_to_json(all_approaches_results)

        return all_approaches_results

    def _save_results_to_json(self, result: dict):
        result_file = join(self.path_to_save, 'pipeline_SA_results.json')
        with open(result_file, 'w', encoding='utf-8') as file:
            file.write(json.dumps(result, indent=4))
