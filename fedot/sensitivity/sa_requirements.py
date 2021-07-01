from typing import List, Optional

from fedot.core.pipelines.node import Node
from collections import namedtuple

HyperparamsAnalysisMetaParams = namedtuple('HyperparamsAnalysisMetaParams', ['analyze_method',
                                                                             'sample_method',
                                                                             'sample_size'])

ReplacementAnalysisMetaParams = namedtuple('ReplacementAnalysisMetaParams', ['nodes_to_replace_to',
                                                                             'number_of_random_operations'])


class SensitivityAnalysisRequirements:
    """
    Use this object to pass all the requirements needed for SA.

    :param hyperparams_analyze_method: defines string name of SA method to use. Defaults: 'sobol'
    :param hyperparams_sample_method: defines string name of sampling method to use. Defaults: 'saltelli'
    :param hyperparams_analysis_samples_size: defines the number of shyperparameters samples used in SA
    :param replacement_nodes_to_replace_to: defines nodes which is used in replacement analysis.
    :param replacement_number_of_random_operations: if replacement_nodes_to_replace_to is not filled, \
    define the number of randomly chosen operations used in replacement analysis.
    :param is_visualize: defines whether the SA visualization needs to be saved to .png files.
    :param is_save_results_to_json: defines whether the SA indices needs to be saved to .json file.
    :param metric: metric used for validation. Default: see MetricByTask
    """

    def __init__(self,
                 metric=None,
                 hyperparams_analyze_method: str = 'sobol',
                 hyperparams_sample_method: str = 'saltelli',
                 hyperparams_analysis_samples_size: int = 100,
                 replacement_nodes_to_replace_to: Optional[List[Node]] = None,
                 replacement_number_of_random_operations: Optional[int] = None,
                 is_visualize: bool = True,
                 is_save_results_to_json: bool = True):
        self.metric = metric
        self.hp_analysis_meta = HyperparamsAnalysisMetaParams(hyperparams_analyze_method,
                                                              hyperparams_sample_method,
                                                              hyperparams_analysis_samples_size,
                                                              )

        self.replacement_meta = ReplacementAnalysisMetaParams(replacement_nodes_to_replace_to,
                                                              replacement_number_of_random_operations)

        self.is_visualize = is_visualize
        self.is_save = is_save_results_to_json
