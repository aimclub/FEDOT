from os import makedirs
from os.path import join, exists

from examples.sensitivity_analysis.chains_access import chain_by_task
from fedot.core.data.data import InputData
from fedot.core.utils import default_fedot_data_dir
from fedot.sensitivity.node_sa_approaches import NodeDeletionAnalyze, NodeReplaceOperationAnalyze
from fedot.sensitivity.operations_hp_sensitivity.multi_operations_sensitivity import MultiOperationsHPAnalyze
from fedot.sensitivity.nodes_sensitivity import NodesAnalysis
from fedot.sensitivity.chain_sensitivity_facade import ChainSensitivityAnalysis
from fedot.sensitivity.chain_sensitivity import ChainAnalysis

SA_CLASS_WITH_APPROACHES = {'ChainSensitivityAnalysis': {'class': ChainSensitivityAnalysis,
                                                         'approaches': [NodeDeletionAnalyze,
                                                                        NodeReplaceOperationAnalyze,
                                                                        MultiOperationsHPAnalyze]},
                            'NodesAnalysis': {'class': NodesAnalysis,
                                              'approaches': [NodeDeletionAnalyze,
                                                             NodeReplaceOperationAnalyze]},
                            'ChainAnalysis': {'class': ChainAnalysis,
                                              'approaches': [MultiOperationsHPAnalyze]},

                            }


def run_case_analysis(train_data: InputData, test_data: InputData,
                      case_name: str, task, metric, sa_class,
                      is_composed=False, result_path=None):
    chain = chain_by_task(task=task, metric=metric,
                          data=train_data, is_composed=is_composed)

    chain.fit(train_data)

    if not result_path:
        result_path = join(default_fedot_data_dir(), 'sensitivity', f'{case_name}')
        if not exists(result_path):
            makedirs(result_path)
    chain.show(join(result_path, f'{case_name}'))

    sa_class_with_approaches = SA_CLASS_WITH_APPROACHES.get(sa_class)
    sa_class = sa_class_with_approaches['class']
    approaches = sa_class_with_approaches['approaches']

    chain_analysis_result = sa_class(chain=chain,
                                     train_data=train_data,
                                     test_data=test_data,
                                     approaches=approaches,
                                     path_to_save=result_path).analyze()

    print(f'chain analysis result {chain_analysis_result}')
