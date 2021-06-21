from os.path import join

from examples.sensitivity_analysis.dataset_access import get_scoring_data
from examples.sensitivity_analysis.chains_access import get_three_depth_manual_class_chain
from fedot.core.utils import default_fedot_data_dir
from fedot.sensitivity.chain_sensitivity_facade import ChainSensitivityAnalysis
from fedot.sensitivity.sa_requirements import SensitivityAnalysisRequirements
from fedot.sensitivity.operations_hp_sensitivity.multi_operations_sensitivity import MultiOperationsHPAnalyze
from fedot.sensitivity.node_sa_approaches import NodeDeletionAnalyze, NodeReplaceOperationAnalyze


def run_analysis(chain, train_data, test_data):
    sa_requirements = SensitivityAnalysisRequirements(is_visualize=True,
                                                      is_save_results_to_json=True)
    approaches = [NodeDeletionAnalyze, NodeReplaceOperationAnalyze, MultiOperationsHPAnalyze]
    result_path = join(default_fedot_data_dir(), 'sensitivity', f'{ChainSensitivityAnalysis.__name__}')

    ChainSensitivityAnalysis(chain=chain, train_data=train_data,
                             test_data=test_data, approaches=approaches,
                             requirements=sa_requirements, path_to_save=result_path).analyze()


if __name__ == '__main__':
    chain = get_three_depth_manual_class_chain()
    train_data, test_data = get_scoring_data()

    run_analysis(chain=chain, train_data=train_data, test_data=test_data)
