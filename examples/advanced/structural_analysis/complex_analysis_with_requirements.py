from os.path import join

from golem.structural_analysis.graph_sa.graph_structural_analysis import GraphStructuralAnalysis
from golem.structural_analysis.graph_sa.node_sa_approaches import NodeReplaceOperationAnalyze, NodeDeletionAnalyze

from examples.advanced.structural_analysis.dataset_access import get_scoring_data
from examples.advanced.structural_analysis.pipelines_access import get_three_depth_manual_class_pipeline
from fedot.core.utils import default_fedot_data_dir
from fedot.structural_analysis.operations_hp_sensitivity.multi_operations_sensitivity import MultiOperationsHPAnalyze
from fedot.structural_analysis.sa_requirements import SensitivityAnalysisRequirements


def run_analysis(pipeline, train_data, test_data):
    sa_requirements = SensitivityAnalysisRequirements(visualization=True,
                                                      is_save_results_to_json=True)
    approaches = [NodeDeletionAnalyze, NodeReplaceOperationAnalyze, MultiOperationsHPAnalyze]
    result_path = join(default_fedot_data_dir(), 'structural_analysis', f'{GraphStructuralAnalysis.__name__}')

    GraphStructuralAnalysis(pipeline=pipeline, train_data=train_data,
                            test_data=test_data, approaches=approaches,
                            requirements=sa_requirements, path_to_save=result_path).analyze()


if __name__ == '__main__':
    pipeline = get_three_depth_manual_class_pipeline()
    train_data, test_data = get_scoring_data()

    run_analysis(pipeline=pipeline, train_data=train_data, test_data=test_data)
