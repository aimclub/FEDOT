from os import makedirs
from os.path import join, exists

from examples.advanced.sensitivity_analysis.pipelines_access import pipeline_by_task
from fedot.core.data.data import InputData
from fedot.core.utils import default_fedot_data_dir
from fedot.sensitivity.node_sa_approaches import NodeDeletionAnalyze, NodeReplaceOperationAnalyze
from fedot.sensitivity.operations_hp_sensitivity.multi_operations_sensitivity import MultiOperationsHPAnalyze
from fedot.sensitivity.nodes_sensitivity import NodesAnalysis
from fedot.sensitivity.pipeline_sensitivity_facade import PipelineSensitivityAnalysis
from fedot.sensitivity.pipeline_sensitivity import PipelineAnalysis

SA_CLASS_WITH_APPROACHES = {'PipelineSensitivityAnalysis': {'class': PipelineSensitivityAnalysis,
                                                         'approaches': [NodeDeletionAnalyze,
                                                                        NodeReplaceOperationAnalyze,
                                                                        MultiOperationsHPAnalyze]},
                            'NodesAnalysis': {'class': NodesAnalysis,
                                              'approaches': [NodeDeletionAnalyze,
                                                             NodeReplaceOperationAnalyze]},
                            'PipelineAnalysis': {'class': PipelineAnalysis,
                                              'approaches': [MultiOperationsHPAnalyze]},

                            }


def run_case_analysis(train_data: InputData, test_data: InputData,
                      case_name: str, task, metric, sa_class,
                      is_composed=False, result_path=None):
    pipeline = pipeline_by_task(task=task, metric=metric,
                          data=train_data, is_composed=is_composed)

    pipeline.fit(train_data)

    if not result_path:
        result_path = join(default_fedot_data_dir(), 'sensitivity', f'{case_name}')
        if not exists(result_path):
            makedirs(result_path)
    pipeline.show(join(result_path, f'{case_name}'))

    sa_class_with_approaches = SA_CLASS_WITH_APPROACHES.get(sa_class)
    sa_class = sa_class_with_approaches['class']
    approaches = sa_class_with_approaches['approaches']

    pipeline_analysis_result = sa_class(pipeline=pipeline,
                                        train_data=train_data,
                                        test_data=test_data,
                                        approaches=approaches,
                                        path_to_save=result_path).analyze()

    print(f'pipeline analysis result {pipeline_analysis_result}')
