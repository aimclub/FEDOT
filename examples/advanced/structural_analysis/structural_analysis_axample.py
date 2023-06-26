import os
from functools import partial
from os.path import join

from golem.core.adapter.nx_adapter import BaseNetworkxAdapter
from golem.core.dag.graph import Graph
from golem.core.dag.graph_verifier import GraphVerifier
from golem.core.dag.verification_rules import DEFAULT_DAG_RULES, has_one_root, has_root, has_no_self_cycled_nodes, \
    has_no_cycle, has_no_isolated_components, has_no_isolated_nodes
from golem.core.optimisers.objective import Objective
from golem.core.optimisers.opt_node_factory import DefaultOptNodeFactory
from golem.structural_analysis.graph_sa.graph_structural_analysis import GraphStructuralAnalysis
from golem.structural_analysis.graph_sa.sa_requirements import StructuralAnalysisRequirements
from typing import Callable, Dict, Any, Optional

from examples.advanced.structural_analysis.dataset_access import get_scoring_data
from examples.advanced.structural_analysis.pipelines_access import get_three_depth_manual_class_pipeline
from fedot.core.composer.metrics import ROCAUC
from fedot.core.data.data import InputData
from fedot.core.optimisers.objective import PipelineObjectiveEvaluate, MetricsObjective
from fedot.core.pipelines.adapters import PipelineAdapter
from fedot.core.pipelines.pipeline_advisor import PipelineChangeAdvisor
from fedot.core.pipelines.pipeline_composer_requirements import PipelineComposerRequirements
from fedot.core.pipelines.pipeline_node_factory import PipelineOptNodeFactory
from fedot.core.repository.quality_metrics_repository import MetricsRepository, ClassificationMetricsEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.utilities.project_import_export import DEFAULT_PATH
from test.unit.composer.test_quality_metrics import default_valid_pipeline

DAG_RULES_FOR_PIPELINES = [has_one_root, has_no_cycle, has_no_isolated_components,
                           has_no_self_cycled_nodes, has_no_isolated_nodes]


class SAObjective(Objective):
    def __init__(self,
                 objective: Callable,
                 quality_metrics: Dict[Any, Callable],
                 reference_data: InputData,
                 complexity_metrics: Optional[Dict[Any, Callable]] = None,
                 is_multi_objective: bool = False,
                 ):
        self.objective = objective
        self.reference_data = reference_data
        super().__init__(quality_metrics=quality_metrics, complexity_metrics=complexity_metrics,
                         is_multi_objective=is_multi_objective)

    def __call__(self, graph: Graph) -> float:
        pip = PipelineAdapter().restore(graph)
        pip.fit(self.reference_data)
        return self.objective(pip)


if __name__ == '__main__':
    pipeline = get_three_depth_manual_class_pipeline()
    train_data, test_data = get_scoring_data()

    metrics = MetricsRepository().metric_by_id(ClassificationMetricsEnum.ROCAUC)

    get_value = partial(ROCAUC().get_value, reference_data=test_data)
    metrics_ = {ClassificationMetricsEnum.ROCAUC: get_value}
    objective = SAObjective(objective=Objective(quality_metrics=get_value), quality_metrics=metrics_,
                            reference_data=train_data)

    task = Task(TaskTypesEnum.classification)
    advisor = PipelineChangeAdvisor(task)
    primary_operations = ['bernb', 'rf', 'qda', 'pca', 'normalization']
    secondary_operations = ['dt', 'logit', 'rf', 'scaling']
    requirements = PipelineComposerRequirements(primary=primary_operations,
                                                secondary=secondary_operations)
    node_factory = PipelineOptNodeFactory(requirements=requirements, advisor=advisor)

    # pipeline.fit(train_data)

    print(f'INITIAL METRIC: {objective(pipeline)}')

    requirements = StructuralAnalysisRequirements(graph_verifier=GraphVerifier(DAG_RULES_FOR_PIPELINES),
                                                  main_metric_idx=0,
                                                  seed=1, replacement_number_of_random_operations_nodes=2,
                                                  replacement_number_of_random_operations_edges=2)

    path_to_save = os.path.join(DEFAULT_PATH, 'sa')
    # structural analysis will optimize given graph if at least one of the metrics was increased.
    sa = GraphStructuralAnalysis(objective=objective, node_factory=node_factory,
                                 requirements=requirements,
                                 path_to_save=path_to_save,
                                 is_visualize_per_iteration=False)

    graph, results = sa.optimize(graph=pipeline, n_jobs=1, max_iter=3)
    #
    # # to show SA results on each iteration
    # optimized_graph = GraphStructuralAnalysis.visualize_on_graph(graph=graph_to_analyze,
    #                                                              analysis_result=results,
    #                                                              metric_idx_to_optimize_by=0,
    #                                                              mode="by_iteration",
    #                                                              font_size_scale=0.6)

    graph.show()
