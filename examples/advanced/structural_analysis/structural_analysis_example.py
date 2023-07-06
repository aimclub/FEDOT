import os
from functools import partial

from golem.core.dag.graph_verifier import GraphVerifier
from golem.core.optimisers.objective import Objective
from golem.structural_analysis.graph_sa.graph_structural_analysis import GraphStructuralAnalysis
from golem.structural_analysis.graph_sa.sa_requirements import StructuralAnalysisRequirements
from typing import Callable, Dict, Any, Optional, Tuple

from examples.advanced.structural_analysis.dataset_access import get_scoring_data
from examples.advanced.structural_analysis.pipelines_access import get_three_depth_manual_class_pipeline
from fedot.core.composer.metrics import ROCAUC
from fedot.core.data.data import InputData
from fedot.core.optimisers.objective import PipelineObjectiveEvaluate
from fedot.core.optimisers.objective.data_source_splitter import DataSourceSplitter
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.pipeline_advisor import PipelineChangeAdvisor
from fedot.core.pipelines.pipeline_composer_requirements import PipelineComposerRequirements
from fedot.core.pipelines.pipeline_node_factory import PipelineOptNodeFactory
from fedot.core.pipelines.verification import common_rules
from fedot.core.repository.quality_metrics_repository import MetricsRepository, ClassificationMetricsEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.utilities.project_import_export import DEFAULT_PATH


class SAObjective(Objective):
    """ Objective for Structural Analysis. . """
    def __init__(self,
                 objective: Callable,
                 quality_metrics: Dict[Any, Callable],
                 complexity_metrics: Optional[Dict[Any, Callable]] = None,
                 is_multi_objective: bool = False,
                 ):
        self.objective = objective
        super().__init__(quality_metrics=quality_metrics, complexity_metrics=complexity_metrics,
                         is_multi_objective=is_multi_objective)

    def __call__(self, pipeline: Pipeline) -> float:
        return self.objective(pipeline)


def set_up(train_data: InputData, test_data: InputData) -> Tuple[PipelineOptNodeFactory, SAObjective, SAObjective]:
    """ Build initial infrastructure for performing SA: node factory, objectives. """
    def _construct_objective(data: InputData) -> SAObjective:
        """ Build objective function with fit and predict functions inside. """
        data_producer = DataSourceSplitter().build(data=data)
        objective_function = PipelineObjectiveEvaluate(objective=Objective(quality_metrics=get_value),
                                                       data_producer=data_producer)
        objective = SAObjective(objective=objective_function, quality_metrics=metrics_)
        return objective

    task = Task(TaskTypesEnum.classification)
    advisor = PipelineChangeAdvisor(task)
    primary_operations = ['bernb', 'rf', 'qda', 'pca', 'normalization']
    secondary_operations = ['dt', 'logit', 'rf', 'scaling']
    requirements = PipelineComposerRequirements(primary=primary_operations,
                                                secondary=secondary_operations)
    node_factory = PipelineOptNodeFactory(requirements=requirements, advisor=advisor)

    get_value = partial(ROCAUC().get_value, reference_data=test_data)
    metrics_ = {ClassificationMetricsEnum.ROCAUC: get_value}

    # build objective function with fit and predict functions inside
    train_objective = _construct_objective(data=train_data)
    test_objective = _construct_objective(data=test_data)
    return node_factory, train_objective, test_objective


if __name__ == '__main__':
    pipeline = get_three_depth_manual_class_pipeline()
    train_data, test_data = get_scoring_data()

    metrics = MetricsRepository().metric_by_id(ClassificationMetricsEnum.ROCAUC)
    main_metric_idx = 0

    node_factory, train_objective, test_objective = set_up(train_data, test_data)

    print(f'INITIAL METRIC: {test_objective(pipeline)}')

    requirements = StructuralAnalysisRequirements(graph_verifier=GraphVerifier(common_rules),
                                                  main_metric_idx=main_metric_idx,
                                                  seed=1, replacement_number_of_random_operations_nodes=1,
                                                  replacement_number_of_random_operations_edges=1)

    path_to_save = os.path.join(DEFAULT_PATH, 'sa')

    # structural analysis will optimize given graph if at least one of the metrics was increased.
    sa = GraphStructuralAnalysis(objective=train_objective, node_factory=node_factory,
                                 requirements=requirements,
                                 path_to_save=path_to_save,
                                 is_visualize_per_iteration=False)

    optimized_pipeline, results = sa.optimize(graph=pipeline, n_jobs=1, max_iter=1)

    print(f'FINAL METRIC: {test_objective(optimized_pipeline)}')

    # to show SA results on each iteration
    GraphStructuralAnalysis.visualize_on_graph(graph=get_three_depth_manual_class_pipeline(),
                                               analysis_result=results,
                                               metric_idx_to_optimize_by=main_metric_idx,
                                               mode="by_iteration",
                                               font_size_scale=0.6)

    optimized_pipeline.show()
