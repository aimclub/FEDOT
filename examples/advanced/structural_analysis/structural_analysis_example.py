import os
from functools import partial

from golem.core.dag.graph_verifier import GraphVerifier
from golem.core.dag.verification_rules import DEFAULT_DAG_RULES, has_one_root, has_no_cycle, has_no_isolated_components, \
    has_no_isolated_nodes, has_no_self_cycled_nodes
from golem.core.optimisers.graph import OptGraph
from golem.core.optimisers.objective import Objective
from golem.structural_analysis.graph_sa.graph_structural_analysis import GraphStructuralAnalysis
from golem.structural_analysis.graph_sa.sa_requirements import StructuralAnalysisRequirements
from typing import Callable, Dict, Any, Optional, Tuple, List

from examples.advanced.structural_analysis.dataset_access import get_scoring_data
from examples.advanced.structural_analysis.pipelines_access import get_three_depth_manual_class_pipeline
from fedot.core.data.data import InputData
from fedot.core.optimisers.objective import PipelineObjectiveEvaluate
from fedot.core.optimisers.objective.data_source_splitter import DataSourceSplitter
from fedot.core.pipelines.adapters import PipelineAdapter
from fedot.core.pipelines.pipeline_advisor import PipelineChangeAdvisor
from fedot.core.pipelines.pipeline_composer_requirements import PipelineComposerRequirements
from fedot.core.pipelines.pipeline_node_factory import PipelineOptNodeFactory
from fedot.core.repository.quality_metrics_repository import ClassificationMetricsEnum, QualityMetricsEnum, \
    MetricsRepository
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.utilities.project_import_export import DEFAULT_PATH


class SAObjective(Objective):
    """ Objective for Structural Analysis.
    This objective has to evaluate pipeline in __call__ method and have 'metrics' field to identify
    which metrics are optimized.
    """
    def __init__(self,
                 objective: Callable,
                 quality_metrics: Dict[Any, Callable],
                 complexity_metrics: Optional[Dict[Any, Callable]] = None,
                 is_multi_objective: bool = False,
                 ):
        self.objective = objective
        super().__init__(quality_metrics=quality_metrics, complexity_metrics=complexity_metrics,
                         is_multi_objective=is_multi_objective)

    def __call__(self, graph: OptGraph) -> float:
        pip = PipelineAdapter().restore(graph)
        return self.objective(pip)


def structural_analysis_set_up(train_data: InputData, test_data: InputData,
                               task: TaskTypesEnum = TaskTypesEnum.classification,
                               metric: QualityMetricsEnum = ClassificationMetricsEnum.ROCAUC,
                               primary_operations: List[str] = None, secondary_operations: List[str] = None) \
        -> Tuple[PipelineOptNodeFactory, SAObjective, SAObjective]:
    """ Build initial infrastructure for performing SA: node factory, objectives.
    Can be reused for other SA applications, appropriate parameters must be specified then. """
    def _construct_objective(data: InputData, metric: QualityMetricsEnum) -> SAObjective:
        """ Build objective function with fit and predict functions inside. """
        metric_func = MetricsRepository.metric_by_id(metric)
        get_value = partial(metric_func, reference_data=data)
        metrics_ = {metric: data}

        data_producer = DataSourceSplitter().build(data=data)
        objective_function = PipelineObjectiveEvaluate(objective=Objective(quality_metrics=get_value),
                                                       data_producer=data_producer)
        objective = SAObjective(objective=objective_function, quality_metrics=metrics_)
        return objective

    task = Task(task)
    advisor = PipelineChangeAdvisor(task)
    primary_operations = primary_operations or ['rf', 'pca', 'normalization', 'scaling']
    secondary_operations = secondary_operations or ['dt', 'logit', 'rf', 'knn']
    requirements = PipelineComposerRequirements(primary=primary_operations,
                                                secondary=secondary_operations)
    node_factory = PipelineOptNodeFactory(requirements=requirements, advisor=advisor)

    # build objective function with fit and predict functions inside
    optimization_metric = metric
    train_objective = _construct_objective(data=train_data, metric=optimization_metric)
    test_objective = _construct_objective(data=test_data, metric=optimization_metric)
    return node_factory, train_objective, test_objective


if __name__ == '__main__':
    initial_graph = PipelineAdapter().adapt(get_three_depth_manual_class_pipeline())
    train_data, test_data = get_scoring_data()

    main_metric_idx = 0

    node_factory, train_objective, test_objective = structural_analysis_set_up(train_data, test_data)

    print(f'INITIAL METRIC: {test_objective(initial_graph)}')

    verification_rules = [has_one_root, has_no_cycle, has_no_isolated_components,
                          has_no_self_cycled_nodes, has_no_isolated_nodes]
    requirements = StructuralAnalysisRequirements(graph_verifier=GraphVerifier(verification_rules),
                                                  main_metric_idx=main_metric_idx,
                                                  seed=1, replacement_number_of_random_operations_nodes=2,
                                                  replacement_number_of_random_operations_edges=2)

    path_to_save = os.path.join(DEFAULT_PATH, 'sa')
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)

    # structural analysis will optimize given graph if the specified main metric increased
    sa = GraphStructuralAnalysis(objective=train_objective, node_factory=node_factory,
                                 requirements=requirements,
                                 path_to_save=path_to_save,
                                 is_visualize_per_iteration=False)

    optimized_graph, results = sa.optimize(graph=initial_graph, n_jobs=1, max_iter=2)

    print(f'FINAL METRIC: {test_objective(optimized_graph)}')

    # to show SA results on each iteration
    GraphStructuralAnalysis.visualize_on_graph(graph=PipelineAdapter().adapt(get_three_depth_manual_class_pipeline()),
                                               analysis_result=results,
                                               metric_idx_to_optimize_by=main_metric_idx,
                                               mode='by_iteration',
                                               save_path=path_to_save,
                                               font_size_scale=0.6)

    optimized_graph.show()
