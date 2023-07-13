import logging
import warnings
from typing import Sequence

import numpy as np

from fedot.api.main import Fedot
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.utils import fedot_project_root

warnings.filterwarnings("ignore")


def test_classification_quality_improvement():
    # input data initialization
    train_data_path = fedot_project_root().joinpath('cases/data/scoring/scoring_train.csv')
    test_data_path = fedot_project_root().joinpath('cases/data/scoring/scoring_test.csv')

    seed = 50
    problem = 'classification'
    with_tuning = False

    common_params = dict(problem=problem,
                         n_jobs=1,
                         use_pipelines_cache=False,
                         use_preprocessing_cache=False,
                         with_tuning=with_tuning,
                         logging_level=logging.DEBUG,
                         seed=seed)

    expected_baseline_quality = 0.750
    baseline_model = Fedot(**common_params)
    baseline_model.fit(features=train_data_path, target='target', predefined_model='bernb')
    baseline_model.predict_proba(features=test_data_path)
    baseline_metrics = baseline_model.get_metrics()

    # Define parameters for composing
    auto_model = Fedot(timeout=2, num_of_generations=20, preset='best_quality',
                       **common_params)
    auto_model.fit(features=train_data_path, target='target')
    auto_model.predict_proba(features=test_data_path)
    auto_metrics = auto_model.get_metrics()

    assert auto_metrics['roc_auc_pen'] > baseline_metrics['roc_auc_pen'] >= expected_baseline_quality


def test_multiobjective_improvement():
    # input data initialization
    train_data_path = fedot_project_root().joinpath('cases/data/scoring/scoring_train.csv')
    test_data_path = fedot_project_root().joinpath('cases/data/scoring/scoring_test.csv')
    problem = 'classification'
    seed = 1

    # Define parameters for composing
    quality_metric = 'roc_auc'
    complexity_metric = 'node_number'
    metrics = [quality_metric, complexity_metric]

    timeout = 3
    composer_params = dict(num_of_generations=10,
                           pop_size=10,
                           with_tuning=False,
                           preset='best_quality',
                           metric=metrics)

    root_node = PipelineNode('logit')
    child_1 = PipelineNode('rf')
    child_2 = PipelineNode('knn')
    [root_node.nodes_from.append(child) for child in [child_1, child_2]]
    initial_pipeline = Pipeline(nodes=[root_node] + root_node.nodes_from)

    auto_model = Fedot(problem=problem, timeout=timeout, seed=seed, logging_level=logging.DEBUG,
                       initial_assumption=initial_pipeline,
                       **composer_params, use_pipelines_cache=False, use_preprocessing_cache=False)
    auto_model.fit(features=train_data_path, target='target')
    auto_model.predict_proba(features=test_data_path)
    auto_metrics = auto_model.get_metrics()

    quality_improved, complexity_improved = check_improvement(auto_model.history)

    assert auto_metrics[quality_metric] > 0.75
    assert auto_metrics[complexity_metric] <= 0.2
    assert quality_improved
    assert complexity_improved


def check_improvement(history):
    first_pop = history.individuals[1]
    pareto_front = history.archive_history[-1]

    first_pop_metrics = get_mean_metrics(first_pop)
    pareto_front_metrics = get_mean_metrics(pareto_front)

    quality_improved = pareto_front_metrics[0] < first_pop_metrics[0]
    complexity_improved = pareto_front_metrics[1] < first_pop_metrics[1]
    return quality_improved, complexity_improved


def get_mean_metrics(population) -> Sequence[float]:
    return np.mean([ind.fitness.values for ind in population], axis=0)
