import logging
import warnings
from typing import Sequence

import numpy as np

from fedot.api.main import Fedot
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
    seed = 50

    # Define parameters for composing
    quality_metric_1 = 'roc_auc'
    quality_metric_2 = 'accuracy'
    metrics = [quality_metric_1, quality_metric_2]

    timeout = 2
    composer_params = dict(num_of_generations=10,
                           pop_size=10,
                           with_tuning=False,
                           preset='best_quality',
                           metric=metrics)

    auto_model = Fedot(problem=problem, timeout=timeout, seed=seed, logging_level=logging.DEBUG,
                       **composer_params, use_pipelines_cache=False, use_preprocessing_cache=False)
    auto_model.fit(features=train_data_path, target='target')
    auto_model.predict_proba(features=test_data_path)
    auto_metrics = auto_model.get_metrics()

    quality_1_improved, quality_2_improved = check_improvement(auto_model.history)

    assert auto_metrics[quality_metric_1] > 0.75
    assert auto_metrics[quality_metric_2] > 0.9
    assert quality_1_improved
    assert quality_2_improved


def check_improvement(history):
    first_pop = history.individuals[1]
    pareto_front = history.archive_history[-1]

    first_pop_metrics = get_mean_metrics(first_pop)
    pareto_front_metrics = get_mean_metrics(pareto_front)

    quality_1_improved = pareto_front_metrics[0] < first_pop_metrics[0]
    quality_2_improved = pareto_front_metrics[1] < first_pop_metrics[1]
    return quality_1_improved, quality_2_improved


def get_mean_metrics(population) -> Sequence[float]:
    return np.mean([ind.fitness.values for ind in population], axis=0)
