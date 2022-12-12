import logging
import warnings

from typing import Sequence

import numpy as np

from fedot.api.main import Fedot
from fedot.core.optimisers.opt_history_objects.opt_history import OptHistory
from fedot.core.utils import fedot_project_root

warnings.filterwarnings("ignore")


def test_classification_quality_improvement():
    # input data initialization
    train_data_path = f'{fedot_project_root()}/cases/data/scoring/scoring_train.csv'
    test_data_path = f'{fedot_project_root()}/cases/data/scoring/scoring_test.csv'

    problem = 'classification'

    baseline_model = Fedot(problem=problem, seed=50)
    baseline_model.fit(features=train_data_path, target='target', predefined_model='rf')
    expected_baseline_quality = 0.823

    baseline_model.predict_proba(features=test_data_path)

    baseline_metrics = baseline_model.get_metrics()

    # Define parameters for composing
    timeout = 2
    composer_params = {
                       'num_of_generations': 20,
        'with_tuning': True,
        'preset': 'best_quality'}

    auto_model = Fedot(problem=problem, timeout=timeout, seed=50, logging_level=logging.DEBUG,
                       **composer_params, use_pipelines_cache=False, use_preprocessing_cache=False)
    auto_model.fit(features=train_data_path, target='target')
    auto_model.predict_proba(features=test_data_path)
    auto_metrics = auto_model.get_metrics()
    assert auto_metrics['roc_auc'] > baseline_metrics['roc_auc'] >= expected_baseline_quality


def test_multiobjective_improvement():
    # input data initialization
    train_data_path = f'{fedot_project_root()}/cases/data/scoring/scoring_train.csv'
    test_data_path = f'{fedot_project_root()}/cases/data/scoring/scoring_test.csv'
    problem = 'classification'

    # Define parameters for composing
    quality_metric = 'roc_auc'
    complexity_metric = 'node_num'
    metrics = [quality_metric, complexity_metric]

    timeout = 2
    composer_params = {
        'num_of_generations': 20,
        'with_tuning': False,
        'preset': 'best_quality',
        'metric': metrics,
    }

    auto_model = Fedot(problem=problem, timeout=timeout, seed=50, logging_level=logging.DEBUG,
                       **composer_params, use_pipelines_cache=False, use_preprocessing_cache=False)
    auto_model.fit(features=train_data_path, target='target')
    auto_model.predict_proba(features=test_data_path)
    auto_metrics = auto_model.get_metrics()

    quality_improved, complexity_improved = check_improvement(auto_model.history)

    assert quality_improved
    assert complexity_improved


def check_improvement(history: OptHistory):
    first_pop = history.individuals[1]
    pareto_front = history.archive_history[-1]

    first_pop_metrics = get_mean_metrics(first_pop)
    pareto_front_metrics = get_mean_metrics(pareto_front)

    quality_improved = pareto_front_metrics[0] < first_pop_metrics[0]
    complexity_improved = pareto_front_metrics[1] < first_pop_metrics[1]
    return quality_improved, complexity_improved


def get_mean_metrics(population) -> Sequence[float]:
    return np.mean([ind.fitness.values for ind in population], axis=0)
