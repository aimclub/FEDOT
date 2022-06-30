import logging
import warnings

from fedot.api.main import Fedot
from fedot.core.utils import fedot_project_root

warnings.filterwarnings("ignore")


def test_classification_quality_improvement():
    # input data initialization
    train_data_path = f'{fedot_project_root()}/cases/data/scoring/scoring_train.csv'
    test_data_path = f'{fedot_project_root()}/cases/data/scoring/scoring_test.csv'

    problem = 'classification'

    baseline_model = Fedot(problem=problem)
    baseline_model.fit(features=train_data_path, target='target', predefined_model='rf')
    expected_baseline_quality = 0.823

    baseline_model.predict_proba(features=test_data_path)

    baseline_metrics = baseline_model.get_metrics()

    # Define parameters for composing
    timeout = 5.
    composer_params = {'max_depth': 3,
                       'max_arity': 3,
                       'pop_size': 20,
                       'num_of_generations': 20,
                       'with_tuning': True}

    auto_model = Fedot(problem=problem, timeout=timeout, seed=42, verbose_level=logging.DEBUG,
                       **composer_params)
    auto_model.fit(features=train_data_path, target='target')
    auto_model.predict_proba(features=test_data_path)
    auto_metrics = auto_model.get_metrics()
    assert auto_metrics['roc_auc'] > baseline_metrics['roc_auc'] >= expected_baseline_quality
