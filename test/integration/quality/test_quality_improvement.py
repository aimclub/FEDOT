import warnings
import pytest
from fedot.api.main import Fedot
from fedot.core.utils import project_root

warnings.filterwarnings("ignore")


# TODO need improvements
@pytest.mark.skip('No improvement after composing')
def test_classification_quality_improvement():
    # input data initialization
    train_data_path = f'{project_root()}/cases/data/scoring/scoring_train.csv'
    test_data_path = f'{project_root()}/cases/data/scoring/scoring_test.csv'

    problem = 'classification'

    baseline_model = Fedot(problem=problem)
    baseline_model.fit(features=train_data_path, target='target', predefined_model='xgboost')
    expected_baseline_quality = 0.827

    baseline_model.predict_proba(features=test_data_path)
    baseline_metrics = baseline_model.get_metrics()

    auto_model = Fedot(problem=problem, seed=42)
    auto_model.fit(features=train_data_path, target='target')
    auto_model.predict_proba(features=test_data_path)
    auto_metrics = auto_model.get_metrics()

    assert auto_metrics['roc_auc'] > baseline_metrics['roc_auc'] >= expected_baseline_quality
