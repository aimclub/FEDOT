import pandas as pd

from fedot.api.main import Fedot
from fedot.core.utils import fedot_project_root


def run_classification_example(timeout=None):
    train_data_path = f'{fedot_project_root()}/cases/data/scoring/scoring_train.csv'
    test_data_path = f'{fedot_project_root()}/cases/data/scoring/scoring_test.csv'

    problem = 'classification'

    baseline_model = Fedot(problem=problem, timeout=timeout)
    baseline_model.fit(features=train_data_path, target='target', predefined_model='xgboost')

    baseline_model.predict(features=test_data_path)
    print(baseline_model.get_metrics())

    auto_model = Fedot(problem=problem, seed=42, timeout=timeout)
    auto_model.fit(features=train_data_path, target='target')
    prediction = auto_model.predict_proba(features=test_data_path)
    print(auto_model.get_metrics())
    auto_model.plot_prediction()

    return prediction


def run_classification_multiobj_example(with_plot=True, timeout=None):
    train_data = pd.read_csv(f'{fedot_project_root()}/examples/data/Hill_Valley_with_noise_Training.data')
    test_data = pd.read_csv(f'{fedot_project_root()}/examples/data/Hill_Valley_with_noise_Testing.data')
    target = test_data['class']
    del test_data['class']
    problem = 'classification'

    auto_model = Fedot(problem=problem, timeout=timeout, preset='light',
                       composer_params={'metric': ['f1', 'node_num']}, seed=42)
    auto_model.fit(features=train_data, target='class')
    prediction = auto_model.predict_proba(features=test_data)
    print(auto_model.get_metrics(target))
    auto_model.plot_prediction()
    if with_plot:
        auto_model.best_models.show()

    return prediction


if __name__ == '__main__':
    run_classification_example()
    run_classification_multiobj_example()
