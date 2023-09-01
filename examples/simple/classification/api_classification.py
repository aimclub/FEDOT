from fedot.api.main import Fedot
from fedot.core.utils import fedot_project_root, set_random_seed


def run_classification_example(timeout: float = None, visualization=False, with_tuning=True):
    problem = 'classification'
    train_data_path = f'{fedot_project_root()}/cases/data/scoring/scoring_train.csv'
    test_data_path = f'{fedot_project_root()}/cases/data/scoring/scoring_test.csv'

    baseline_model = Fedot(problem=problem, timeout=timeout)
    baseline_model.fit(features=train_data_path, target='target', predefined_model='rf')

    baseline_model.predict(features=test_data_path)
    print(baseline_model.get_metrics())

    auto_model = Fedot(problem=problem, timeout=timeout, n_jobs=-1, preset='best_quality',
                       max_pipeline_fit_time=5, metric='roc_auc', with_tuning=with_tuning)
    auto_model.fit(features=train_data_path, target='target')
    prediction = auto_model.predict_proba(features=test_data_path)

    print(auto_model.get_metrics(decimal_places_num=4))  # we can control the rounding of metrics
    if visualization:
        auto_model.plot_prediction()
    return prediction


if __name__ == '__main__':
    set_random_seed(42)

    run_classification_example(timeout=10.0, visualization=True)
