from fedot import FedotBuilder
from fedot.core.utils import fedot_project_root, set_random_seed


def run_classification_example(timeout: float = None, visualization=False, with_tuning=True):
    problem = 'classification'
    train_data_path = f'{fedot_project_root()}/cases/data/scoring/scoring_train.csv'
    test_data_path = f'{fedot_project_root()}/cases/data/scoring/scoring_test.csv'

    auto_model = (FedotBuilder(problem=problem)
                  .setup_composition(timeout=timeout, with_tuning=with_tuning, preset='best_quality')
                  .setup_pipeline_evaluation(max_pipeline_fit_time=5, metric=['roc_auc', 'precision'])
                  .build())
    auto_model.fit(features=train_data_path, target='target')
    prediction = auto_model.predict_proba(features=test_data_path)

    print(auto_model.get_metrics(rounding_order=4))  # we can control the rounding of metrics
    if visualization:
        auto_model.plot_prediction()
    return prediction


if __name__ == '__main__':
    set_random_seed(42)

    run_classification_example(timeout=2.0, visualization=True)
