from fedot import FedotBuilder
from fedot.core.utils import fedot_project_root


if __name__ == '__main__':
    train_data_path = f'{fedot_project_root()}/examples/real_cases/data/scoring/scoring_train.csv'
    test_data_path = f'{fedot_project_root()}/examples/real_cases/data/scoring/scoring_test.csv'

    fedot = (FedotBuilder(problem='classification')
             .setup_composition(timeout=10, with_tuning=True, preset='best_quality')
             .setup_pipeline_evaluation(max_pipeline_fit_time=5, metric=['roc_auc', 'precision'])
             .build())
    fedot.fit(features=train_data_path, target='target')
    fedot.predict_proba(features=test_data_path)
    fedot.plot_prediction()
