from fedot import FedotBuilder
from fedot.api.api_utils.presets import PresetsEnum
from fedot.core.utils import fedot_project_root


if __name__ == '__main__':
    train_data_path = f'{fedot_project_root()}/cases/data/scoring/scoring_train.csv'
    test_data_path = f'{fedot_project_root()}/cases/data/scoring/scoring_test.csv'

    fedot = (FedotBuilder(problem='classification')
             .setup_composition(timeout=10, with_tuning=True, preset=PresetsEnum.BEST_QUALITY)
             .setup_pipeline_evaluation(max_pipeline_fit_time=5, metric=['roc_auc', 'precision'])
             .build())
    fedot.fit(features=train_data_path, target='target')
    fedot.predict_proba(features=test_data_path)
    fedot.plot_prediction()
