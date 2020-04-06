import os

import joblib
from sklearn.metrics import roc_auc_score
from tpot import TPOTClassifier

from benchmark.benchmark_utils import get_scoring_case_data_paths
from core.models.data import InputData


def run_tpot(train_file_path: str, test_file_path: str, case_name='tpot_default'):
    train_data = InputData.from_csv(train_file_path)
    test_data = InputData.from_csv(test_file_path)

    tpot_config_data = {'generations': 50, 'population_size': 10}

    model = TPOTClassifier(generations=tpot_config_data['generations'],
                           population_size=tpot_config_data['population_size'],
                           verbosity=2,
                           random_state=42)
    result_filename = f"{case_name}_g{tpot_config_data['generations']}_p{tpot_config_data['population_size']}.pkl"
    current_file_path = str(os.path.dirname(__file__))
    result_file_path = os.path.join(current_file_path, result_filename)

    if result_filename not in os.listdir(current_file_path):
        model.fit(train_data.features, train_data.target)
        model.export(output_file_name='tpot_exported_pipeline.py')

        # sklearn pipeline object
        fitted_model_config = model.fitted_pipeline_
        joblib.dump(fitted_model_config, result_file_path, compress=1)

    imported_model = joblib.load(result_file_path)
    predicted = imported_model.predict(test_data.features)
    print(f'BEST_model: {imported_model}')
    print(f'TPOT_model_score: {imported_model.score(test_data.features, test_data.target)}')
    print(f'TPOT_ROC_AUC: {roc_auc_score(test_data.target, predicted)}')


if __name__ == '__main__':
    train_data_path, test_data_path = get_scoring_case_data_paths()

    run_tpot(train_data_path, test_data_path)
