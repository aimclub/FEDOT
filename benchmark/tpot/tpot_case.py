import os

import joblib
from sklearn.metrics import roc_auc_score
from tpot import TPOTClassifier

from benchmark.benchmark_utils import get_initial_data_paths
from core.models.data import InputData


def run_tpot(*args):
    train_data, test_data = args
    model = TPOTClassifier(generations=50, population_size=10, verbosity=2, random_state=42)
    result_filename = 'result_model.pkl'

    if result_filename not in os.listdir('.'):
        model.fit(train_data.features, train_data.target)
        model.export(output_file_name='tpot_exported_pipeline.py')

        # sklearn pipeline object
        fitted_model_config = model.fitted_pipeline_
        joblib.dump(fitted_model_config, result_filename, compress=1)

    imported_model = joblib.load(result_filename)
    predicted = imported_model.predict(test_data.features)
    print(f'Model score: {imported_model.score(test_data.features, test_data.target)}')
    print(f'ROC_AUC: {roc_auc_score(test_data.target, predicted)}')


if __name__ == '__main__':
    train_data_path, test_data_path = get_initial_data_paths()

    train_input_data = InputData.from_csv(train_data_path)
    test_input_data = InputData.from_csv(test_data_path)

    run_tpot(train_input_data, test_input_data)
