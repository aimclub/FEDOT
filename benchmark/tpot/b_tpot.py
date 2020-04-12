import os

import joblib
from sklearn.metrics import roc_auc_score
from tpot import TPOTClassifier
from tpot import TPOTRegressor

from benchmark.benchmark_utils import get_scoring_case_data_paths
from core.models.data import InputData
from sklearn.metrics import mean_squared_error as mse

# MAX_RUNTIME_MINS should be equivalent to MAX_RUNTIME_SECS in b_h20.py
MAX_RUNTIME_MINS = 30
GENERATIONS = 50
POPULATION_SIZE = 10


def run_tpot(train_file_path: str, test_file_path: str, case_name='tpot_default', is_classification=True):
    train_data = InputData.from_csv(train_file_path)
    test_data = InputData.from_csv(test_file_path)

    tpot_config_data = {'generations': GENERATIONS, 'population_size': POPULATION_SIZE}
    task = 'clss' if is_classification else 'reg'

    result_model_filename = f"{case_name}_g{tpot_config_data['generations']}" \
                            f"_p{tpot_config_data['population_size']}_{task}.pkl"
    current_file_path = str(os.path.dirname(__file__))
    result_file_path = os.path.join(current_file_path, result_model_filename)

    if result_model_filename not in os.listdir(current_file_path):
        estimator = TPOTClassifier if is_classification else TPOTRegressor

        model = estimator(generations=tpot_config_data['generations'],
                          population_size=tpot_config_data['population_size'],
                          verbosity=2,
                          random_state=42,
                          max_time_mins=MAX_RUNTIME_MINS)

        model.fit(train_data.features, train_data.target)
        model.export(output_file_name=f'{result_model_filename[:-4]}_pipeline.py')

        # sklearn pipeline object
        fitted_model_config = model.fitted_pipeline_
        joblib.dump(fitted_model_config, result_file_path, compress=1)

    imported_model = joblib.load(result_file_path)

    predicted = imported_model.predict(test_data.features)
    print(f'BEST_model: {imported_model}')

    if is_classification:
        result_metric = f'TPOT_ROC_AUC_test: {roc_auc_score(test_data.target, predicted)}'
        print(result_metric)
    else:
        result_metric = f'TPOT_MSE: {mse(test_data.target, predicted)}'
        print(result_metric)

    return result_metric


if __name__ == '__main__':
    train_data_path, test_data_path = get_scoring_case_data_paths()

    run_tpot(train_data_path, test_data_path)
