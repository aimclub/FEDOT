import os

import joblib
from sklearn.metrics import roc_auc_score
from tpot import TPOTClassifier
from tpot import TPOTRegressor

from benchmark.benchmark_model_types import ModelTypesEnum
from benchmark.benchmark_utils import get_scoring_case_data_paths, get_models_hyperparameters
from core.models.data import InputData
from sklearn.metrics import mean_squared_error as mse

from core.repository.task_types import MachineLearningTasksEnum


def run_tpot(train_file_path: str, test_file_path: str, config_data: dict, task: MachineLearningTasksEnum,
             case_name='tpot_default'):
    max_runtime_mins = config_data['MAX_RUNTIME_MINS']
    generations = config_data['GENERATIONS']
    population_size = config_data['POPULATION_SIZE']

    train_data = InputData.from_csv(train_file_path)
    test_data = InputData.from_csv(test_file_path)

    result_model_filename = f"{case_name}_g{generations}" \
                            f"_p{population_size}_{task.value}.pkl"
    current_file_path = str(os.path.dirname(__file__))
    result_file_path = os.path.join(current_file_path, result_model_filename)

    if result_model_filename not in os.listdir(current_file_path):
        estimator = TPOTClassifier if task is MachineLearningTasksEnum.classification else TPOTRegressor

        model = estimator(generations=generations,
                          population_size=population_size,
                          verbosity=2,
                          random_state=42,
                          max_time_mins=max_runtime_mins)

        model.fit(train_data.features, train_data.target)
        model.export(output_file_name=f'{result_model_filename[:-4]}_pipeline.py')

        # sklearn pipeline object
        fitted_model_config = model.fitted_pipeline_
        joblib.dump(fitted_model_config, result_file_path, compress=1)

    imported_model = joblib.load(result_file_path)

    if task is MachineLearningTasksEnum.classification:
        predicted = imported_model.predict_proba(test_data.features)[:, 1]
        print(predicted)
        print(f'BEST_model: {imported_model}')
        result_metric = {'TPOT_ROC_AUC_test': round(roc_auc_score(test_data.target, predicted), 3)}
        print(f"TPOT_ROC_AUC_test:{result_metric['TPOT_ROC_AUC_test']} ")
    else:
        predicted = imported_model.predict(test_data.features)
        print(predicted)
        print(f'BEST_model: {imported_model}')
        result_metric = {'TPOT_MSE': round(mse(test_data.target, predicted), 3)}
        print(f"TPOT_MSE: {result_metric['TPOT_MSE']}")

    return result_metric


if __name__ == '__main__':
    train_data_path, test_data_path = get_scoring_case_data_paths()

    # MAX_RUNTIME_MINS should be equivalent to MAX_RUNTIME_SECS in b_h20.py
    tpot_config = get_models_hyperparameters()['TPOT']

    run_tpot(train_data_path, test_data_path, config_data=tpot_config, task=MachineLearningTasksEnum.classification)
