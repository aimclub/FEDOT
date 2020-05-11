"""
Before running this script make sure you have Java Development Kit installed.
Link to download: https://www.oracle.com/java/technologies/javase-jdk8-downloads.html
"""

import os

import h2o
from sklearn.metrics import roc_auc_score

from benchmark.benchmark_utils import (get_scoring_case_data_paths,
                                       get_models_hyperparameters,
                                       get_h2o_connect_config)
from core.models.data import InputData
from core.models.evaluation.automl_eval import fit_h2o, predict_h2o
from core.repository.task_types import MachineLearningTasksEnum

CURRENT_PATH = str(os.path.dirname(__file__))


def run_h2o(train_file_path: str, test_file_path: str, task: MachineLearningTasksEnum, case_name='h2o_default'):
    config_data = get_models_hyperparameters()['H2O']
    max_models = config_data['MAX_MODELS']
    max_runtime_secs = config_data['MAX_RUNTIME_SECS']

    result_filename = f'{case_name}_m{max_models}_rs{max_runtime_secs}_{task.name}'
    exported_model_path = os.path.join(CURRENT_PATH, result_filename)

    # TODO Regression
    if result_filename not in os.listdir(CURRENT_PATH):
        train_data = InputData.from_csv(train_file_path)
        best_model = fit_h2o(train_data)
        temp_exported_model_path = h2o.save_model(model=best_model, path=CURRENT_PATH)

        os.renames(temp_exported_model_path, exported_model_path)

    ip, port = get_h2o_connect_config()
    h2o.init(ip=ip, port=port, name='h2o_server')

    imported_model = h2o.load_model(exported_model_path)

    test_frame = InputData.from_csv(test_file_path)
    true_target = test_frame.target

    predictions = predict_h2o(imported_model, test_frame)

    if task is MachineLearningTasksEnum.classification:
        train_roc_auc_value = round(imported_model.auc(train=True), 3)
        valid_roc_auc_value = round(imported_model.auc(valid=True), 3)
        test_roc_auc_value = round(roc_auc_score(true_target, predictions), 3)

        metrics = {'H2O_ROC_AUC_train': train_roc_auc_value, 'H2O_ROC_AUC_valid': valid_roc_auc_value,
                   'H2O_ROC_AUC_test': test_roc_auc_value}

        print(f"H2O_ROC_AUC_train: {metrics['H2O_ROC_AUC_train']}")
        print(f"H2O_ROC_AUC_valid: {metrics['H2O_ROC_AUC_valid']}")
        print(f"H2O_ROC_AUC_test: {metrics['H2O_ROC_AUC_test']}")
    else:
        mse_train = imported_model.mse()
        rmse_train = imported_model.rmse()

        metrics = {'H2O_MSE_train': mse_train, 'H2O_RMSE_train': rmse_train}

        print(f"H2O_MSE_train: {metrics['H2O_MSE_train']}")
        print(f"H2O_RMSE_train: {metrics['H2O_RMSE_train']}")

    h2o.shutdown(prompt=False)

    return metrics


if __name__ == '__main__':
    train_file, test_file = get_scoring_case_data_paths()

    run_h2o(train_file, test_file, task=MachineLearningTasksEnum.classification)
