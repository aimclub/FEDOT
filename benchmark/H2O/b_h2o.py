"""
Before running this script make sure you have Java Development Kit installed.
Link to download: https://www.oracle.com/java/technologies/javase-jdk8-downloads.html
"""

import os

import h2o
from h2o.automl import H2OAutoML

from benchmark.benchmark_model_types import ModelTypesEnum
from benchmark.benchmark_utils import get_scoring_case_data_paths, get_models_hyperparameters
from core.repository.task_types import MachineLearningTasksEnum

CURRENT_PATH = str(os.path.dirname(__file__))
IP = '127.0.0.1'
PORT = 8888


# MAX_RUNTIME_SECS should be equivalent to MAX_RUNTIME_MINS in b_tpot.py


def run_h2o(train_file_path: str, test_file_path: str, config_data: dict, target_name: str,
            task: MachineLearningTasksEnum, case_name='h2o_default'):
    max_models = config_data['MAX_MODELS']
    max_runtime_secs = config_data['MAX_RUNTIME_SECS']

    h2o.init(ip=IP, port=PORT, name='h2o_server')

    # Data preprocessing
    train_frame = h2o.import_file(train_file_path)
    test_frame = h2o.import_file(test_file_path)
    predictors = train_frame.columns.remove(target_name)
    train_frame[target_name] = train_frame[target_name].asfactor()
    test_frame[target_name] = test_frame[target_name].asfactor()

    result_filename = f"{case_name}_m{max_models}_rs{max_runtime_secs}_{task.value}"
    exported_model_path = os.path.join(CURRENT_PATH, result_filename)

    # TODO Regression
    if result_filename not in os.listdir(CURRENT_PATH):
        model = H2OAutoML(max_models=max_models, seed=1, max_runtime_secs=max_runtime_secs)
        model.train(x=predictors, y=target_name, training_frame=train_frame, validation_frame=test_frame)
        best_model = model.leader
        temp_exported_model_path = h2o.save_model(model=best_model, path=CURRENT_PATH)

        os.renames(temp_exported_model_path, exported_model_path)
        print(model.leaderboard)

    imported_model = h2o.load_model(exported_model_path)

    if task is MachineLearningTasksEnum.classification:
        train_roc_auc_value = round(imported_model.auc(train=True), 3)
        test_roc_auc_value = round(imported_model.auc(valid=True), 3)

        metrics = {'H2O_ROC_AUC_train': train_roc_auc_value, 'H2O_ROC_AUC_test': test_roc_auc_value}

        print(f"H2O_ROC_AUC_train: {metrics['H2O_ROC_AUC_train']}")
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

    h2o_config = get_models_hyperparameters()['H2O']

    run_h2o(train_file, test_file, config_data=h2o_config, target_name='default',
            task=MachineLearningTasksEnum.classification)
