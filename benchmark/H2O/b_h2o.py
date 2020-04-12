"""
Before running this script make sure you have Java Development Kit installed.
Link to download: https://www.oracle.com/java/technologies/javase-jdk8-downloads.html
"""

import os

import h2o
from h2o.automl import H2OAutoML

from benchmark.benchmark_utils import get_scoring_case_data_paths

MAX_MODELS = 20
CURRENT_PATH = str(os.path.dirname(__file__))
IP = '127.0.0.1'
PORT = 8888

# MAX_RUNTIME_SECS should be equivalent to MAX_RUNTIME_MINS in b_tpot.py
MAX_RUNTIME_SECS = 1800


def run_h2o(train_file_path: str, test_file_path: str, case_name='h2o_default', is_classification=True):
    h2o.init(ip=IP, port=PORT)

    task = 'clss' if is_classification else 'reg'

    # Data preprocessing
    target_column_name = 'default'
    train_frame = h2o.import_file(train_file_path)
    test_frame = h2o.import_file(test_file_path)
    predictors = train_frame.columns.remove(target_column_name)
    train_frame[target_column_name] = train_frame[target_column_name].asfactor()
    test_frame[target_column_name] = test_frame[target_column_name].asfactor()

    result_filename = f"{case_name}_m{MAX_MODELS}_rs{MAX_RUNTIME_SECS}_{task}"
    exported_model_path = os.path.join(CURRENT_PATH, result_filename)

    if result_filename not in os.listdir(CURRENT_PATH):
        model = H2OAutoML(max_models=MAX_MODELS, seed=1, max_runtime_secs=MAX_RUNTIME_SECS)
        model.train(x=predictors, y=target_column_name, training_frame=train_frame, validation_frame=test_frame)
        best_model = model.leader
        temp_exported_model_path = h2o.save_model(model=best_model, path=CURRENT_PATH)

        os.renames(temp_exported_model_path, exported_model_path)

    imported_model = h2o.load_model(exported_model_path)

    if is_classification:
        train_roc_auc_value = imported_model.auc(train=True)
        test_roc_auc_value = imported_model.auc(valid=True)

        metrics = (f'H2O_ROC_AUC_train: {train_roc_auc_value}', f'H2O_ROC_AUC_test: {test_roc_auc_value}')

        print(metrics[0])
        print(metrics[1])
    else:
        mse_train = imported_model.mse()
        rmse_train = imported_model.rmse()

        metrics = (f'H2O_MSE_train:{mse_train}', f'H2O_RMSE_train: {rmse_train}')

        print(metrics[0])
        print(metrics[1])

    return metrics


if __name__ == '__main__':
    train_file, test_file = get_scoring_case_data_paths()
    run_h2o(train_file, test_file)
