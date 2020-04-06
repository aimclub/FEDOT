"""
Before running this script make sure you have Java Development Kit installed.
Link to download: https://www.oracle.com/java/technologies/javase-jdk8-downloads.html
"""

import os

import h2o
from h2o.automl import H2OAutoML

from benchmark.benchmark_utils import get_scoring_case_data_paths

MAX_MODELS = 5
CURRENT_PATH = str(os.path.dirname(__file__))
IP = '127.0.0.1'
PORT = 8888


def run_h2o(train_file_path: str, test_file_path: str, case_name='h2o_default'):
    h2o.init(ip=IP, port=PORT)

    # Data preprocessing
    target_column_name = 'default'
    train_frame = h2o.import_file(train_file_path)
    test_frame = h2o.import_file(test_file_path)
    predictors = train_frame.columns.remove(target_column_name)
    train_frame[target_column_name] = train_frame[target_column_name].asfactor()
    test_frame[target_column_name] = test_frame[target_column_name].asfactor()

    result_filename = f"{case_name}_m{MAX_MODELS}"
    exported_model_path = os.path.join(CURRENT_PATH, result_filename)

    if result_filename not in os.listdir(CURRENT_PATH):
        model = H2OAutoML(max_models=MAX_MODELS, seed=1)
        model.train(x=predictors, y=target_column_name, training_frame=train_frame, validation_frame=test_frame)
        best_model = model.leader
        temp_exported_model_path = h2o.save_model(model=best_model, path=CURRENT_PATH)

        os.renames(temp_exported_model_path, exported_model_path)

    imported_model = h2o.load_model(exported_model_path)

    train_roc_auc_value = imported_model.auc(train=True)
    test_roc_auc_value = imported_model.auc(valid=True)

    print(f'H2O_ROC_AUC_train: {train_roc_auc_value}')
    print(f'H2O_ROC_AUC_test: {test_roc_auc_value}')


if __name__ == '__main__':
    train_file, test_file = get_scoring_case_data_paths()
    run_h2o(train_file, test_file)
