"""
Before running this script make sure you have Java Development Kit installed.
Link to download: https://www.oracle.com/java/technologies/javase-jdk8-downloads.html
"""

import os

import h2o

from benchmark.benchmark_utils import (get_h2o_connect_config, get_models_hyperparameters)
from core.models.data import InputData
from core.models.evaluation.automl_eval import fit_h2o, predict_h2o

CURRENT_PATH = str(os.path.dirname(__file__))


def run_h2o(params: 'ExecutionParams'):
    train_file_path = params.train_file
    test_file_path = params.test_file
    case_label = params.case_label
    task = params.task

    config_data = get_models_hyperparameters()['H2O']
    max_models = config_data['MAX_MODELS']
    max_runtime_secs = config_data['MAX_RUNTIME_SECS']

    result_filename = f'{case_label}_m{max_models}_rs{max_runtime_secs}_{task.name}'
    exported_model_path = os.path.join(CURRENT_PATH, result_filename)

    # TODO Regression
    if result_filename not in os.listdir(CURRENT_PATH):
        train_data = InputData.from_csv(train_file_path)
        best_model = fit_h2o(train_data, round(max_runtime_secs / 60))
        temp_exported_model_path = h2o.save_model(model=best_model, path=CURRENT_PATH)

        os.renames(temp_exported_model_path, exported_model_path)

    ip, port = get_h2o_connect_config()
    h2o.init(ip=ip, port=port, name='h2o_server')

    imported_model = h2o.load_model(exported_model_path)

    test_frame = InputData.from_csv(test_file_path)
    true_target = test_frame.target

    predicted = predict_h2o(imported_model, test_frame)

    h2o.shutdown(prompt=False)

    return true_target, predicted
