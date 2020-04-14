import os

from core.utils import project_root
import json


def get_scoring_case_data_paths():
    train_file_path = 'cases/data/scoring/scoring_train.csv'
    test_file_path = 'cases/data/scoring/scoring_test.csv'
    full_train_file_path = os.path.join(str(project_root()), train_file_path)
    full_test_file_path = os.path.join(str(project_root()), test_file_path)

    return full_train_file_path, full_test_file_path


def save_metrics_result_file(data: dict, file_name: str):
    with open(f'{file_name}.json', 'w') as file:
        json.dump(data, file, indent=4)


def get_models_hyperparameters():
    # MAX_RUNTIME_MINS should be equivalent to MAX_RUNTIME_SECS

    tpot_config = {'MAX_RUNTIME_MINS': 30,
                   'GENERATIONS': 10,
                   'POPULATION_SIZE': 5
                   }

    h2o_config = {'MAX_MODELS': 10,
                  'MAX_RUNTIME_SECS': 1800}

    autokeras_config = {'MAX_TRIAL': 10,
                        'EPOCH': 10}

    config_dictionary = {'TPOT': tpot_config, 'H2O': h2o_config, 'AutoKeras': autokeras_config}

    print(config_dictionary)

    return config_dictionary


def get_cancer_case_data_paths():
    train_file_path = 'cases/data/benchmark/cancer_train.csv'
    test_file_path = 'cases/data/benchmark/cancer_test.csv'
    full_train_file_path = os.path.join(str(project_root()), train_file_path)
    full_test_file_path = os.path.join(str(project_root()), test_file_path)

    return full_train_file_path, full_test_file_path
