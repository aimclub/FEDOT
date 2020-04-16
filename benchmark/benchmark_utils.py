import os

from core.utils import project_root
import json

import gc


def get_scoring_case_data_paths():
    train_file_path = os.path.join('cases', 'data', 'scoring', 'scoring_train.csv')
    test_file_path = os.path.join('cases', 'data', 'scoring', 'scoring_test.csv')
    print(train_file_path)
    print(test_file_path)
    full_train_file_path = os.path.join(str(project_root()), train_file_path)
    full_test_file_path = os.path.join(str(project_root()), test_file_path)

    return full_train_file_path, full_test_file_path


def get_cancer_case_data_paths():
    train_file_path = os.path.join('cases', 'data', 'scoring', 'cancer_train.csv')
    test_file_path = os.path.join('cases', 'data', 'scoring', 'cancer_test.csv')
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

    h2o_config = {'MAX_MODELS': 20,
                  'MAX_RUNTIME_SECS': 1800}

    autokeras_config = {'MAX_TRIAL': 1,
                        'EPOCH': 1}

    mlbox_config = {

        'ne__numerical_strategy': {"space": [0, 'mean']},

        'ce__strategy': {"space": ["label_encoding", "random_projection", "entity_embedding"]},

        'fs__strategy': {"space": ["variance", "rf_feature_importance"]},
        'fs__threshold': {"search": "choice", "space": [0.1, 0.2, 0.3, 0.4, 0.5]},

        'est__strategy': {"space": ["LightGBM"]},
        'est__max_depth': {"search": "choice", "space": [5, 6]},
        'est__subsample': {"search": "uniform", "space": [0.6, 0.9]},
        'est__learning_rate': {"search": "choice", "space": [0.07]}

    }

    config_dictionary = {'TPOT': tpot_config, 'H2O': h2o_config, 'AutoKeras': autokeras_config, 'MLBox': mlbox_config}
    gc.collect()

    return config_dictionary
