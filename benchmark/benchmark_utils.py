import gc
import json
from sklearn.model_selection import train_test_split
import pandas as pd
from pmlb import fetch_data
import os
from core.utils import project_root
from typing import Tuple

def get_scoring_case_data_paths():
    train_file_path = os.path.join('cases', 'data', 'scoring', 'scoring_train.csv')
    test_file_path = os.path.join('cases', 'data', 'scoring', 'scoring_test.csv')
    full_train_file_path = os.path.join(str(project_root()), train_file_path)
    full_test_file_path = os.path.join(str(project_root()), test_file_path)

    return full_train_file_path, full_test_file_path


def get_cancer_case_data_paths() -> Tuple[str, str]:
    train_file_path = os.path.join('cases', 'data', 'benchmark', 'cancer_train.csv')
    test_file_path = os.path.join('cases', 'data', 'benchmark', 'cancer_test.csv')
    full_train_file_path = os.path.join(str(project_root()), train_file_path)
    full_test_file_path = os.path.join(str(project_root()), test_file_path)

    return full_train_file_path, full_test_file_path


def init_penn_data_paths(name_of_dataset: str):
    tmp_dir = os.path.join(str(project_root()), 'cases', 'data', 'penn', str(name_of_dataset))
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    else:
        print('Dataset already exist')


def get_penn_case_data_paths(name_of_dataset: str, t_size: float = 0.2) -> str:
    df = fetch_data(name_of_dataset)
    penn_train, penn_test = train_test_split(df.iloc[:, :], test_size=t_size, random_state=42)
    init_penn_data_paths(name_of_dataset)
    train_file_path = os.path.join('cases', 'data', 'penn', str(name_of_dataset), 'penn_train.csv')
    test_file_path = os.path.join('cases', 'data', 'penn', str(name_of_dataset), 'penn_test.csv')
    full_train_file_path = os.path.join(str(project_root()), train_file_path)
    full_test_file_path = os.path.join(str(project_root()), test_file_path)
    penn_train.to_csv(full_train_file_path, sep=',')
    penn_test.to_csv(full_test_file_path, sep=',')
    return full_train_file_path, full_test_file_path


def save_metrics_result_file(data: dict, file_name: str):
    with open(f'{file_name}.json', 'w') as file:
        json.dump(data, file, indent=4)


def get_models_hyperparameters(timedelta: int = 30) -> dict:
    # MAX_RUNTIME_MINS should be equivalent to MAX_RUNTIME_SECS

    tpot_config = {'MAX_RUNTIME_MINS': timedelta,
                   'GENERATIONS': 50,
                   'POPULATION_SIZE': 10
                   }

    h2o_config = {'MAX_MODELS': 20,
                  'MAX_RUNTIME_SECS': timedelta * 60}

    autokeras_config = {'MAX_TRIAL': 10,
                        'EPOCH': 100}

    space_for_mlbox = {

        'ne__numerical_strategy': {"space": [0, 'mean']},

        'ce__strategy': {"space": ["label_encoding", "random_projection", "entity_embedding"]},

        'fs__strategy': {"space": ["variance", "rf_feature_importance"]},
        'fs__threshold': {"search": "choice", "space": [0.1, 0.2, 0.3, 0.4, 0.5]},

        'est__strategy': {"space": ["LightGBM"]},
        'est__max_depth': {"search": "choice", "space": [5, 6]},
        'est__subsample': {"search": "uniform", "space": [0.6, 0.9]},
        'est__learning_rate': {"search": "choice", "space": [0.07]}

    }

    mlbox_config = {'space': space_for_mlbox, 'max_evals': 40}

    config_dictionary = {'TPOT': tpot_config, 'H2O': h2o_config,
                         'autokeras': autokeras_config, 'MLBox': mlbox_config}
    gc.collect()

    return config_dictionary


def get_target_name(file_path: str) -> str:
    print('Make sure that your dataset target column is the last one')
    dataframe = pd.read_csv(file_path)
    column_names = dataframe.columns()
    target_name = column_names[-1]

    return target_name


def get_h2o_connect_config():
    IP = '127.0.0.1'
    PORT = 8888
    return IP, PORT
