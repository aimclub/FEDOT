import gc
import json
import os
from glob import glob
from typing import Tuple, Tuple

import pandas as pd
from pmlb import fetch_data
from sklearn.model_selection import train_test_split

from core.utils import ensure_directory_exists, get_split_data_paths, project_root, project_root, save_file_to_csv, \
    split_data


def get_scoring_case_data_paths() -> Tuple[str, str]:
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


def get_penn_case_data_paths(name_of_dataset: str) -> Tuple[str, str]:
    df = fetch_data(name_of_dataset)
    directory_names = ['benchmark', 'data', name_of_dataset]
    penn_train, penn_test = split_data(df)
    ensure_directory_exists(directory_names)
    full_train_file_path, full_test_file_path = get_split_data_paths(directory_names)
    save_file_to_csv(penn_train, full_train_file_path)
    save_file_to_csv(penn_test, full_test_file_path)
    return full_train_file_path, full_test_file_path


def convert_json_stats_to_csv(dataset: list, include_hyper: bool = True):
    list_of_df = []
    new_col = []
    dataset_name_column_place = 1
    for filename, name_of_dataset in zip(glob('*.json'), dataset):
        with open(filename, 'r') as f:
            data = json.load(f)
            df = pd.json_normalize(data)
            df.insert(dataset_name_column_place, 'name_of_dataset', name_of_dataset, True)
            list_of_df.append(df)

    df_final = pd.concat(list_of_df)

    for column_name in df_final.columns:
        if 'hyper' not in column_name:
            new_col.append(column_name)

    if include_hyper:
        df_final = df_final[new_col]

    pd.DataFrame.to_csv(df_final, './final_combined.csv', sep=',', index=False)
    return df_final


def save_metrics_result_file(data: dict, file_name: str):
    with open(f'{file_name}.json', 'w') as file:
        json.dump(data, file, indent=4)


def get_models_hyperparameters(timedelta: int = 10) -> dict:
    # MAX_RUNTIME_MINS should be equivalent to MAX_RUNTIME_SECS

    tpot_config = {'MAX_RUNTIME_MINS': timedelta,
                   'GENERATIONS': 50,
                   'POPULATION_SIZE': 10
                   }

    fedot_config = {'MAX_RUNTIME_MINS': timedelta,
                    'GENERATIONS': 10,
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

    config_dictionary = {'TPOT': tpot_config, 'FEDOT': fedot_config, 'H2O': h2o_config,
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
