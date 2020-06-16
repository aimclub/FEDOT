import os
import gc
import json
from core.utils import project_root
from typing import Tuple
from sklearn.model_selection import train_test_split
import pandas as pd
from pmlb import fetch_data
from glob import glob

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


def init_penn_data_paths(name_of_dataset: str):
    main_dir = os.path.join(str(project_root()), 'benchmark', 'data')
    dataset_dir = os.path.join(str(project_root()), 'benchmark', 'data', str(name_of_dataset))
    if not os.path.exists(main_dir):
        os.mkdir(main_dir)
        
    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)
    else:
        print('Dataset already exist')


def get_penn_case_data_paths(name_of_dataset: str, t_size: float = 0.2) -> Tuple[str, str]:
    df = fetch_data(name_of_dataset)
    penn_train, penn_test = train_test_split(df.iloc[:, :], test_size=t_size, random_state=42)
    init_penn_data_paths(name_of_dataset)
    train_file_path = os.path.join('benchmark', 'data', str(name_of_dataset), 'train.csv')
    test_file_path = os.path.join('benchmark', 'data', str(name_of_dataset), 'test.csv')
    full_train_file_path = os.path.join(str(project_root()), train_file_path)
    full_test_file_path = os.path.join(str(project_root()), test_file_path)
    penn_train.to_csv(full_train_file_path, sep=',')
    penn_test.to_csv(full_test_file_path, sep=',')
    return full_train_file_path, full_test_file_path


def convert_json_to_csv(dataset: list, include_hyper: bool = True):
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
