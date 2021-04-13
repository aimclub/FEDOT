import os
from enum import Enum
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def project_root() -> Path:
    """Returns project root folder."""
    return Path(__file__).parent.parent.parent


def default_fedot_data_dir() -> str:
    """ Returns the folder where all the output data
    is recorded to. Default: home/Fedot
    """
    default_data_path = os.path.join(str(Path.home()), 'Fedot')
    if 'Fedot' not in os.listdir(str(Path.home())):
        os.mkdir(default_data_path)
    return default_data_path


def labels_to_dummy_probs(prediction: np.array):
    """ Returns converted predictions using one-hot probability encoding """
    df = pd.Series(prediction)
    pred_probas = pd.get_dummies(df).values
    return pred_probas


def probs_to_labels(prediction: np.array):
    """ Converts predicted probabilities into labels """
    list_with_labels = []
    for list_with_probs in prediction:
        list_with_labels.append(list_with_probs.argmax() + 1.0)

    return list_with_labels


def split_data(df: pd.DataFrame, t_size: float = 0.2):
    """ Split pandas DataFrame into train and test parts """
    train, test = train_test_split(df.iloc[:, :], test_size=t_size, random_state=42)
    return train, test


def save_file_to_csv(df: pd.DataFrame, path_to_save: str):
    return df.to_csv(path_to_save, sep=',')


def get_split_data_paths(directory_names: list):
    train_file_path = os.path.join(directory_names[0], directory_names[1], directory_names[2], 'train.csv')
    full_train_file_path = os.path.join(str(project_root()), train_file_path)
    test_file_path = os.path.join(directory_names[0], directory_names[1], directory_names[2], 'test.csv')
    full_test_file_path = os.path.join(str(project_root()), test_file_path)
    return full_train_file_path, full_test_file_path


def ensure_directory_exists(dir_names: list):
    main_dir = os.path.join(str(project_root()), dir_names[0], dir_names[1])
    dataset_dir = os.path.join(str(project_root()), dir_names[0], dir_names[1], dir_names[2])
    if not os.path.exists(main_dir):
        os.mkdir(main_dir)
    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)


def make_chain_generator(chain):
    visited_nodes = []

    for node in chain.nodes:
        if node not in visited_nodes:
            visited_nodes.append(node)
            yield node


class ComparableEnum(Enum):
    """
    The Enum implementation that allows to avoid the multi-module enum comparison problem
    (https://stackoverflow.com/questions/26589805/python-enums-across-modules)
    """

    def __eq__(self, other):
        return str(self) == str(other)

    def __hash__(self):
        return hash(str(self))
