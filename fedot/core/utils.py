import math
import os
import platform
import random
import tempfile
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from golem.utilities.random import RandomStateHandler
from sklearn.model_selection import train_test_split

DEFAULT_PARAMS_STUB = 'default_params'
NESTED_PARAMS_LABEL = 'nested_space'


def fedot_project_root() -> Path:
    """Returns FEDOT project root folder."""
    return Path(__file__).parent.parent.parent


def default_fedot_data_dir() -> str:
    """ Returns the folder where all the output data
    is recorded to. Default: home/FEDOT
    """
    temp_dir = Path("/tmp" if platform.system() == "Darwin" else tempfile.gettempdir())
    default_data_path = os.path.join(temp_dir, 'FEDOT')

    if 'FEDOT' not in os.listdir(temp_dir):
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
        list_with_labels.append(list_with_probs.argmax())

    return np.asarray(list_with_labels).reshape((-1, 1))


def split_data(df: pd.DataFrame, t_size: float = 0.2):
    """ Split pandas DataFrame into train and test parts """
    train, test = train_test_split(df.iloc[:, :], test_size=t_size, random_state=42)
    return train, test


def save_file_to_csv(df: pd.DataFrame, path_to_save: str):
    return df.to_csv(path_to_save, sep=',')


def get_split_data_paths(directory_names: list):
    train_file_path = os.path.join(directory_names[0], directory_names[1], directory_names[2], 'train.csv')
    full_train_file_path = os.path.join(str(fedot_project_root()), train_file_path)
    test_file_path = os.path.join(directory_names[0], directory_names[1], directory_names[2], 'test.csv')
    full_test_file_path = os.path.join(str(fedot_project_root()), test_file_path)
    return full_train_file_path, full_test_file_path


def ensure_directory_exists(dir_names: list):
    main_dir = os.path.join(str(fedot_project_root()), dir_names[0], dir_names[1])
    dataset_dir = os.path.join(str(fedot_project_root()), dir_names[0], dir_names[1], dir_names[2])
    if not os.path.exists(main_dir):
        os.mkdir(main_dir)
    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)


def set_random_seed(seed: Optional[int]):
    """ Sets random seed for evaluation of models"""
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        RandomStateHandler.MODEL_FITTING_SEED = seed


def df_to_html(df: pd.DataFrame, save_path: Union[str, os.PathLike], name: str = 'table', caption: str = ''):
    '''
    Makes html table out of DataFrame look like csv-table.
    Requires BeatifulSoup to be installed.

    Args:
        df: the pd.DataFrame to convert to html
        save_path: where to save the output
        name: output table identificator
        caption: caption above the output table
    '''
    from bs4 import BeautifulSoup

    df_styler = df.round(3).style.highlight_max(props='color: blue; font-weight: bold;', axis=1)
    df_styler.format(precision=3)
    if caption:
        df_styler.set_caption(caption)
    df_styler.set_table_attributes((
        'style="width: 100%; border-collapse: collapse;'
        'font-family: Lato,proxima-nova,Helvetica Neue,Arial,sans-serif;"'
    ))
    df_styler.set_table_styles([
        {'selector': 'table, th, td',
         'props': 'border: 1px solid #e1e4e5; text-align: center; font-size: .9rem;'},
        {'selector': 'th, td',
         'props': 'padding: 8px 16px;'},
        {'selector': 'tr',
         'props': 'background-color: #fff;'},
        {'selector': 'tbody tr:nth-child(odd)',
         'props': 'background-color: #f3f6f6;'}
    ])
    file = Path(save_path)
    df_styler.to_html(file, table_uuid=name)

    doc = BeautifulSoup(file.read_text(), 'html.parser')
    table = doc.find('table')
    if table.parent.name != 'div':
        table = table.wrap(doc.new_tag('div', style='overflow: auto;'))
        file.write_text(doc.prettify())


def convert_memory_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    digit_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    integer_size_value = int(math.floor(math.log(size_bytes, 1024)))
    byte_digit = math.pow(1024, integer_size_value)
    size_in_digit_name = round(size_bytes / byte_digit, 2)
    return "%s %s" % (size_in_digit_name, digit_name[integer_size_value])