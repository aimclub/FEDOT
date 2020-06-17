import os

import pandas as pd

from core.utils import ensure_directory_exist, get_split_data_paths, project_root, save_file_to_csv, split_data


def create_multi_clf_examples_from_excel(file_path: str, return_df: bool = False):
    df = pd.read_excel(file_path)
    train, test = split_data(df)
    file_dir_name = file_path.replace('.', '/').split('/')[-2]
    file_csv_name = f'{file_dir_name}.csv'
    directory_names = ['examples', 'data', file_dir_name]
    ensure_directory_exist(directory_names)
    if return_df:
        path = os.path.join(directory_names[0], directory_names[1], directory_names[2], file_csv_name)
        full_file_path = os.path.join(str(project_root()), path)
        save_file_to_csv(df, full_file_path)
        return df, full_file_path
    else:
        full_train_file_path, full_test_file_path = get_split_data_paths(directory_names)
        save_file_to_csv(train, full_train_file_path)
        save_file_to_csv(train, full_test_file_path)
        return full_train_file_path, full_test_file_path
