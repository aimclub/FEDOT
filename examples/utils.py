from core.utils import project_root
from sklearn.model_selection import train_test_split
import pandas as pd
import os


def get_multi_clf_data_paths(file_path: str, t_size: float = 0.2, name_of_dataset: str = 'Example',
                             return_df: bool = False):
    main_dir = os.path.join(str(project_root()), 'cases', 'data')
    dataset_dir = os.path.join(str(project_root()), 'cases', 'data', 'multiclf')
    if not os.path.exists(main_dir):
        os.mkdir(main_dir)
    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)

    df = pd.read_excel(file_path)
    if return_df:
        path = os.path.join('cases', 'data', 'multiclf', name_of_dataset)
        full_file_path = os.path.join(str(project_root()), path)
        df.to_csv(full_file_path, sep=',')
        return df, full_file_path
    else:
        train, test = train_test_split(df.iloc[:, :], test_size=t_size, random_state=42)
        train_file_path = os.path.join('cases', 'data', 'multiclf', 'multi_train.csv')
        full_train_file_path = os.path.join(str(project_root()), train_file_path)
        test_file_path = os.path.join('cases', 'data', 'multiclf', 'multi_test.csv')
        full_test_file_path = os.path.join(str(project_root()), test_file_path)
        train.to_csv(full_train_file_path, sep=',')
        test.to_csv(full_test_file_path, sep=',')
        return full_train_file_path, full_test_file_path
