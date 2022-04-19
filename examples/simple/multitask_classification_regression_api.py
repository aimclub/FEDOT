import os

import numpy as np
import pandas as pd

from fedot.api.main import Fedot
from test.unit.api.test_api_cli_params import project_root_path


def load_train_test_dataframes() -> (pd.DataFrame, pd.DataFrame):
    """ Load data for multitask regression / classification problem """
    data_path = os.path.join(project_root_path, 'examples/data')
    train_df = pd.read_csv(os.path.join(data_path, 'train_synthetic_regression_classification.csv'))
    test_df = pd.read_csv(os.path.join(data_path, 'test_synthetic_regression_classification.csv'))

    # Prepare numpy arrays into dictionaries
    # For regression and classification features are the same
    mm_train_features = {'regression': np.array(train_df[['feature_1', 'feature_2']]),
                         'classification': np.array(train_df[['feature_1', 'feature_2']])}
    # Target is different for train
    mm_train_target = {'regression': np.array(train_df['concentration']),
                       'classification': np.array(train_df['class'])}
    # For test features the same
    mm_test_features = {'regression': np.array(test_df[['feature_1', 'feature_2']]),
                        'classification': np.array(test_df[['feature_1', 'feature_2']])}
    return mm_train_features, mm_train_target, mm_test_features


def launch_multitask_api_example():
    """
    Demonstration of an example with running a multitask pipeline composing procedure through FEDOT API.
    Synthetic data is used. Task: predict the category of the substance (column "class") <- classification,
    and then predict the concentration based on the predicted category (column "concentration") <- regression.
    """
    train_features, train_target, test_features = load_train_test_dataframes()

    # The priority of the task is determined by the order. So, main task is regression
    problem = 'regression/classification'

    # TODO finish this example - it is not working now and represents the desired interface for multitask
    model = Fedot(problem=problem, timeout=5)
    model.fit(features=train_features,
              target=train_target)

    predicted_concentrations, predicted_classes = model.predict(features=test_features)
    print(f'Predicted classes: {predicted_classes}')
    print(f'Predicted concentrations: {predicted_concentrations}')


if __name__ == '__main__':
    launch_multitask_api_example()
