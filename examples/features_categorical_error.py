import os
import numpy as np
import pandas as pd

from fedot.api.main import Fedot
from fedot.core.utils import fedot_project_root

np.random.seed(42)


def load_data(n_rows=500):
    dataset_path = 'examples/data/sampled_app_train.csv'
    full_path = os.path.join(str(fedot_project_root()), dataset_path)
    data = pd.read_csv(full_path)

    columns = ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'TARGET']
    # Take first n rows
    df_to_process = data[columns].head(n_rows)

    # Convert data into numpy arrays
    features = np.array(df_to_process[columns[:-1]], dtype=str)
    target = np.array(df_to_process[columns[-1]])

    return features, target


def run():
    features, target = load_data()
    auto_model = Fedot(problem='classification', seed=42, verbose_level=4)
    pipeline = auto_model.fit(features=features, target=target)
    prediction = auto_model.predict_proba(features=features)
    auto_metrics = auto_model.get_metrics()


if __name__ == '__main__':
    run()
