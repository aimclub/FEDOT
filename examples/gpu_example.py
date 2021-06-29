import os
import sys

curdir = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.join(curdir, '..'))
ROOT = os.path.abspath(os.curdir)
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "fedot"))

import numpy as np
from sklearn.datasets import load_iris

from fedot.api.main import Fedot
from fedot.core.utils import fedot_project_root


def run_small_gpu_example():
    synthetic_data = load_iris()
    features = np.asarray(synthetic_data.data).astype(np.float32)
    features_test = np.asarray(synthetic_data.data).astype(np.float32)
    target = synthetic_data.target

    problem = 'classification'

    baseline_model = Fedot(problem=problem, preset='gpu')
    baseline_model.fit(features=features_test, target=target, predefined_model='rf')

    baseline_model.predict(features=features)
    print(baseline_model.get_metrics())


def run_large_gpu_example():
    # train-test swapped to avoid out-of-memory-error
    train_data_path = f'{fedot_project_root()}/cases/data/scoring/scoring_test.csv'
    test_data_path = f'{fedot_project_root()}/cases/data/scoring/scoring_train.csv'

    problem = 'classification'

    baseline_model = Fedot(problem=problem, preset='gpu')
    baseline_model.fit(features=train_data_path, target='target', predefined_model='logit')

    baseline_model.predict(features=test_data_path)
    print(baseline_model.get_metrics())


if __name__ == '__main__':
    run_small_gpu_example()
    run_large_gpu_example()
