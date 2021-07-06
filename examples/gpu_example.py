import os
import sys
from datetime import datetime

curdir = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.join(curdir, '..'))
ROOT = os.path.abspath(os.curdir)
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "fedot"))

import numpy as np
from sklearn.datasets import load_iris
from sklearn.datasets import make_classification
from sklearn.datasets import make_moons
from fedot.core.data.data import InputData
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.chains.chain import Chain
from fedot.core.chains.node import PrimaryNode

from fedot.api.main import Fedot
from fedot.core.utils import fedot_project_root


def get_synthetic_input_data(n_samples=10000, n_features=10, random_state=None) -> InputData:
    synthetic_data = make_classification(n_samples=n_samples,
                                         n_features=n_features, random_state=random_state)
    input_data = InputData(idx=np.arange(0, len(synthetic_data[1])),
                           features=synthetic_data[0],
                           target=synthetic_data[1],
                           task=Task(TaskTypesEnum.classification),
                           data_type=DataTypesEnum.table)
    return input_data


def run_small_gpu_example():
    synthetic_data = load_iris()
    features = np.asarray(synthetic_data.data).astype(np.float32)
    features_test = np.asarray(synthetic_data.data).astype(np.float32)
    target = synthetic_data.target

    problem = 'classification'

    baseline_model = Fedot(problem=problem, preset='gpu')
    # baseline_model = Fedot(problem=problem)
    baseline_model.fit(features=features_test, target=target, predefined_model='svc')

    baseline_model.predict(features=features)
    print(baseline_model.get_metrics())


def run_large_gpu_example(n_samples, mode: str = None):
    # train-test swapped to avoid out-of-memory-error
    # train_data_path = f'{fedot_project_root()}/cases/data/scoring/scoring_train.csv'
    # test_data_path = f'{fedot_project_root()}/cases/data/scoring/scoring_test.csv'
    # train_data = get_synthetic_input_data(10000)
    # test_data = get_synthetic_input_data(1000)

    problem = 'classification'
    features, target = make_moons(n_samples=n_samples, shuffle=True, noise=0.1, random_state=137)

    train_data = InputData(features=features, target=target, task=Task(task_type=TaskTypesEnum.classification),
                           idx=len(features), data_type=DataTypesEnum.table)
    test_data = InputData(features=features, target=target, task=Task(task_type=TaskTypesEnum.classification),
                          idx=len(features), data_type=DataTypesEnum.table)

    if mode == 'gpu':
        baseline_model = Fedot(problem=problem, preset='gpu')
    else:
        baseline_model = Fedot(problem=problem)
    start = datetime.now()
    # baseline_model.fit(features=train_data_path, target='target', predefined_model='logit')
    baseline_model.fit(features=train_data, target='target', predefined_model='svc')

    print(f'Completed {n_samples} in: {datetime.now() - start}')
    # baseline_model.predict(features=test_data_path)
    baseline_model.predict(features=test_data)
    print(baseline_model.get_metrics())


def run_large_gpu_example_with_preset(n_samples, mode: str = None):
    # train_data = get_synthetic_input_data(10000)
    # test_data = get_synthetic_input_data(1000)

    problem = 'classification'
    features, target = make_moons(n_samples=n_samples, shuffle=True, noise=0.1, random_state=137)

    train_data = InputData(features=features, target=target, task=Task(task_type=TaskTypesEnum.classification),
                           idx=len(features), data_type=DataTypesEnum.table)
    test_data = InputData(features=features, target=target, task=Task(task_type=TaskTypesEnum.classification),
                          idx=len(features), data_type=DataTypesEnum.table)

    if mode == 'gpu':
        baseline_model = Fedot(problem=problem, preset='gpu')
    else:
        baseline_model = Fedot(problem=problem)
    svc_node_with_custom_params = PrimaryNode('svc')
    svc_node_with_custom_params.custom_params = dict(kernel='rbf', C=10, gamma=1, cache_size=2000, probability=True)
    preset_pipeline = Chain(nodes=[svc_node_with_custom_params])

    start = datetime.now()
    baseline_model.fit(features=train_data, target='target', predefined_model=preset_pipeline)
    print(f'Completed {n_samples} with custom params in: {datetime.now() - start}')

    baseline_model.predict(features=test_data)
    print(baseline_model.get_metrics())


if __name__ == '__main__':
    n_samples = 100000
    run_small_gpu_example()
    run_large_gpu_example(n_samples, mode='gpu')
    run_large_gpu_example_with_preset(n_samples, mode='gpu')
