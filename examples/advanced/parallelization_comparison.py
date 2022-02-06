import json

import pandas as pd
from matplotlib import pyplot as plt

from fedot.api.main import Fedot
from fedot.core.utils import fedot_project_root


def count_pipelines(opt_history):
    opt_history = json.loads(opt_history)
    count = 0
    for i in range(len(opt_history['individuals'])):
        count += len(opt_history['individuals'][i])
    return count


def run_experiments(timeout: float = None, partitions_n=10, n_n_jobs=-1):
    train_data_path = f'{fedot_project_root()}/cases/data/scoring/scoring_train.csv'
    test_data_path = f'{fedot_project_root()}/cases/data/scoring/scoring_test.csv'

    problem = 'classification'

    train_data = pd.read_csv(train_data_path)

    data_len = len(train_data)

    partitions = []
    for i in range(1, partitions_n + 1):
        partitions.append(int(data_len * (i / partitions_n)))

    pipelines_count = {1: [], n_n_jobs: []}
    for n_worker in [1, n_n_jobs]:
        for partition in partitions:
            print(f'n_jobs: {n_worker}, {partition} rows in dataset')
            train_data = train_data.iloc[:partition, :]
            auto_model = Fedot(problem=problem, seed=42, timeout=timeout, n_jobs=n_worker,
                               composer_params={'with_tuning': False}, preset='fast_train')
            auto_model.fit(features=train_data_path, target='target')
            auto_model.predict_proba(features=test_data_path)
            c_pipelines = count_pipelines(auto_model.history.save())
            pipelines_count[n_worker].append(c_pipelines)
            print(f'Count of pipelines: {c_pipelines}')

    plt.title('Num of pipelines that were evaluated correctly')
    plt.xlabel = 'rows in train dataset'
    plt.ylabel = 'Num of pipelines'
    plt.plot(partitions, pipelines_count[1], label='one process')
    plt.plot(partitions, pipelines_count[n_n_jobs], label=f'{n_n_jobs} processes')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    run_experiments(timeout=10, partitions_n=5, n_n_jobs=-1)
