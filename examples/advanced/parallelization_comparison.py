import json
import timeit

import pandas as pd
from matplotlib import pyplot as plt, cm, colors

from fedot.api.main import Fedot
from fedot.core.utils import fedot_project_root


def count_pipelines(opt_history):
    opt_history = json.loads(opt_history)
    count = 0
    for i in range(len(opt_history['individuals'])):
        count += len(opt_history['individuals'][i])
    return count


def run_experiments(timeout: float = None, partitions_n=10, n_jobs=-1):
    """
    Performs experiment to show how much better to use multiprocessing mode in FEDOT

    :param timeout: timeout for optimization
    :param partitions_n: on how many folds you want. f.e. if dataset contains 20000 rows, partition_n=5 will create
    such folds: [4000 rows, 8000 rows, 1200 rows, 16000 rows, 20000 rows]
    :param n_jobs: how many processors you want to use in a multiprocessing mode

    """
    train_data_path = f'{fedot_project_root()}/cases/data/scoring/scoring_train.csv'
    test_data_path = f'{fedot_project_root()}/cases/data/scoring/scoring_test.csv'

    problem = 'classification'

    train_data = pd.read_csv(train_data_path)

    data_len = len(train_data)

    partitions = []
    for i in range(1, partitions_n + 1):
        partitions.append(int(data_len * (i / partitions_n)))

    pipelines_count = {1: [], n_jobs: []}
    times = {1: [], n_jobs: []}

    for partition in partitions:
        for _n_jobs in [1, n_jobs]:
            print(f'n_jobs: {_n_jobs}, {partition} rows in dataset')
            train_data = train_data.iloc[:partition, :]
            start_time = timeit.default_timer()
            auto_model = Fedot(problem=problem, seed=42, timeout=timeout, n_jobs=_n_jobs,
                               composer_params={'with_tuning': False}, preset='fast_train',
                               verbose_level=4)
            auto_model.fit(features=train_data_path, target='target')
            auto_model.predict_proba(features=test_data_path)
            c_pipelines = count_pipelines(auto_model.history.save())
            pipelines_count[_n_jobs].append(c_pipelines)
            times[_n_jobs].append((timeit.default_timer() - start_time) / 60)
            print(f'Count of pipelines: {c_pipelines}')

    plt.title('Comparison parallel mode with a single mode')
    plt.xlabel('rows in train dataset')
    plt.ylabel('Num of pipelines that were evaluated correctly')
    c_norm = colors.Normalize(vmin=timeout - timeout / 2, vmax=timeout + timeout / 2)
    plt.plot(partitions, pipelines_count[1], label='one process', zorder=1)
    plt.scatter(partitions, pipelines_count[1], c=times[1],
                cmap=cm.get_cmap('cool'), norm=c_norm, zorder=2)
    plt.plot(partitions, pipelines_count[n_jobs], label=f'{n_jobs} processes', zorder=1)
    plt.scatter(partitions, pipelines_count[n_jobs], c=times[n_jobs],
                cmap=cm.get_cmap('cool'), norm=c_norm, zorder=2)
    print(times)
    cb = plt.colorbar(cm.ScalarMappable(norm=c_norm, cmap=cm.get_cmap('cool')))
    cb.ax.set_ylabel('time for optimization in minutes', rotation=90)
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    run_experiments(timeout=2, partitions_n=5, n_jobs=-1)
