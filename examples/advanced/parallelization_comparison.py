import logging
import operator
import timeit
from functools import reduce
from typing import Optional

import pandas as pd
from matplotlib import cm, colors, pyplot as plt

from fedot.api.main import Fedot
from fedot.core.optimisers.opt_history_objects.opt_history import OptHistory
from fedot.core.utils import fedot_project_root


def _count_pipelines(opt_history: Optional[OptHistory]) -> int:
    if opt_history is not None:
        return reduce(operator.add, map(len, opt_history.individuals), 0)
    return 0


def run_experiments(timeout: float = None, partitions_n=10, n_jobs=-1):
    """
    Performs experiment to show how much better to use multiprocessing mode in FEDOT

    :param timeout: timeout for optimization in minutes
    :param partitions_n: on how many folds you want. f.e. if dataset contains 20000 rows, partitions_n=5 will create
        such folds: [4000 rows, 8000 rows, 12000 rows, 16000 rows, 20000 rows]
    :param n_jobs: how many processors you want to use in a multiprocessing mode

    """
    train_data_path = f'{fedot_project_root()}/cases/data/scoring/scoring_train.csv'

    problem = 'classification'

    train_data = pd.read_csv(train_data_path)

    data_len = len(train_data)

    partitions = [int(data_len * (i / partitions_n)) for i in range(1, partitions_n + 1)]

    pipelines_count, times = [{1: [], n_jobs: []} for _ in range(2)]

    for _n_jobs in [1, n_jobs]:
        print(f'n_jobs: {_n_jobs}')
        for partition in partitions:
            train_data_tmp = train_data.iloc[:partition].copy()
            start_time = timeit.default_timer()
            auto_model = Fedot(problem=problem, seed=42, timeout=timeout,
                               n_jobs=_n_jobs, logging_level=logging.NOTSET,
                               with_tuning=False, preset='fast_train')
            auto_model.fit(features=train_data_tmp, target='target')
            times[_n_jobs].append((timeit.default_timer() - start_time) / 60)
            c_pipelines = _count_pipelines(auto_model.history)
            pipelines_count[_n_jobs].append(c_pipelines)
            print(f'\tDataset length: {partition}, number of pipelines: {c_pipelines}')

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
