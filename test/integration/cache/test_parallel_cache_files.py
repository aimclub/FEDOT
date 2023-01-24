import multiprocessing
import os
import sqlite3
from functools import partial
from pathlib import Path
from typing import Callable

import psutil
from golem.core.tuning.sequential import SequentialTuner
from golem.core.tuning.simultaneous import SimultaneousTuner

from examples.simple.classification.api_classification import run_classification_example
from examples.simple.classification.classification_pipelines import classification_random_forest_pipeline
from examples.simple.classification.classification_with_tuning import run_classification_tuning_experiment
from examples.simple.regression.api_regression import run_regression_example
from examples.simple.regression.regression_pipelines import regression_ransac_pipeline
from examples.simple.regression.regression_with_tuning import run_experiment as run_regression_tuning
from examples.simple.time_series_forecasting.api_forecasting import run_ts_forecasting_example
from examples.simple.time_series_forecasting.ts_pipelines import ts_locf_ridge_pipeline
from examples.simple.time_series_forecasting.tuning_pipelines import run_experiment as run_ts_tuning
from fedot.core.utils import default_fedot_data_dir


def run_example(target: Callable):
    target()


def get_unused_pid() -> int:
    busy_pids = set(psutil.pids())
    for test_pid in range(1, 10000):
        if test_pid not in busy_pids:
            return test_pid
    return -1


def test_parallel_cache_files():
    # all files cache files in test dir must be removed
    # if `cache_folder` api param wasn't specified explicitly
    unused_test_pid = get_unused_pid()
    test_file_1 = Path(default_fedot_data_dir(), f'cache_{unused_test_pid}.operations_db')
    test_file_1.touch()
    test_file_2 = Path(default_fedot_data_dir(), f'cache_{unused_test_pid}.preprocessors_db')
    test_file_2.touch()

    tasks = [
        partial(run_regression_tuning, regression_ransac_pipeline(), SequentialTuner),
        run_regression_example,
        partial(run_classification_tuning_experiment, classification_random_forest_pipeline(), SimultaneousTuner),
        partial(run_classification_example, timeout=2.),
        partial(run_ts_forecasting_example, dataset='beer', horizon=10, timeout=2.),
        partial(run_ts_tuning, 'australia', ts_locf_ridge_pipeline(), len_forecast=50, tuning=True)
    ]

    cpus = multiprocessing.cpu_count()
    try:
        with multiprocessing.Pool(processes=cpus) as pool:
            list(pool.imap(run_example, tasks))
    except sqlite3.OperationalError:
        assert False, 'DBs collides'

    assert not test_file_1.exists()
    assert not test_file_2.exists()
