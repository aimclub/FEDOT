import sqlite3
from pathlib import Path

import psutil
from joblib import Parallel, cpu_count, delayed

from examples.simple.classification.api_classification import run_classification_example
from examples.simple.regression.api_regression import run_regression_example
from examples.simple.time_series_forecasting.api_forecasting import run_ts_forecasting_example
from fedot.core.utils import default_fedot_data_dir


def get_unused_pid() -> int:
    busy_pids = set(psutil.pids())
    for test_pid in range(1, 10000):
        if test_pid not in busy_pids:
            return test_pid
    return -1


def test_parallel_cache_files():
    # all files cache files in test dir must be removed
    # if `cache_dir` api param wasn't specified explicitly
    unused_test_pid = get_unused_pid()
    test_file_1 = Path(default_fedot_data_dir(), f'cache_{unused_test_pid}.operations_db')
    test_file_1.touch()
    test_file_2 = Path(default_fedot_data_dir(), f'cache_{unused_test_pid}.preprocessors_db')
    test_file_2.touch()

    tasks = [
        delayed(run_regression_example)(with_tuning=False),
        delayed(run_classification_example)(timeout=1., with_tuning=False),
        delayed(run_ts_forecasting_example)(dataset='beer', horizon=10, timeout=1., with_tuning=False),
    ]

    cpus = cpu_count()
    if cpus > 1:
        try:
            Parallel(n_jobs=cpus)(tasks)
        except sqlite3.OperationalError:
            assert False, 'DBs collides'
        assert not test_file_1.exists()
        assert not test_file_2.exists()
