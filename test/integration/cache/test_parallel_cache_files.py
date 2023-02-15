import multiprocessing
import sqlite3
from functools import partial
from pathlib import Path
from typing import Callable

from examples.simple.classification.api_classification import run_classification_example
from examples.simple.classification.classification_pipelines import classification_random_forest_pipeline
from examples.simple.classification.classification_with_tuning import run_classification_tuning_experiment
from examples.simple.regression.api_regression import run_regression_example
from examples.simple.regression.regression_pipelines import regression_ransac_pipeline
from examples.simple.regression.regression_with_tuning import run_experiment as run_regression_tuning
from examples.simple.time_series_forecasting.api_forecasting import run_ts_forecasting_example
from examples.simple.time_series_forecasting.ts_pipelines import ts_locf_ridge_pipeline
from examples.simple.time_series_forecasting.tuning_pipelines import run_experiment as run_ts_tuning
from fedot.core.pipelines.tuning.sequential import SequentialTuner
from fedot.core.pipelines.tuning.unified import PipelineTuner
from fedot.core.utils import default_fedot_data_dir


def run_example(target: Callable):
    target()


def test_parallel_cache_files():
    test_file_1 = Path(default_fedot_data_dir(), 'cache_1.operations_db')
    test_file_1.touch()
    test_file_2 = Path(default_fedot_data_dir(), 'cache_2.preprocessors_db')
    test_file_2.touch()
    tasks = [
        partial(run_regression_tuning, regression_ransac_pipeline(), tuner=None),
        partial(run_regression_example, with_tuning=False),
        partial(run_classification_tuning_experiment, classification_random_forest_pipeline(), tuner=None),
        partial(run_classification_example, timeout=2., with_tuning=False),
        partial(run_ts_forecasting_example, dataset='beer', horizon=10, timeout=2., with_tuning=False),
        partial(run_ts_tuning, 'australia', ts_locf_ridge_pipeline(), len_forecast=50, tuning=False)
    ]

    cpus = multiprocessing.cpu_count()
    if cpus > 1:
        try:
            with multiprocessing.Pool(processes=cpus) as pool:
                list(pool.imap(run_example, tasks))
        except sqlite3.OperationalError:
            assert False, 'DBs collides'

        assert not test_file_1.exists()
        assert not test_file_2.exists()
