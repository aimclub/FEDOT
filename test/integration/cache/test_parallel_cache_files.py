import logging
import multiprocessing
from pathlib import Path

from fedot.api.main import Fedot
from fedot.core.utils import default_fedot_data_dir


def run():
    problem = 'ts_forecasting'
    Fedot(problem=problem, timeout=1.,
          logging_level=logging.CRITICAL, show_progress=False)


def test_parallel_cache_files():
    test_file_1 = Path(default_fedot_data_dir(), 'cache_1.operations_db')
    test_file_1.touch()
    test_file_2 = Path(default_fedot_data_dir(), 'cache_2.preprocessors_db')
    test_file_2.touch()

    cpus = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=cpus) as pool:
        pool.starmap(run, [()] * 10)

    assert not test_file_1.exists()
    assert not test_file_2.exists()
