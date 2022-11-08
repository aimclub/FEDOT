import os
import shutil

import pytest

from cases.credit_scoring.credit_scoring_problem import get_scoring_data, run_credit_scoring_problem
from fedot.utilities.profiler.automl_memory_profiler import MemoryProfiler
from fedot.utilities.profiler.time_profiler import TimeProfiler


@pytest.fixture(scope='session', autouse=True)
def preprocessing_files_before_and_after_tests(request):
    path = ['time_profiler', 'memory_profiler']

    delete_files = create_func_delete_files(path)
    request.addfinalizer(delete_files)


def create_func_delete_files(paths):
    """
    Create function to delete files that created after tests.
    """

    def wrapper():
        for path in paths:
            if os.path.isdir(path) or path.endswith('.log'):
                shutil.rmtree(path)

    return wrapper


def test_time_profiler_correctly():
    """
    Profilers requirements are needed
    """
    profiler = TimeProfiler()
    full_path_train, full_path_test = get_scoring_data()
    run_credit_scoring_problem(full_path_train, full_path_test,
                               timeout=5)
    path = os.path.abspath('time_profiler')
    profiler.profile(path=path, node_percent=0.5, edge_percent=0.1, open_web=False)

    assert os.path.exists(path)


def test_memory_profiler_correctly():
    """
    Profilers requirements are needed
    """

    path = os.path.abspath('memory_profiler')
    full_path_train, full_path_test = get_scoring_data()
    arguments = {'train_file_path': full_path_train, 'test_file_path': full_path_test,
                 'timeout': 1.5}
    MemoryProfiler(run_credit_scoring_problem, kwargs=arguments,
                   path=path, roots=[run_credit_scoring_problem], max_depth=8)

    assert os.path.exists(path)
