import os
import random

import numpy as np

from cases.credit_scoring.credit_scoring_problem import run_credit_scoring_problem, get_scoring_data
from fedot.utilities.profiler.memory_profiler import MemoryProfiler
from fedot.utilities.profiler.time_profiler import TimeProfiler

random.seed(1)
np.random.seed(1)

if __name__ == '__main__':
    # JUST UNCOMMENT WHAT TYPE OF PROFILER DO YOU NEED
    # EXAMPLE of MemoryProfiler.

    path = os.path.join(os.path.expanduser("~"), 'memory_profiler')
    full_path_train, full_path_test = get_scoring_data()
    arguments = {'train_file_path': full_path_train, 'test_file_path': full_path_test}
    MemoryProfiler(run_credit_scoring_problem, kwargs=arguments, path=path,
                   roots=[run_credit_scoring_problem], max_depth=8)

    # EXAMPLE of TimeProfiler.

    profiler = TimeProfiler()
    full_path_train, full_path_test = get_scoring_data()
    run_credit_scoring_problem(full_path_train, full_path_test)
    path = os.path.join(os.path.expanduser("~"), 'time_profiler')
    profiler.profile(path=path, node_percent=0.5, edge_percent=0.1, open_web=True)
