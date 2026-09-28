import os
from pathlib import Path

PROJECT_PATH = str(Path(__file__).parent.parent.parent.parent.parent)
EXAMPLES_PATH = os.path.join(PROJECT_PATH, 'examples')
EXAMPLES_DATA_PATH = os.path.join(EXAMPLES_PATH, 'data')
BENCHMARK_RESULTS_PATH = os.path.join(PROJECT_PATH, 'benchmark', 'results')

# Default parameters of feature generators
PATH_TO_DEFAULT_PARAMS = os.path.join(
    PROJECT_PATH,
    'fedot/industrial/core/repository/data/default_operation_params.json')

# For results collection
DS_INFO_PATH = os.path.join(
    PROJECT_PATH,
    'fedot',
    'industrial'
    'core',
    'architecture',
    'postprocessing',
    'ucr_datasets.json')

DEFAULT_PATH_RESULTS = os.path.join(PROJECT_PATH, 'results_of_experiments')
