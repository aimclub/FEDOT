import platform

import pytest
import pickle
from typing import List
import itertools

from golem.core.optimisers.opt_history_objects.individual import Individual
from golem.core.log import default_log, Log

from fedot.core.utils import fedot_project_root
from fedot.core.pipelines.prediction_intervals.ts_mutation import get_ts_mutation, get_different_mutations
from fedot.core.pipelines.prediction_intervals.utils import get_last_generations

from fedot.core.pipelines.prediction_intervals.graph_distance import get_distance_between
from fedot.core.pipelines.prediction_intervals.params import PredictionIntervalsParams

import pathlib

plt = platform.system()
if plt == 'Linux':
    pathlib.WindowsPath = pathlib.PosixPath


@pytest.fixture
def params():

    model_name = f'{fedot_project_root()}/test/unit/pipelines/prediction_intervals/data/pred_ints_model_test.pickle'
    with open(model_name, 'rb') as f:
        model = pickle.load(f)

    Log().reset_logging_level(10)
    logger = default_log(prefix='PredictionIntervals_test_mutations')

    return {'individual': get_last_generations(model)['final_choice'],
            'operations': PredictionIntervalsParams().mutations_operations,
            'logger': logger}


def check_uniqueness_mutations_structures(a: List[Individual]):
    ans = True
    for x in itertools.combinations(a, 2):
        if get_distance_between(x[0].graph, x[1].graph, compare_node_params=False) == 0:
            ans = False
            break
    return ans


def test_get_ts_mutation(params):
    for i in range(20):
        assert type(get_ts_mutation(individual=params['individual'],
                                    operations=params['operations'])) == Individual, f"mutation {i+1} failed."


def test_get_different_mutations(params):
    mutations = get_different_mutations(individual=params['individual'],
                                        number_mutations=15,
                                        operations=params['operations'],
                                        logger=params['logger'])

    assert check_uniqueness_mutations_structures(mutations), "Some mutations have identical structure."
