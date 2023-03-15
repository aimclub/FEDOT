import pytest
import pickle
from typing import List

from fedot.core.utils import fedot_project_root
from fedot.core.pipelines.prediction_intervals.utils import pipeline_simple_structure
from fedot.core.pipelines.prediction_intervals.ts_mutation import get_ts_mutation, get_different_mutations


@pytest.fixture
def get_individual():

    model_name = f'{fedot_project_root()}/test/unit/data/pred_ints_model_test.pickle'
    with open(model_name, 'rb') as f:
        model = pickle.load(f)

    return model.history.individuals[-1][0]


def check_uniqueness_mutations_structures(a: List[list]):
    ans = True
    for x in a:
        a.remove(x)
        if x in a:
            ans = False
            break
    return ans


def test_get_ts_mutation(get_individual):
    for i in range(20):
        assert type(get_ts_mutation(get_individual)) == type(get_individual), f"mutation {i+1} of failed"


def test_get_different_mutations(get_individual):
    mutations = get_different_mutations(get_individual, 15)
    mutations_structure = list(map(lambda x: pipeline_simple_structure(x), mutations))

    assert check_uniqueness_mutations_structures(mutations_structure), "Some mutations have identical structure."
