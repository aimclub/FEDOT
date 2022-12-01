import json
from copy import deepcopy
from itertools import product

import numpy as np
import pytest

from fedot.core.optimisers.fitness.fitness import *
from fedot.core.optimisers.fitness.multi_objective_fitness import MultiObjFitness
from fedot.core.optimisers.objective.objective import to_fitness
from fedot.core.serializers import Serializer


def get_fitness_objects():
    return [
        null_fitness(),
        SingleObjFitness(None),
        SingleObjFitness(None, 12.34),
        SingleObjFitness(11),
        SingleObjFitness(1, 2, 3, 4),

        MultiObjFitness(),
        MultiObjFitness([1.]),
        MultiObjFitness([1., 2., 3.]),
        MultiObjFitness([1., 2., 3.], weights=-1),
        MultiObjFitness([1., 2., 3.], weights=[-1, -2, -3]),
    ]


@pytest.fixture()
def fitness_objects():
    return get_fitness_objects()


def test_fitness_hash(fitness_objects):
    assert all(hash(f) for f in fitness_objects)
    hashed = set(fitness_objects)
    assert len(hashed) == len(fitness_objects)


@pytest.mark.parametrize('fitness', get_fitness_objects())
def test_fitness_values_property(fitness):
    fitness_values = (3.1415,) * len(fitness.values)

    del fitness.values
    assert not fitness.valid

    fitness.values = fitness_values

    if len(fitness_values) > 0:
        assert fitness.valid
    assert np.array_equal(fitness.values, np.multiply(fitness_values, fitness.weights))


@pytest.mark.parametrize('fitness', get_fitness_objects())
def test_fitness_validity(fitness):
    has_nones = any(value is None for value in fitness.values)
    is_empty = len(fitness.values) == 0
    if has_nones or is_empty:
        assert not fitness.valid
    else:
        assert fitness.valid


def test_fitness_invalid_are_unequal(fitness_objects):
    '''Invalid fitnesses are not equal'''
    for fitness1, fitness2 in product(fitness_objects, repeat=2):
        if not fitness1.valid or not fitness2.valid:
            assert fitness1 != fitness2


def test_fitness_equality(fitness_objects):
    # trivial equality
    for fitness in fitness_objects:
        if fitness.valid:
            assert fitness == deepcopy(fitness)
        else:
            assert fitness != deepcopy(fitness)

    # and some special cases
    assert SingleObjFitness(1., 2., 3. + 1e-12) == SingleObjFitness(1., 2., 3.)
    assert SingleObjFitness(1) != MultiObjFitness([1])
    assert SingleObjFitness(1., 2., 3.) != MultiObjFitness([1., 2., 3.])


def test_fitness_compare_with_null_fitness():
    assert SingleObjFitness(1, 10) > null_fitness()
    assert SingleObjFitness(None, 10) != null_fitness()
    assert MultiObjFitness((1, 123, 123)) > null_fitness()
    assert MultiObjFitness((0, 0, 0)) > null_fitness()


def test_fitness_compare_prioritised_invalid():
    assert SingleObjFitness(None, 10) < SingleObjFitness(1, 20)
    assert SingleObjFitness(1, 10) > SingleObjFitness(None, 20)
    # right-side operand takes precedence in ambiguous case
    assert SingleObjFitness(None, 123) < SingleObjFitness(None)


def test_fitness_compare_prioritised():
    # we minimise fitness, so lesser values are better ones
    assert SingleObjFitness(1, 10) > SingleObjFitness(1, 20)
    assert SingleObjFitness(1, 10, 100) > SingleObjFitness(1, 10, 101.)
    assert SingleObjFitness(0, 20) > SingleObjFitness(1, 10)


def test_fitness_multiobj_dominates():
    # we minimise fitness, so lesser values are better ones
    assert MultiObjFitness([1.]).dominates(MultiObjFitness([2.]))
    assert MultiObjFitness([1., 1., 1.]).dominates(MultiObjFitness([2., 2., 2.]))
    assert MultiObjFitness([1., 1., 3.]).dominates(MultiObjFitness([1., 2., 3.]))

    assert MultiObjFitness([1.]).dominates(MultiObjFitness([1.], weights=[2.]))
    assert MultiObjFitness([1., 1., 1.]).dominates(MultiObjFitness([1., 1., 1.], weights=2.))

    assert not MultiObjFitness([1., 1., 1.]).dominates(MultiObjFitness([1., 1., 1.]))
    assert not MultiObjFitness([1., 1., 2.]).dominates(MultiObjFitness([1., 2., 1.]))


def test_universal_fitness_compare():
    assert to_fitness([1., 1., 3.], multi_objective=False).dominates(to_fitness([1., 2., 3.], multi_objective=False))
    assert to_fitness([1., 1., 3.], multi_objective=True).dominates(to_fitness([1., 2., 3.], multi_objective=True))

    assert to_fitness([1., 1., 3.], multi_objective=False).dominates(to_fitness([1., 2., 1.], multi_objective=False))
    assert not to_fitness([1., 1., 3.], multi_objective=True).dominates(to_fitness([1., 2., 1.], multi_objective=True))


@pytest.mark.parametrize('fitness', get_fitness_objects())
def test_fitness_serialization(fitness):
    dumped = json.dumps(fitness, cls=Serializer)
    reserialized = json.loads(dumped, cls=Serializer)

    assert fitness.__class__ == reserialized.__class__
    assert np.array_equal(fitness.values, reserialized.values)
    assert fitness.valid == reserialized.valid
    if fitness.valid:
        assert fitness == reserialized
