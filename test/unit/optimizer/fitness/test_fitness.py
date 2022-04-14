from copy import deepcopy
from itertools import product

import pytest

from fedot.core.optimisers.fitness.fitness import *
from fedot.core.optimisers.fitness.multi_objective_fitness import MultiObjFitness


def get_fitness_objects():
    return [
        none_fitness(),
        single_value_fitness(42.),

        PrioritisedFitness(None),
        PrioritisedFitness(None, 12.34),
        PrioritisedFitness(11),
        PrioritisedFitness(1, 2, 3, 4),

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
    assert fitness.values == fitness_values


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
    assert PrioritisedFitness(1., 2., 3. + 1e-12) == PrioritisedFitness(1., 2., 3.)
    assert PrioritisedFitness(1) != MultiObjFitness([1])
    assert PrioritisedFitness(1., 2., 3.) != MultiObjFitness([1., 2., 3.])


def test_fitness_compare_prioritised_invalid():
    assert PrioritisedFitness(None, 10) < PrioritisedFitness(1, 20)
    assert PrioritisedFitness(1, 10) > PrioritisedFitness(None, 20)
    # right-side operand takes precedence in ambigious case
    assert PrioritisedFitness(None, 123) < PrioritisedFitness(None)


def test_fitness_compare_prioritised():
    assert PrioritisedFitness(1, 10) < PrioritisedFitness(1, 20)
    assert PrioritisedFitness(1, 10, 100) < PrioritisedFitness(1, 10, 101.)
    assert PrioritisedFitness(0, 20) < PrioritisedFitness(1, 10)


def test_fitness_multiobj_dominates():
    assert MultiObjFitness([2.]).dominates(MultiObjFitness([1.]))
    assert MultiObjFitness([2., 2., 2.]).dominates(MultiObjFitness([1., 1., 1.]))
    assert MultiObjFitness([1., 2., 3.]).dominates(MultiObjFitness([1., 1., 3.]))

    assert MultiObjFitness([1.], weights=[2.]).dominates(MultiObjFitness([1.]))
    assert MultiObjFitness([1., 1., 1.], weights=2.).dominates(MultiObjFitness([1., 1., 1.]))

    assert not MultiObjFitness([1., 1., 1.]).dominates(MultiObjFitness([1., 1., 1.]))
    assert not MultiObjFitness([1., 2., 1.]).dominates(MultiObjFitness([1., 1., 2.]))
