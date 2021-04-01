from fedot.core.composer.optimisers.utils.population_utils import is_equal_archive
from fedot.core.composer.optimisers.utils.multi_objective_fitness import MultiObjFitness
from test.unit.chains.test_node_cache import chain_first, chain_third
from deap import tools
from copy import deepcopy


def test_is_eq_archive():
    archive_chain_first = chain_third()
    archive_chain_second = chain_first()
    archive_chain_trird = chain_third()
    archive_chain_forth = chain_third()
    population_first = [archive_chain_first, archive_chain_second, archive_chain_trird]
    population_second = deepcopy(population_first)
    population_first.append(archive_chain_forth)
    eval_fitness = [(-0.9821, 0.8), (-0.8215, 0.6), (-0.21111, 0.4), (-0.92, 0.9)]
    for population in (population_first, population_second):
        for chain_num, chain in enumerate(population):
            fitness = MultiObjFitness(values=eval_fitness[chain_num],
                                      weights=tuple([-1 for _ in range(len(eval_fitness[chain_num]))]))
            chain.fitness = fitness
    archive_first = tools.ParetoFront()
    archive_first.update(population_first)
    assert len(archive_first.items) == 3
    archive_second = tools.ParetoFront()
    archive_second.update(population_second)
    assert is_equal_archive(archive_first, archive_second)
    new_fitness = (-0.9821, 0.80001)
    population_second[0].fitness.values = new_fitness
    archive_third = tools.ParetoFront()
    archive_third.update(population_second)
    assert not is_equal_archive(archive_first, archive_third)
