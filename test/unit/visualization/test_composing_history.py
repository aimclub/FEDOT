from fedot.core.chains.chain import Chain
from fedot.core.chains.chain_template import ChainTemplate
from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.composer.composing_history import ComposingHistory
from fedot.core.composer.optimisers.gp_comp.individual import Individual
from fedot.core.composer.optimisers.utils.multi_objective_fitness import MultiObjFitness


def create_individual():
    first = PrimaryNode(operation_type='logit')
    second = PrimaryNode(operation_type='lda')
    final = SecondaryNode(operation_type='knn', nodes_from=[first, second])

    indiviual = Individual(chain=Chain(final))
    indiviual.fitness = 1
    return indiviual


def generate_history(generations_quantity, pop_size):
    history = ComposingHistory()
    for _ in range(generations_quantity):
        new_pop = []
        for _ in range(pop_size):
            ind = create_individual()
            new_pop.append(ind)
        history.add_to_history(new_pop)
    return history


def test_history_adding():
    generations_quantity = 2
    pop_size = 10
    history = generate_history(generations_quantity, pop_size)

    assert len(history.individuals) == generations_quantity
    for gen in range(generations_quantity):
        assert len(history.individuals[gen]) == pop_size


def test_convert_chain_to_chain_template():
    generations_quantity = 2
    pop_size = 10
    history = generate_history(generations_quantity, pop_size)
    for gen in range(generations_quantity):
        for ind in range(pop_size):
            assert type(history.individuals[gen][ind].chain) == ChainTemplate


def test_prepare_for_visualisation():
    generations_quantity = 2
    pop_size = 10
    history = generate_history(generations_quantity, pop_size)
    assert len(history.historical_chains) == pop_size * generations_quantity
    assert len(history.all_historical_fitness) == pop_size * generations_quantity


def test_all_historical_quality():
    pop_size = 4
    generations_quantity = 3
    history = generate_history(generations_quantity, pop_size)
    eval_fitness = [[-0.9, 0.8], [-0.8, 0.6], [-0.2, 0.4], [-0.9, 0.9]]
    for pop_num, population in enumerate(history.individuals):
        if pop_num != 0:
            eval_fitness = [[fit[0] - 0.5, fit[1]] for fit in eval_fitness]
        for chain_num, chain in enumerate(population):
            fitness = MultiObjFitness(values=eval_fitness[chain_num],
                                      weights=tuple([-1 for _ in range(len(eval_fitness[chain_num]))]))
            chain.fitness = fitness
    all_quality = history.all_historical_quality
    assert all_quality[0] == -0.9 and all_quality[4] == -1.4 and all_quality[5] == -1.3 and all_quality[10] == -1.2
