import pytest

from fedot.core.chains.chain import Chain
from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.utilities.synthetic.chain_template_new import ChainTemplate
from fedot.core.composer.composing_history import ComposingHistory


def create_chain():
    first = PrimaryNode(model_type='logit')
    second = PrimaryNode(model_type='lda')
    final = SecondaryNode(model_type='knn', nodes_from=[first, second])

    chain = Chain(final)
    chain.fitness = 1
    return chain


def generate_history(generations_quantity, pop_size):
    history = ComposingHistory()
    for gen in range(generations_quantity):
        new_pop = []
        for idx in range(pop_size):
            chain = create_chain()
            new_pop.append(chain)
        history.add_to_history(new_pop)
    return history


def test_history_adding():
    generations_quantity = 2
    pop_size = 10
    history = generate_history(generations_quantity, pop_size)

    assert len(history.history) == generations_quantity
    for gen in range(generations_quantity):
        assert len(history.history[gen]) == pop_size


def test_convert_chain_to_chain_template():
    generations_quantity = 2
    pop_size = 10
    history = generate_history(generations_quantity, pop_size)
    for gen in range(generations_quantity):
        for chain in range(pop_size):
            assert type(history.history[gen][chain]) == ChainTemplate


def test_prepare_for_visualisation():
    generations_quantity = 2
    pop_size = 10
    history = generate_history(generations_quantity, pop_size)
    assert len(history.historical_chains) == pop_size * generations_quantity
    assert len(history.all_historical_fitness) == pop_size * generations_quantity
