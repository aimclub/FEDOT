import csv
import math
import os
from dataclasses import dataclass
from functools import partial
from typing import (
    Callable,
    Optional,
    List
)

from core.chain_validation import validate
from core.composer.chain import Chain, SharedChain
from core.composer.composer import Composer, ComposerRequirements
from core.composer.node import NodeGenerator
from core.composer.optimisers.gp_optimiser import GPChainOptimiser, GPChainOptimiserParameters
from core.composer.visualisation import ComposerVisualiser
from core.models.data import InputData
from core.models.data import train_test_data_setup


@dataclass
class GPComposerRequirements(ComposerRequirements):
    pop_size: Optional[int] = 50
    num_of_generations: Optional[int] = 50
    crossover_prob: Optional[float] = None
    mutation_prob: Optional[float] = None


class GPComposer(Composer):
    def __init__(self):
        super(Composer, self).__init__()
        self.shared_cache = {}

    def compose_chain(self, data: InputData, initial_chain: Optional[Chain],
                      composer_requirements: Optional[GPComposerRequirements],
                      metrics: Optional[Callable], optimiser_parameters: GPChainOptimiserParameters = None,
                      is_visualise: bool = False) -> Chain:

        train_data, test_data = train_test_data_setup(data, 0.8)
        self.shared_cache.clear()

        metric_function_for_nodes = partial(self.metric_for_nodes,
                                            metrics, train_data, test_data, True)

        optimiser = GPChainOptimiser(initial_chain=initial_chain,
                                     requirements=composer_requirements,
                                     primary_node_func=NodeGenerator.primary_node,
                                     secondary_node_func=NodeGenerator.secondary_node, chain_class=Chain,
                                     parameters=optimiser_parameters)

        best_chain, self.history = optimiser.optimise(metric_function_for_nodes)

        historical_chains, historical_fitness = [list(hist_tuple) for hist_tuple in list(zip(*self.history))]

        if is_visualise:
            ComposerVisualiser.visualise_history(historical_chains, historical_fitness)

        write_composer_history_to_csv(historical_fitness=historical_fitness, historical_chains=historical_chains,
                                      pop_size=composer_requirements.pop_size)

        print("GP composition finished")
        return best_chain

    def metric_for_nodes(self, metric_function, train_data: InputData,
                         test_data: InputData, is_chain_shared: bool,
                         chain: Chain) -> float:

        validate(chain)
        if is_chain_shared:
            chain = SharedChain(base_chain=chain, shared_cache=self.shared_cache)
        chain.fit(input_data=train_data)
        return metric_function(chain, test_data)


def write_composer_history_to_csv(historical_fitness: List[int], historical_chains: List[Chain], pop_size: int,
                                  f='history.csv'):
    f = f'../../tmp/{f}'
    if not os.path.isdir('../../tmp'):
        os.mkdir('../../tmp')
    write_header_to_csv(f)
    for i, fitness in enumerate(historical_fitness):
        gen_num = math.ceil(i / pop_size)
        historical_chain = historical_chains[i]
        add_history_to_csv(f, fitness, len(historical_chain.nodes), historical_chain.depth, i, gen_num)


def write_header_to_csv(f):
    with open(f, 'w', newline='') as file:
        writer = csv.writer(file, quoting=csv.QUOTE_ALL)
        writer.writerow(['num', 'generation', 'fitness, num_of_models, depth'])


def add_history_to_csv(f, fitness: float, models_num: int, depth: int, num: int = None, generation: int = None):
    with open(f, 'a', newline='') as file:
        writer = csv.writer(file, quoting=csv.QUOTE_ALL)
        writer.writerow([num, generation, fitness, models_num, depth])
