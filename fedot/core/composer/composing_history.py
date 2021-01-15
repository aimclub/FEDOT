import csv
import os
import itertools
from typing import (Any, List)

from fedot.utilities.synthetic.chain_template_new import ChainTemplate
from fedot.core.utils import default_fedot_data_dir


class ComposingHistory:
    """
    Contain history, convert Chain to ChainTemplate, save history to csv
    """

    def __init__(self):
        self.chains = []

    def _convert_chain_to_template(self, chain):
        chain_template = ChainTemplate(chain)
        chain_template.fitness = chain.fitness
        return chain_template

    def add_to_history(self, individuals: List[Any]):
        new_individuals = []
        for chain in individuals:
            new_individuals.append(self._convert_chain_to_template(chain))
        self.chains.append(new_individuals)

    def write_composer_history_to_csv(self, file='history.csv'):
        history_dir = os.path.join(default_fedot_data_dir(), 'composing_history')
        file = os.path.join(history_dir, file)
        if not os.path.isdir(history_dir):
            os.mkdir(history_dir)
        self._write_header_to_csv(file)
        idx = 0
        for gen_num, gen_chains in enumerate(self.chains):
            for chain in gen_chains:
                self._add_history_to_csv(file, chain.fitness, len(chain.model_templates), chain.depth, idx, gen_num)
                idx += 1

    def _write_header_to_csv(self, f):
        with open(f, 'w', newline='') as file:
            writer = csv.writer(file, quoting=csv.QUOTE_ALL)
            writer.writerow(['index', 'generation', 'fitness', 'quantity_of_models', 'depth'])

    def _add_history_to_csv(self, f, fitness: float, models_quantity: int, depth: int, idx: int = None,
                           generation: int = None):
        with open(f, 'a', newline='') as file:
            writer = csv.writer(file, quoting=csv.QUOTE_ALL)
            writer.writerow([idx, generation, fitness, models_quantity, depth])

    @property
    def all_historical_fitness(self):
        historical_fitness = [[chain.fitness for chain in pop] for pop in self.chains]
        return list(itertools.chain(*historical_fitness))

    @property
    def historical_chains(self):
        return list(itertools.chain(*self.chains))
