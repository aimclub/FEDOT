import csv
import os
import itertools
from typing import (Any, List)

from fedot.core.chains.chain_template import ChainTemplate
from fedot.core.utils import default_fedot_data_dir
from fedot.core.composer.optimisers.multi_objective_fitness import MultiObjFitness


class ComposingHistory:
    """
    Contain history, convert Chain to ChainTemplate, save history to csv
    """

    def __init__(self):
        self.chains = []
        self.archive_history = []

    def _convert_chain_to_template(self, chain):
        chain_template = ChainTemplate(chain)
        chain_template.fitness = chain.fitness
        return chain_template

    def add_to_history(self, individuals: List[Any]):
        new_individuals = []
        for chain in individuals:
            new_individuals.append(self._convert_chain_to_template(chain))
        self.chains.append(new_individuals)

    def add_to_archive_history(self, individuals: List[Any]):
        new_individuals = []
        for chain in individuals:
            new_individuals.append(self._convert_chain_to_template(chain))
        self.archive_history.append(new_individuals)

    def write_composer_history_to_csv(self, file='history.csv'):
        history_dir = os.path.join(default_fedot_data_dir(), 'composing_history')
        file = os.path.join(history_dir, file)
        if not os.path.isdir(history_dir):
            os.mkdir(history_dir)
        self._write_header_to_csv(file)
        idx = 0
        for gen_num, gen_chains in enumerate(self.chains):
            for chain in gen_chains:
                self._add_history_to_csv(file, chain.fitness, len(chain.model_templates), chain.depth, idx, gen_num,
                                         chain.computation_time)
                idx += 1

    def _write_header_to_csv(self, f):
        with open(f, 'w', newline='') as file:
            writer = csv.writer(file, quoting=csv.QUOTE_ALL)
            row = ['index', 'generation', 'quality_metric', 'quantity_of_models', 'depth', 'computation_time']
            if self.is_multi_objective:
                row.append('complexity_metric')
            writer.writerow(row)

    def _add_history_to_csv(self, f, fitness: float, models_quantity: int, depth: int, idx: int = None,
                            generation: int = None, comp_time: float = None, compl_metric: float = None):
        with open(f, 'a', newline='') as file:
            writer = csv.writer(file, quoting=csv.QUOTE_ALL)
            row = [idx, generation, fitness, models_quantity, depth, comp_time]
            if compl_metric is not None:
                row.append(compl_metric)
            writer.writerow(row)

    @property
    def all_historical_fitness(self):
        if self.is_multi_objective:
            historical_fitness = [[chain.fitness.values[0] for chain in pop] for pop in self.chains]
        else:
            historical_fitness = [[chain.fitness for chain in pop] for pop in self.chains]
        return list(itertools.chain(*historical_fitness))

    @property
    def historical_chains(self):
        return list(itertools.chain(*self.chains))

    @property
    def is_multi_objective(self):
        return type(self.chains[0][0].fitness) is MultiObjFitness
