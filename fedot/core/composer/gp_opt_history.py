import csv
import os
import itertools
from typing import (Any, List)

from fedot.utilities.synthetic.chain_template_new import ChainTemplate
from fedot.core.utils import default_fedot_data_dir


class GPOptHistory:
    """
    Contain history, convert Chain to ChainTemplate, save history to csv
    """

    def __init__(self):
        self.history = []

    def convert_chain_to_template(self, chain):
        chain_template = ChainTemplate(chain)
        chain_template.fitness = chain.fitness
        return chain_template

    def add_to_history(self, individuals: List[Any]):
        new_individuals = []
        for chain in individuals:
            new_individuals.append(self.convert_chain_to_template(chain))
        self.history.append(new_individuals)

    def write_composer_history_to_csv(self, file='history.csv'):
        history_dir = os.path.join(default_fedot_data_dir(), 'composing_history')
        file = f'{history_dir}/{file}'
        if not os.path.isdir(history_dir):
            os.mkdir(history_dir)
        self.write_header_to_csv(file)
        i = 0
        for gen_num, gen_chains in enumerate(self.history):
            for chain in gen_chains:
                self.add_history_to_csv(file, chain.fitness, len(chain.model_templates), chain.depth, i, gen_num)
                i += 1

    def write_header_to_csv(self, f):
        with open(f, 'w', newline='') as file:
            writer = csv.writer(file, quoting=csv.QUOTE_ALL)
            writer.writerow(['num', 'generation', 'fitness, num_of_models, depth'])

    def add_history_to_csv(self, f, fitness: float, models_num: int, depth: int, num: int = None,
                           generation: int = None):
        with open(f, 'a', newline='') as file:
            writer = csv.writer(file, quoting=csv.QUOTE_ALL)
            writer.writerow([num, generation, fitness, models_num, depth])

    def prepare_for_visualisation(self):
        historical_fitness = [[chain.fitness for chain in pop] for pop in self.history]
        self.all_historical_fitness = list(itertools.chain(*historical_fitness))
        self.historical_chains = list(itertools.chain(*self.history))
        import pickle
        with open('history_chain.pickle', 'wb') as f:
            pickle.dump(self.history, f)
