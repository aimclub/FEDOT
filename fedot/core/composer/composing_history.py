import csv
import os
import itertools
from typing import (Any, List)

from FEDOT.fedot.core.chains.chain_template import ChainTemplate
from FEDOT.fedot.core.composer.optimisers.utils.multi_objective_fitness import MultiObjFitness
from FEDOT.fedot.core.composer.optimisers.utils.population_utils import get_metric_position
from FEDOT.fedot.core.repository.quality_metrics_repository import QualityMetricsEnum
from FEDOT.fedot.core.utils import default_fedot_data_dir


class ComposingHistory:
    """
    Contain history, convert Chain to ChainTemplate, save history to csv
    """

    def __init__(self, metrics=None):
        self.metrics = metrics
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
                if self.is_multi_objective:
                    fitness = chain.fitness.values
                else:
                    fitness = chain.fitness
                row = [idx, gen_num, fitness, len(chain.model_templates), chain.depth, chain.computation_time]
                self._add_history_to_csv(file, row)
                idx += 1

    def _write_header_to_csv(self, f):
        with open(f, 'w', newline='') as file:
            writer = csv.writer(file, quoting=csv.QUOTE_ALL)
            metric_str = 'metric'
            if self.is_multi_objective:
                metric_str += 's'
            row = ['index', 'generation', metric_str, 'quantity_of_models', 'depth', 'computation_time']
            writer.writerow(row)

    def _add_history_to_csv(self, f: str, row: List[Any]):
        with open(f, 'a', newline='') as file:
            writer = csv.writer(file, quoting=csv.QUOTE_ALL)
            writer.writerow(row)

    @property
    def historical_fitness(self):
        if self.is_multi_objective:
            historical_fitness = []
            for objective_num in range(len(self.chains[0][0].fitness.values)):
                objective_history = [[chain.fitness.values[objective_num] for chain in pop] for pop in self.chains]
                historical_fitness.append(objective_history)
        else:
            historical_fitness = [[chain.fitness for chain in pop] for pop in self.chains]
        return historical_fitness

    @property
    def all_historical_fitness(self):
        historical_fitness = self.historical_fitness
        if self.is_multi_objective:
            all_historical_fitness = []
            for obj_num in range(len(historical_fitness)):
                all_historical_fitness.append(list(itertools.chain(*historical_fitness[obj_num])))
        else:
            all_historical_fitness = list(itertools.chain(*historical_fitness))
        return all_historical_fitness

    @property
    def all_historical_quality(self):
        if self.is_multi_objective:
            if self.metrics:
                metric_position = get_metric_position(self.metrics, QualityMetricsEnum)
            else:
                metric_position = 0
            all_historical_quality = self.all_historical_fitness[metric_position]
        else:
            all_historical_quality = self.all_historical_fitness
        return all_historical_quality

    @property
    def historical_chains(self):
        return list(itertools.chain(*self.chains))

    @property
    def is_multi_objective(self):
        return type(self.chains[0][0].fitness) is MultiObjFitness
