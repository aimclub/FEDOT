import csv
import os
from typing import (Any, List)

from fedot.core.utils import default_fedot_data_dir


def write_composer_history_to_csv(historical_chains: List[Any], file='history.csv'):
    history_dir = os.path.join(default_fedot_data_dir(), 'composing_history')
    file = f'{history_dir}/{file}'
    if not os.path.isdir(history_dir):
        os.mkdir(history_dir)
    write_header_to_csv(file)
    i = 0
    for gen_num, gen_chains in enumerate(historical_chains):
        for chain in gen_chains:
            add_history_to_csv(file, chain.fitness, len(chain.nodes), chain.depth, i, gen_num)
            i += 1


def write_header_to_csv(f):
    with open(f, 'w', newline='') as file:
        writer = csv.writer(file, quoting=csv.QUOTE_ALL)
        writer.writerow(['num', 'generation', 'fitness, num_of_models, depth'])


def add_history_to_csv(f, fitness: float, models_num: int, depth: int, num: int = None, generation: int = None):
    with open(f, 'a', newline='') as file:
        writer = csv.writer(file, quoting=csv.QUOTE_ALL)
        writer.writerow([num, generation, fitness, models_num, depth])
