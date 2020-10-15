import csv
import math
import os
from pathlib import Path
from typing import (Any, List)


def write_composer_history_to_csv(historical_fitness: List[int], historical_chains: List[Any], pop_size: int,
                                  f='history.csv'):
    # f = f'../../tmp/{f}'
    history_dir = f'{str(Path.home())}/composing_history'
    f = f'{history_dir}/{f}'
    if not os.path.isdir(history_dir):
        os.mkdir(history_dir)
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
