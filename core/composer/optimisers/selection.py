from random import randint
from typing import (
    List,
    Any
)

import numpy as np


def random_selection(pop_size: int, group_size: int):
    return [randint(0, pop_size - 1) for _ in range(group_size)]


def tournament_selection(fitnesses: List[Any], group_size: int = 5, minimization=True):
    group_size = min(group_size, len(fitnesses))
    chosen = []
    for i in range(len(fitnesses)):
        chosen.append([])
        for j in range(2):
            group = random_selection(len(fitnesses), group_size)
            choice_func = np.argmin if minimization else np.argmax
            chosen[i].append(group[choice_func([fitnesses[ind_num] for ind_num in group])])
    return chosen
