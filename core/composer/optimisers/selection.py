from random import randint
from typing import (
    List,
    Any,
    Tuple
)

import numpy as np


def random_selection(pop_size: int, group_size: int) -> List[int]:
    return [randint(0, pop_size - 1) for _ in range(group_size)]


def tournament_selection(fitnesses: List[float], population: List[Any], group_size: int = 5, num_of_parents=2) -> List[
    Tuple[int, int]]:
    group_size = min(group_size, len(fitnesses))
    chosen = []

    for i in range(len(fitnesses)):
        choice_item = []
        for _ in range(num_of_parents):
            group = random_selection(len(fitnesses), group_size)
            num_of_chosen_ind = group[np.argmin([fitnesses[ind_num] for ind_num in group])]
            choice_item.append(population[num_of_chosen_ind])
        chosen.append(choice_item)
    return chosen
