import numpy as np
from typing import Any


def is_equal_fitness(first_fitness, second_fitness, atol=1e-10, rtol=1e-10):
    return np.isclose(first_fitness, second_fitness, atol=atol, rtol=rtol)


def is_equal_archive(old_archive: Any, new_archive: Any) -> bool:
    if len(old_archive.items) != len(new_archive.items):
        fronts_coincidence = False
    else:
        are_inds_found = []
        for ind in new_archive:
            eq_inds = list(filter(lambda item: all(
                [is_equal_fitness(obj, ind.fitness.values[obj_num]) for obj_num, obj in
                 enumerate(item.fitness.values)]), old_archive.items))
            are_inds_found.append(len(eq_inds) > 0)
        fronts_coincidence = all(are_inds_found)

    return fronts_coincidence


def get_metric_position(metrics, metric_type):
    metric_position = None
    for num, metric in enumerate(metrics):
        if isinstance(metric, metric_type):
            metric_position = num
            break
    return metric_position
