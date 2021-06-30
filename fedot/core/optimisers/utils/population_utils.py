from typing import Any

import numpy as np

from fedot.core.data.data import InputData


def is_equal_fitness(first_fitness, second_fitness, atol=1e-10, rtol=1e-10):
    return np.isclose(first_fitness, second_fitness, atol=atol, rtol=rtol)


def is_equal_archive(old_archive: Any, new_archive: Any) -> bool:
    fronts_coincidence = True
    if len(old_archive.items) != len(new_archive.items):
        fronts_coincidence = False
    else:
        for new_ind in new_archive.items:
            is_ind_found = False
            for old_ind in old_archive.items:
                if new_ind.fitness == old_ind.fitness:
                    is_ind_found = True
                    break
            if not is_ind_found:
                fronts_coincidence = False
                break
    return fronts_coincidence


def get_metric_position(metrics, metric_type):
    metric_position = None
    for num, metric in enumerate(metrics):
        if isinstance(metric, metric_type):
            metric_position = num
            break
    return metric_position


def nested_list_transform_to_tuple(data_field):
    if isinstance(data_field, (list, np.ndarray)):
        transformed = tuple(map(nested_list_transform_to_tuple, data_field))
    else:
        transformed = data_field
    return transformed


def input_data_characteristics(data: InputData, log):
    data_type = data.data_type
    if data.features is not None:
        features_hash = hash(nested_list_transform_to_tuple(data.features))
    else:
        features_hash = None
        log.info('Input data features is None')
    if data.target is not None:
        target_hash = hash(nested_list_transform_to_tuple(data.target))
    else:
        log.info('Input data target is None')
        target_hash = None
    return data_type, features_hash, target_hash
