import numpy as np

from fedot.core.data.data import InputData


def get_metric_position(metrics, metric_type):
    for num, metric in enumerate(metrics):
        if isinstance(metric, metric_type):
            return num


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
