from typing import Union

# imports are required for the eval
from fedot.core.repository.dataset_types import *
from fedot.core.repository.tasks import *


def read_field(source, field_name, default):
    # TODO add docstring
    if field_name in source.keys():
        field_value = source[field_name]
        if isinstance(field_value, str):
            return eval_field_str(field_value)
        else:
            return field_value
    else:
        return default


def eval_field_str(field_value) -> Union[List[DataTypesEnum],
                                         List[TaskTypesEnum]]:
    # TODO add docstring
    return eval(field_value)


def eval_strategy_str(field_value):
    # TODO add docstring
    namespace = field_value[0]
    exec(f'from {namespace} import {field_value[1]}')
    return eval(field_value[1])
