from typing import Any, Union

# imports are required for the eval
from core.repository.dataset_types import *
from core.repository.tasks import *


def read_field(source, field_name, default):
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
    return eval(field_value)


def eval_strategy_str(field_value):
    exec(f'from core.models.evaluation.evaluation import {field_value}')
    return eval(field_value)
