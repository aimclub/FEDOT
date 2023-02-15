from importlib import import_module
from typing import Union, TYPE_CHECKING, List

# imports are required for the eval
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TaskTypesEnum

if TYPE_CHECKING:
    from fedot.core.operations.evaluation.evaluation_interfaces import EvaluationStrategy


def read_field(source: dict, field_name: str, default: list):
    """
    Function for reading field in the dictionary

    Args:
        source: dictionary with information
        field_name: name of the looked up field in the ``source``
        default: default list if ``field_name`` is not in the source dict keys

    Returns:
        list with field values
    """
    field_value = source.get(field_name, default)
    if isinstance(field_value, str):
        return import_enums_from_str(field_value)
    return field_value


def import_enums_from_str(field_value: str) -> Union[List[DataTypesEnum],
                                                     List[TaskTypesEnum]]:
    """
    Imports enums by theirs string name representation and returns list of theirs values

    Args:
        field_value: str representing list of
            either class:`DataTypesEnum` or class:`TaskTypesEnum` values

    Returns:
        list of either class:`DataTypesEnum` or class:`TaskTypesEnum` values
    """
    enums = [full_val.split('.') for full_val in field_value.strip('][').split(', ') if full_val != '']
    return [
        getattr(globals()[data_type], value)
        for (data_type, value) in enums]


def import_strategy_from_str(field_value: List[str]) -> 'EvaluationStrategy':
    """
    Imports evaluation strategy module and returns its particular type

    Args:
        field_value: list of [namespace, type_name]

    Returns:
        specific evaluation strategy
    """
    return getattr(import_module(field_value[0]), field_value[1])
