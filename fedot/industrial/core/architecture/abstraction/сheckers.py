"""This module contains value checking functions common to the whole framework."""

from typing import Set, Any


def parameter_value_check(
        parameter: str,
        value: Any,
        valid_values: Set
) -> None:
    """Checks if the parameter value is in the set of valid values.

    Args:
        parameter: Name of the checked parameter.
        value: Value of the checked parameter.
        valid_values: Set of the valid parameter values.

    Rises:
        ValueError: If ``value`` is not in ``valid_values``.


    """
    if value not in valid_values:
        raise ValueError(
            f"{parameter} must be one of {valid_values}, but got {parameter}='{value}'"
        )
