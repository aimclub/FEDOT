import pytest

from fedot_ind.core.architecture.abstraction.сheckers import parameter_value_check


@pytest.fixture
def valid_data():
    parameter = 'custom'
    value = 1.35
    valid_values = {1.35, 4, 1}
    return parameter, value, valid_values


@pytest.fixture
def invalid_data():
    parameter = 'custom'
    value = 1.35
    valid_values = {1.354, 4, 1}
    return parameter, value, valid_values


def test_valid_parameter_value_check(valid_data):
    parameter, value, valid_values = valid_data
    assert parameter_value_check(parameter, value, valid_values) is None


def test_invalid_parameter_value_check(invalid_data):
    parameter, value, valid_values = invalid_data
    with pytest.raises(ValueError) as execution_info:
        parameter_value_check(parameter, value, valid_values)
    assert str(
        execution_info.value) == f"{parameter} must be one of {valid_values}, but got {parameter}='{value}'"
