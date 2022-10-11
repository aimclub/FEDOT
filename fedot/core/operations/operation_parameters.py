from copy import deepcopy
from typing import Optional, Iterable

from fedot.core.repository.default_params_repository import DefaultOperationParamsRepository


class OperationParameters:
    """Stores parameters for models and data operations implementations and records what parameters were changed.
    Uses operation_type to set default parameters from default_parameters_repository if a parameter was not passed
    with parameters.
    If neither parameters nor operation_type were passed, parameters are equal to empty dict.

    Args:
        operation_type: type of the operation defined in operation repository
            the custom prefix can be added after ``/`` (to highlight the specific node)\n
            **The prefix will be ignored at Implementation stage**
        parameters: dict with parameters

    """
    def __init__(self, **parameters):
        self._parameters = parameters
        self._changed_keys: list = []

    def __bool__(self):
        return bool(self._parameters)

    @staticmethod
    def from_operation_type(operation_type: str, **parameters):
        default_parameters = get_default_params(operation_type)
        parameters = {**default_parameters, **parameters}
        return OperationParameters(**parameters)

    def update(self, **params):
        for key, value in params.items():
            if key not in self._changed_keys:
                if self._parameters.get(key) != value:
                    self._changed_keys.append(key)
            self._parameters.update({key: value})

    def get(self, key, default_value=None):
        return self._parameters.get(key, default_value)

    def setdefault(self, key, value):
        if key not in self._parameters.keys():
            self.update(**{key: value})
        return self.get(key)

    def to_dict(self) -> dict:
        return deepcopy(self._parameters)

    def keys(self) -> Iterable:
        return self._parameters.keys()

    @property
    def changed_parameters(self) -> dict:
        changed_parameters = {key: self._parameters[key] for key in self._changed_keys}
        return changed_parameters


def get_default_params(operation_type: str) -> dict:
    """Gets default params for chosen model name

    :param operation_type: the operation name to choose default parameters for

    :return: default repository parameters for the model name
    """
    with DefaultOperationParamsRepository() as default_params_repo:
        return default_params_repo.get_default_params_for_operation(operation_type)
