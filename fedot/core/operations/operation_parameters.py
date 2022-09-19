from typing import Optional, Iterable

from fedot.core.repository.default_params_repository import DefaultOperationParamsRepository


class OperationParameters:
    def __init__(self, operation_type: Optional[str] = None, parameters: Optional[dict] = None):
        if not parameters and operation_type:
            parameters = get_default_params(operation_type)
        elif not parameters and not operation_type:
            parameters = {}
        self.changed_keys: list = []
        self.parameters = parameters

    def __bool__(self):
        return bool(self.parameters)

    def update(self, key, value):
        if key not in self.changed_keys:
            if self.parameters.get(key) != value:
                self.changed_keys.append(key)
        self.parameters.update({key: value})

    def get(self, key, default_value=None):
        self.parameters.get(key, default_value)

    def get_parameters(self) -> dict:
        return self.parameters

    def keys(self) -> Iterable:
        return self.parameters.keys()

    @property
    def changed_parameters(self) -> dict:
        changed_parameters = {key: self.parameters[key] for key in self.changed_keys}
        return changed_parameters


def get_default_params(operation_type: str) -> dict:
    """Gets default params for chosen model name

    :param operation_type: the operation name to choose default parameters for

    :return: default repository parameters for the model name
    """
    with DefaultOperationParamsRepository() as default_params_repo:
        return default_params_repo.get_default_params_for_operation(operation_type)
