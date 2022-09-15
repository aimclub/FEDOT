from typing import Optional

from fedot.core.repository.default_params_repository import DefaultOperationParamsRepository


class OperationParameters(dict):
    def __init__(self, operation_type: str, parameters: Optional[dict] = None):
        if not parameters:
            parameters = get_default_params(operation_type)
        super(OperationParameters, self).__init__(**parameters)
        self.changed_parameters = []

    def __setitem__(self, key, value):
        super(OperationParameters, self).__setitem__(key, value)
        if key not in self.changed_parameters:
            self.changed_parameters.append(key)


def get_default_params(operation_type: str) -> dict:
    """Gets default params for chosen model name

    :param operation_type: the operation name to choose default parameters for

    :return: default repository parameters for the model name
    """
    with DefaultOperationParamsRepository() as default_params_repo:
        return default_params_repo.get_default_params_for_operation(operation_type)
