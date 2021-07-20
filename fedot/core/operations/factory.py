import os

from fedot.core.operations.data_operation import DataOperation
from fedot.core.operations.model import Model
from fedot.core.operations.operation import Operation
from fedot.core.repository.operation_types_repository import OperationTypesRepository
from fedot.core.utils import fedot_project_root


class OperationFactory:
    """
    Base class for determining what type of operations should be defined
    in the node. Possible operations are models (ML models with fit and predict
    methods) and data operations (e.g. scaling) with fit and transform methods

    """

    def __init__(self, operation_name):
        self.operation_name = operation_name
        self.operation_type = self._define_operation_type()

    def get_operation(self) -> Operation:
        """
        Factory method returns the desired object of the 'Data_operation' or
        'Model' class which depends on model_type variable

        """

        if self.operation_type == 'model':
            operation = Model(operation_type=self.operation_name)
        elif self.operation_type == 'data_operation':
            operation = DataOperation(operation_type=self.operation_name)
        else:
            raise ValueError(f'Operation type {self.operation_type} is not supported')

        return operation

    @property
    def operation_type_name(self):
        return self.operation_type

    def _define_operation_type(self) -> str:
        """
        The method determines what type of operations is set for this node

        :return : operations type 'model' or 'data_operation'
        """

        # Get available models from model_repository.json file
        operations_repo = OperationTypesRepository('data_operation_repository.json')
        models = operations_repo.operations

        # If there is a such model in the list
        if any(self.operation_name == model.id for model in models):
            operation_type = 'data_operation'
        # Otherwise - it is model
        else:
            operation_type = 'model'
        return operation_type

    @staticmethod
    def get_repository(repo_tag):
        """
        Uses the fedot/core/repository/data directory
        to extract the correct repository by uniq substring - repo_tag

        :param repo_tag: uniq substring in repository filename
        :return: filename of repository
        """
        repositories_path = os.path.abspath(os.path.join(fedot_project_root(), 'fedot', 'core', 'repository', 'data'))
        repositories = [file_name for file_name in os.listdir(repositories_path)
                        if os.path.splitext(file_name)[1] == '.json' and 'repository' in file_name]
        correct_repo_by_tag = [file_name for file_name in repositories if repo_tag in file_name][0]

        return correct_repo_by_tag

    @classmethod
    def define_repository_by_tag(cls, operation: Model, tag: str = None):
        if isinstance(operation, Model):
            repository_name = OperationFactory.get_repository(repo_tag=tag)
            operation.operations_repo = OperationTypesRepository(repository_name=repository_name)
