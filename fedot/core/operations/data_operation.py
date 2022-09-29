from typing import Optional

from fedot.core.data.data import OutputData
from fedot.core.operations.operation import Operation
from fedot.core.repository.operation_types_repository import OperationMetaInfo, OperationTypesRepository


class DataOperation(Operation):
    """Class with ``fit``/``predict`` methods defining the evaluation strategy for the task

    Args:
        operation_type: name of the data operation
    """
    def __init__(self, operation_type: str):
        super().__init__(operation_type)
        self.operations_repo = OperationTypesRepository('data_operation')

    @property
    def metadata(self) -> OperationMetaInfo:
        operation_info = self.operations_repo.operation_info_by_id(self.operation_type)
        if not operation_info:
            raise ValueError(f'Data operation {self.operation_type} not found')
        return operation_info

    @staticmethod
    def assign_tabular_column_types(output_data: OutputData, output_mode: str) -> OutputData:
        """Assign new column types if it necessary.
        By default, all data operations must define column types at lower levels (:obj:`EvalStrategies` and :obj:`Implementations`).
        In some cases the previously defined data types are passed.
        """
        return output_data
