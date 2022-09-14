import numpy as np

from fedot.core.data.data import OutputData
from fedot.core.operations.operation import Operation
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.operation_types_repository import OperationTypesRepository
from fedot.core.repository.tasks import TaskTypesEnum


class Model(Operation):
    """Class with ``fit``/``predict`` methods defining the evaluation strategy for the task

    Args:
        operation_type: name of the model
    """

    def __init__(self, operation_type: str):
        super().__init__(operation_type=operation_type)
        self.operations_repo = OperationTypesRepository('model')

    @staticmethod
    def assign_tabular_column_types(output_data: OutputData, output_mode: str) -> OutputData:
        """Assign types for tabular data obtained from model predictions.\n
        By default, all types of model predictions for tabular data can be clearly defined
        """
        if output_data.data_type is not DataTypesEnum.table:
            # No column data types info for non-tabular data
            return output_data

        is_regression_task = output_data.task.task_type is TaskTypesEnum.regression
        is_ts_forecasting_task = output_data.task.task_type is TaskTypesEnum.ts_forecasting

        predict_shape = np.array(output_data.predict).shape
        # Add information about features
        if is_regression_task or is_ts_forecasting_task:
            if len(predict_shape) < 2:
                column_info = {'features': [str(float)] * predict_shape[0]}
            else:
                column_info = {'features': [str(float)] * predict_shape[1]}
        else:
            if len(predict_shape) < 2:
                output_data.predict = output_data.predict.reshape((-1, 1))
                predict_shape = output_data.predict.shape
            # Classification task or clustering
            if output_mode == 'labels':
                column_info = {'features': [str(int)] * predict_shape[1]}
            else:
                column_info = {'features': [str(float)] * predict_shape[1]}

        # Add information about target
        target_shape = output_data.target.shape if output_data.target is not None else None
        if target_shape is None:
            # There is no target column in output data
            output_data.supplementary_data.column_types = column_info
            return output_data

        if is_regression_task or is_ts_forecasting_task:
            if len(target_shape) > 1:
                column_info.update({'target': [str(float)] * target_shape[1]})
            else:
                # Array present "time series"
                column_info.update({'target': [str(float)] * len(output_data.target)})
        else:
            # Classification task or clustering
            column_info.update({'target': [str(int)] * predict_shape[1]})

        output_data.supplementary_data.column_types = column_info
        return output_data
