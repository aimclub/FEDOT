import numpy as np

from fedot.core.data.data import OutputData
from fedot.core.operations.operation import Operation
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.operation_types_repository import OperationTypesRepository, OperationReposEnum
from fedot.core.repository.tasks import TaskTypesEnum
from fedot.preprocessing.data_types import TYPE_TO_ID


class Model(Operation):
    """Class with ``fit``/``predict`` methods defining the evaluation strategy for the task

    Args:
        operation_type: name of the model
    """

    def __init__(self, operation_type: str):
        super().__init__(operation_type=operation_type)
        self.operations_repo = OperationTypesRepository(OperationReposEnum.MODEL)

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
                col_type_ids = {'features': [TYPE_TO_ID[float]] * predict_shape[0]}
            else:
                col_type_ids = {'features': [TYPE_TO_ID[float]] * predict_shape[1]}
        else:
            if len(predict_shape) < 2:
                output_data.predict = output_data.predict.reshape((-1, 1))
                predict_shape = output_data.predict.shape
            # Classification task or clustering
            target_type = int if output_mode == 'labels' else float
            col_type_ids = {'features': [TYPE_TO_ID[target_type]] * predict_shape[1]}

        # Make feature types static to suit supplementary data contract
        col_type_ids['features'] = np.array(col_type_ids['features'])

        # Add information about target
        target_shape = output_data.target.shape if output_data.target is not None else None
        if target_shape is None:
            # There is no target column in output data
            output_data.supplementary_data.col_type_ids = col_type_ids
            return output_data

        if is_regression_task or is_ts_forecasting_task:
            if len(target_shape) > 1:
                col_type_ids['target'] = [TYPE_TO_ID[float]] * target_shape[1]
            else:
                # Array present "time series"
                col_type_ids['target'] = [TYPE_TO_ID[float]] * len(output_data.target)
        else:
            # Classification task or clustering
            col_type_ids['target'] = [TYPE_TO_ID[int]] * predict_shape[1]

        # Make target types static to suit supplementary data contract
        col_type_ids['target'] = np.array(col_type_ids['target'])

        output_data.supplementary_data.col_type_ids = col_type_ids
        return output_data
