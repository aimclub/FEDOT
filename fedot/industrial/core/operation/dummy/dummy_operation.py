from typing import Optional

import pandas as pd
import torch
from dask_ml.preprocessing import LabelEncoder
from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import \
    DataOperationImplementation
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TaskTypesEnum, Task

from fedot.industrial.core.architecture.settings.computational import backend_methods as np


def check_multivariate_data(data: pd.DataFrame) -> tuple:
    """
    Checks if the provided pandas DataFrame contains multivariate data.

    Args:
        data (pd.DataFrame): The DataFrame to be analyzed.

    Returns:
        bool: True if the DataFrame contains multivariate data (nested columns), False otherwise.
    """
    if not isinstance(data, pd.DataFrame):
        return len(data.shape) > 2, data
    else:
        return isinstance(data.iloc[0, 0], pd.Series), data.values


def init_input_data(X: pd.DataFrame, y: Optional[np.ndarray], task: str = 'classification') -> InputData:
    """
    Initializes a Fedot InputData object from input features and target.

    Args:
        X: The DataFrame containing features.
        y: The NumPy array containing target values.
        task: The machine learning task type ("classification" or "regression"). Defaults to "classification".

    Returns:
        InputData: The initialized Fedot InputData object.

    """

    is_multivariate_data, features = check_multivariate_data(X)
    task_dict = {'classification': Task(TaskTypesEnum.classification),
                 'regression': Task(TaskTypesEnum.regression)}

    if y is not None and isinstance(
            y[0], np.str_) and task == 'classification':
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
    elif y is not None and isinstance(y[0], np.str_) and task == 'regression':
        y = y.astype(float)

    data_type = DataTypesEnum.image if is_multivariate_data else DataTypesEnum.table
    input_data = InputData(idx=np.arange(len(X)),
                           features=np.array(features.tolist()).astype(float),
                           target=y.reshape(-1, 1) if y is not None else y,
                           task=task_dict[task],
                           data_type=data_type)

    if input_data.target is not None:
        if task == 'regression':
            input_data.target = input_data.target.squeeze()
        elif task == 'classification':
            input_data.target[input_data.target == -1] = 0

    # Replace NaN and infinite values with 0 in features
    input_data.features = np.where(
        np.isnan(input_data.features), 0, input_data.features)
    input_data.features = np.where(
        np.isinf(input_data.features), 0, input_data.features)

    return input_data


def init_input_data_tensor(X: pd.DataFrame, y: Optional[np.ndarray], task: str = 'classification') -> InputData:
    """
    Initializes a Fedot InputData object from input features and target.

    Args:
        X: The DataFrame containing features.
        y: The NumPy array containing target values.
        task: The machine learning task type ("classification" or "regression"). Defaults to "classification".

    Returns:
        InputData: The initialized Fedot InputData object.

    """

    is_multivariate_data, features = check_multivariate_data(X)
    task_dict = {'classification': Task(TaskTypesEnum.classification),
                 'regression': Task(TaskTypesEnum.regression)}

    if y is not None and isinstance(
            y[0], np.str_) and task == 'classification':
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
    elif y is not None and isinstance(y[0], np.str_) and task == 'regression':
        y = y.astype(float)

    data_type = DataTypesEnum.image if is_multivariate_data else DataTypesEnum.table
    input_data = InputData(idx=np.arange(len(X)),
                           features=torch.Tensor(features.tolist()).to(X.device),
                           target=y.reshape(-1, 1) if y is not None else y,
                           task=task_dict[task],
                           data_type=data_type)

    if input_data.target is not None:
        if task == 'regression':
            input_data.target = input_data.target.squeeze()
        elif task == 'classification':
            input_data.target[input_data.target == -1] = 0

    # Replace NaN and infinite values with 0 in features
    input_data.features = torch.where(
        torch.isnan(input_data.features),
        torch.tensor(0.0, device=input_data.features.device, dtype=input_data.features.dtype),
        input_data.features
    )
    input_data.features = torch.where(
        torch.isinf(input_data.features),
        torch.tensor(0.0, device=input_data.features.device, dtype=input_data.features.dtype),
        input_data.features
    )
    return input_data


class DummyOperation(DataOperationImplementation):
    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.average = params.get('average_type', None)
        self.prediction_length = params.get('prediction_length', None)

    def fit(self, input_data: InputData):
        pass

    def transform(self, input_data: InputData) -> OutputData:
        if self.average is not None:
            transformed_features = np.average(
                input_data.features.reshape(-1, self.prediction_length), axis=0)
        else:
            transformed_features = input_data.features
        predict = self._convert_to_output(
            input_data, transformed_features, data_type=DataTypesEnum.table)
        return predict
