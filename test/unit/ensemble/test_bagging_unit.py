import pytest
import numpy as np

from sklearn.datasets import make_classification, make_regression

from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum


# ================== Test Data Preparation ========================

def get_binclass_data():
    """Generate binary classification data"""
    X, y = make_classification(n_samples=100, n_features=5,
                               n_classes=2, random_state=42)
    return InputData(idx=np.arange(0, len(X)), features=X, target=y,
                     data_type=DataTypesEnum.table,
                     task=Task(TaskTypesEnum.classification))


def get_multiclass_data():
    """Generate multiclass classification data"""
    X, y = make_classification(n_samples=100, n_features=5,
                               n_classes=3, n_informative=3, random_state=42)
    return InputData(idx=np.arange(0, len(X)), features=X, target=y,
                     data_type=DataTypesEnum.table,
                     task=Task(TaskTypesEnum.classification))


def get_regression_data():
    """Generate regression data"""
    X, y = make_regression(n_samples=100, n_features=5,
                           n_informative=3, random_state=42)
    return InputData(idx=np.arange(0, len(X)), features=X, target=y,
                     data_type=DataTypesEnum.table,
                     task=Task(TaskTypesEnum.regression))

# ================== Tests ========================
