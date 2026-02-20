import numpy as np
import torch
import pytest

from fedot.core.data.tensordata import TensorData
from fedot.core.utils import fedot_project_root


def test_create_from_numpy():
    """Test TensorData creation from numpy array."""

    features = np.random.rand(100, 10)

    td = TensorData.create(
        features
    )
    assert isinstance(td, TensorData)
    assert isinstance(td.features, torch.Tensor)
    assert isinstance(td.target, torch.Tensor)


def test_create_from_csv():
    """Test TensorData creation from CSV file."""

    csv_path = f'{fedot_project_root()}/examples/real_cases/data/scoring/scoring_train.csv'

    td = TensorData.create(
        csv_path
    )

    assert isinstance(td, TensorData)
    assert isinstance(td.features, torch.Tensor)
    assert isinstance(td.target, torch.Tensor)
