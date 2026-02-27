import dask
import pytest
from fedot.core.data.data import OutputData

from fedot_ind.core.operation.dummy.dummy_operation import init_input_data
from fedot_ind.core.architecture.settings.computational import backend_methods as np
from fedot_ind.core.operation.transformation.basis.fourier import FourierBasisImplementation
from fedot_ind.tools.synthetic.ts_datasets_generator import TimeSeriesDatasetsGenerator


@pytest.fixture
def dataset():
    (X_train, y_train), (X_test, y_test) = TimeSeriesDatasetsGenerator(
        num_samples=20, max_ts_len=50, binary=True, test_size=0.5).generate_data()
    return X_train, y_train, X_test, y_test


# @pytest.fixture
# def input_train(dataset):
#     X_train, y_train, X_test, y_test = dataset
#     input_train_data = init_input_data(X_train, y_train)
#     return input_train_data

@pytest.fixture
def input_train():
    x_train = np.random.rand(100, 1, 100)
    y_train = np.random.rand(100).reshape(-1, 1)
    input_train_data = init_input_data(x_train, y_train)
    return input_train_data


def test_transform(input_train):
    basis = FourierBasisImplementation({})
    train_features = basis.transform(input_data=input_train)
    assert isinstance(train_features, OutputData)
    assert train_features.features.shape[0] == input_train.features.shape[0]


def test_transform_one_sample(input_train):
    basis = FourierBasisImplementation({})
    sample = input_train.features[0]
    transformed_sample = basis._transform_one_sample(sample)
    transformed_sample = dask.compute(transformed_sample)[0]
    assert isinstance(transformed_sample, list)
    assert transformed_sample[0].shape[0] == len(sample)


def test_decompose_signal(input_train):
    basis = FourierBasisImplementation({})
    sample = input_train.features[0].reshape(-1)
    transformed_sample = basis._decompose_signal(sample)
    assert isinstance(transformed_sample, np.ndarray)
    assert transformed_sample.shape[1] == len(sample)
