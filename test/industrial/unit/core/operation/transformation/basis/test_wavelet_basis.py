import pytest
import dask
import pywt

from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.data.data import OutputData, InputData

from fedot_ind.core.architecture.settings.computational import backend_methods as np
from fedot_ind.core.operation.transformation.basis.wavelet import WaveletBasisImplementation


def wavelet_components_combination():
    return [(w, c) for w in ['mexh', 'shan', 'morl', 'cmor', 'fbsp', 'db5', 'sym5'] for c in list(range(2, 12, 2))]


@pytest.fixture
def input_train():
    x_train = np.random.rand(100, 1, 100)
    y_train = np.random.randn(100).reshape(-1, 1)
    input_train_data = InputData(idx=np.arange(len(x_train)),
                                 features=x_train,
                                 target=y_train,
                                 task=Task(TaskTypesEnum.classification),
                                 data_type=DataTypesEnum.image)
    return input_train_data


@pytest.mark.parametrize('wavelet, n_components', wavelet_components_combination())
def test_transform(input_train, wavelet, n_components):
    basis = WaveletBasisImplementation(OperationParameters(wavelet=wavelet, n_components=n_components))
    train_features = basis.transform(input_data=input_train)
    assert isinstance(train_features, OutputData)
    assert train_features.features.shape[0] == input_train.features.shape[0]


@pytest.mark.parametrize('wavelet, n_components', wavelet_components_combination())
def test_decompose_signal(input_train, wavelet, n_components):
    basis = WaveletBasisImplementation(OperationParameters(wavelet=wavelet, n_components=n_components))
    sample = input_train.features[0]
    transformed_sample = basis._decompose_signal(sample)
    assert isinstance(transformed_sample, tuple)
    assert len(transformed_sample) == 2


@pytest.mark.parametrize('wavelet, n_components', wavelet_components_combination())
def test_decomposing_level(input_train, wavelet, n_components):
    basis = WaveletBasisImplementation(OperationParameters(wavelet=wavelet, n_components=n_components))
    sample = input_train.features[0]
    discrete_wavelets = pywt.wavelist(kind='discrete')
    basis.time_series = sample
    basis.wavelet = np.random.choice(discrete_wavelets)
    decomposing_level = basis._decomposing_level()
    assert isinstance(decomposing_level, int)


@pytest.mark.parametrize('wavelet, n_components', wavelet_components_combination())
def test_transform_one_sample(input_train, wavelet, n_components):
    basis = WaveletBasisImplementation(OperationParameters(wavelet=wavelet, n_components=n_components))
    sample = input_train.features[0]
    transformed_sample = basis._transform_one_sample(sample)
    transformed_sample = dask.compute(transformed_sample)[0]
    assert isinstance(transformed_sample, np.ndarray)


@pytest.mark.parametrize('wavelet, n_components', wavelet_components_combination())
def test_get_1d_bassis(input_train, wavelet, n_components):
    basis = WaveletBasisImplementation(OperationParameters(wavelet=wavelet, n_components=n_components))
    sample = input_train.features[0]
    extracted_basis = basis._get_1d_basis(sample)
    assert isinstance(extracted_basis, np.ndarray)
