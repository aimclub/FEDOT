import pytest
from fedot.core.data.data import InputData

from fedot_ind.core.operation.dummy.dummy_operation import init_input_data
from fedot_ind.core.architecture.preprocessing.data_convertor import FedotConverter, CustomDatasetCLF
from fedot_ind.tools.synthetic.ts_datasets_generator import TimeSeriesDatasetsGenerator


@pytest.fixture
def data():
    ts_generator = TimeSeriesDatasetsGenerator()
    train_data, test_data = ts_generator.generate_data()
    return train_data, test_data


def test_fedot_converter(data):
    train_data, test_data = data
    converter = FedotConverter(data=train_data)

    assert isinstance(converter.input_data, InputData)


def get_ts_data(task):
    ts_generator = TimeSeriesDatasetsGenerator(task=task)
    train_data, test_data = ts_generator.generate_data()
    return (init_input_data(train_data[0], train_data[1]),
            init_input_data(test_data[0], test_data[1]))


def test_custom_dataset_clf():
    train, test = get_ts_data(task='classification')
    dataset = CustomDatasetCLF(ts=train)
    assert len(dataset) == len(train.target)
