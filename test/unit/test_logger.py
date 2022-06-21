import os

import pytest

from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.log import Log, default_log, SingletonMeta
from fedot.core.operations.model import Model
from fedot.core.utils import DEFAULT_PARAMS_STUB, default_fedot_data_dir


@pytest.fixture()
def get_config_file():
    test_file_path = str(os.path.dirname(__file__))
    file = os.path.join(test_file_path, '../data', 'logging.json')
    if os.path.exists(file):
        return file


@pytest.fixture()
def get_bad_config_file():
    test_file_path = str(os.path.dirname(__file__))
    file = os.path.join(test_file_path, '../data', 'bad_log_config_file.yml')
    if os.path.exists(file):
        return file


def release_log(logger, log_file):
    logger.release_handlers()
    if os.path.exists(log_file):
        os.remove(log_file)


def test_default_logger_setup_correctly():
    SingletonMeta._instances = {}
    expected_logger_info_level = 10
    test_default_log = default_log('default_test_logger')

    assert test_default_log.logger.getEffectiveLevel() == expected_logger_info_level


@pytest.mark.parametrize('data_fixture', ['get_config_file'])
def test_logger_from_config_file_setup_correctly(data_fixture, request):
    SingletonMeta._instances = {}
    expected_logger_error_level = 40
    test_config_file = request.getfixturevalue(data_fixture)
    log = Log('test_logger', config_json_file=test_config_file)

    assert log.logger.getEffectiveLevel() == expected_logger_error_level


def test_logger_write_logs_correctly():
    SingletonMeta._instances = {}
    test_file_path = str(os.path.dirname(__file__))

    # Model data preparation
    file = os.path.join('../data', 'advanced_classification.csv')
    data = InputData.from_csv(os.path.join(test_file_path, file))
    train_data, test_data = train_test_data_setup(data=data)

    try:
        knn = Model(operation_type='knnreg')
        model, _ = knn.fit(params=DEFAULT_PARAMS_STUB, data=train_data, is_fit_pipeline_stage=True)
    except Exception:
        print('Captured error')

    default_log_path = os.path.join(default_fedot_data_dir(), 'log.log')

    if os.path.exists(default_log_path):
        with open(default_log_path, 'r') as file:
            content = file.readlines()

    # Is there a required message in the logs
    assert any('Can not find evaluation strategy' in log_message for log_message in content)


@pytest.mark.parametrize('data_fixture', ['get_bad_config_file'])
def test_logger_from_config_file_raise_exception(data_fixture, request):
    SingletonMeta._instances = {}
    test_bad_config_file = request.getfixturevalue(data_fixture)

    with pytest.raises(Exception) as exc:
        assert Log('test_logger', config_json_file=test_bad_config_file)

    assert 'Can not open the log config file because of' in str(exc.value)


def test_logger_str():
    SingletonMeta._instances = {}
    test_default_log = default_log('default_test_logger_for_str')

    expected_str_value = "Log object for default_test_logger_for_str module"

    assert str(test_default_log) == expected_str_value


def test_logger_repr():
    SingletonMeta._instances = {}
    test_default_log = default_log('default_test_logger_for_repr')

    expected_repr_value = "Log object for default_test_logger_for_repr module"

    assert repr(test_default_log) == expected_repr_value
