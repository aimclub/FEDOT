import logging
import os
from importlib import reload
from pathlib import Path

import pytest

from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.log import DEFAULT_LOG_PATH, Log, default_log
from fedot.core.operations.model import Model
from fedot.core.utilities.singleton_meta import SingletonMeta


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


def clear_singleton_class(cls=Log):
    if cls in SingletonMeta._instances:
        del SingletonMeta._instances[cls]


@pytest.fixture(autouse=True)
def cleanup_log_singleton():
    clear_singleton_class(Log)
    # and reboot logging
    logging.shutdown()
    reload(logging)


def test_default_logger_setup_correctly():
    expected_logger_info_level = logging.INFO
    test_default_log = default_log(prefix='default_test_logger')

    assert test_default_log.logger.getEffectiveLevel() == expected_logger_info_level


@pytest.mark.parametrize('data_fixture', ['get_config_file'])
def test_logger_from_config_file_setup_correctly(data_fixture, request):
    expected_logger_error_level = logging.ERROR
    test_config_file = request.getfixturevalue(data_fixture)
    log = Log(config_json_file=test_config_file)

    assert log.logger.getEffectiveLevel() == expected_logger_error_level


def test_logger_write_logs_correctly():
    test_file_path = str(os.path.dirname(__file__))

    # Model data preparation
    file = os.path.join('../data', 'advanced_classification.csv')
    data = InputData.from_csv(os.path.join(test_file_path, file))
    train_data, test_data = train_test_data_setup(data=data)

    try:
        knn = Model(operation_type='knnreg')
        model, _ = knn.fit({}, data=train_data)
    except Exception as ex:
        print(f'Captured error: {ex}')

    content = ''
    if Path(DEFAULT_LOG_PATH).exists():
        content = Path(DEFAULT_LOG_PATH).read_text()

    # Is there a required message in the logs
    assert 'Can not find evaluation strategy' in content


@pytest.mark.parametrize('data_fixture', ['get_bad_config_file'])
def test_logger_from_config_file_raise_exception(data_fixture, request):
    test_bad_config_file = request.getfixturevalue(data_fixture)

    with pytest.raises(Exception) as exc:
        assert Log(config_json_file=test_bad_config_file)

    assert 'Can not open the log config file because of' in str(exc.value)


def test_log_str():
    logger_name = ''
    log = Log()

    assert logger_name in str(log)


def test_logger_adapter_str():
    prefix = 'default_prefix'
    test_default_log = default_log(prefix=prefix)

    assert prefix in str(test_default_log)


def test_multiple_adapters_with_one_prefix():
    """ Tests that messages are written correctly to log file if multiple adapters have the same prefix """
    log_1 = default_log(prefix='prefix_1')
    log_2 = default_log(prefix='prefix_1')

    info_1 = 'Info from log_1'
    log_1.info(info_1)
    info_2 = 'Info from log_2'
    log_2.info(info_2)

    content = ''
    if Path(DEFAULT_LOG_PATH).exists():
        content = Path(DEFAULT_LOG_PATH).read_text()

    assert f'prefix_1 - {info_1}' in content
    assert f'prefix_1 - {info_2}' in content
