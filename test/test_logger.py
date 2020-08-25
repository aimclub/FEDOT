import os

import pytest

from core.log import Log, default_logger
from core.models.data import train_test_data_setup, InputData
from core.models.model import Model


@pytest.fixture()
def get_config_file():
    test_file_path = str(os.path.dirname(__file__))
    file = os.path.join(test_file_path, 'data', 'logging.json')
    if os.path.exists(file):
        return file


def tear_down(logger, log_file):
    logger.release_handlers()
    if os.path.exists(log_file):
        os.remove(log_file)


def test_default_logger_setup_correctly():
    expected_logger_info_level = 20
    logger = default_logger('default_test_logger')

    assert logger.logger.level == expected_logger_info_level


@pytest.mark.parametrize('data_fixture', ['get_config_file'])
def test_logger_from_config_file_setup_correctly(data_fixture, request):
    expected_logger_error_level = 40
    test_file = request.getfixturevalue(data_fixture)
    logger = Log('test_logger', config_json_file=test_file)
    root_logger = logger.logger.parent

    assert root_logger.level == expected_logger_error_level


def test_logger_write_logs_correctly():
    test_file_path = str(os.path.dirname(__file__))
    test_log_file = os.path.join(test_file_path, 'test_log.log')
    test_logger = default_logger('test_logger',
                                 log_file=test_log_file)

    # Model data preparation
    file = 'data/advanced_classification.csv'
    data = InputData.from_csv(os.path.join(test_file_path, file))
    train_data, test_data = train_test_data_setup(data=data)

    try:
        knn = Model(model_type='knnreg', logger=test_logger)
        model, _ = knn.fit(data=train_data)
    except Exception:
        print('Captured error')

    if os.path.exists(test_log_file):
        with open(test_log_file, 'r') as file:
            content = file.readlines()

    tear_down(logger=test_logger, log_file=test_log_file)
    assert 'Can not find evaluation strategy' in content[0]
