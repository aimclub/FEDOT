from core.log import Logger, default_logger
import pytest
import os


@pytest.fixture()
def get_config_file():
    test_file_path = str(os.path.dirname(__file__))
    file = os.path.join(test_file_path, 'data', 'logging.json')
    if os.path.exists(file):
        return file


def test_default_logger():
    info_level_numeric = 20
    logger = default_logger('default_test_logger')

    assert logger.logger.level == info_level_numeric


@pytest.mark.parametrize('data_fixture', ['get_config_file'])
def test_logger_setup_from_file(data_fixture, request):
    error_level_numeric = 40
    test_file = request.getfixturevalue(data_fixture)
    logger = Logger('test_logger', config_json_file=test_file)
    root_logger = logger.logger.parent

    assert root_logger.level == error_level_numeric
