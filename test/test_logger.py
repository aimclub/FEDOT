from core.log import Logger
import pytest
import os


@pytest.fixture()
def get_config_file():
    test_file_path = str(os.path.dirname(__file__))
    path = os.path.join(test_file_path, 'data', 'logging.json')
    if os.path.exists(path):
        return path


def test_loger_setup():
    logger = Logger('test_logger')
    assert logger.logger.level == 20


@pytest.mark.parametrize('data_fixture', ['get_config_file'])
def test_logger_setup_from_file(data_fixture, request):
    test_file = request.getfixturevalue(data_fixture)
    logger = Logger('test_logger', path=test_file)
    assert logger.logger.parent.level == 20
