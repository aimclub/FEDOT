import logging

import pytest

from fedot.core.log import Log, default_log


@pytest.fixture(scope='session', autouse=True)
def disable_test_logging():
    """Workaround for issue #765 (https://github.com/nccr-itmo/FEDOT/issues/765)
    Completely disable all logging before tests.
    """
    default_log(logging_level=logging.CRITICAL+1, write_logs=False)
