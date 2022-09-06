import logging
from importlib import reload

import pytest

from fedot.core.utilities.singleton_meta import SingletonMeta


@pytest.fixture(autouse=True)
def cleanup_singletons():
    saved_classes = list(SingletonMeta._instances)
    for cls in saved_classes:
        del SingletonMeta._instances[cls]

    # and reboot logging
    logging.shutdown()
    reload(logging)
