import os
from pathlib import Path

from fedot.core.utils import default_fedot_data_dir


def test_default_fedot_data_dir():
    default_fedot_data_dir()
    assert 'Fedot' in os.listdir(str(Path.home()))
