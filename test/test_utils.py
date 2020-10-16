from core.utils import default_fedot_data_dir
from pathlib import Path
import os


def test_default_fedot_data_dir():
    default_fedot_data_dir()
    assert 'Fedot' in os.listdir(str(Path.home()))
    os.rmdir(default_fedot_data_dir())
