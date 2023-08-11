import pytest

from fedot.core.utils import set_random_seed


@pytest.fixture(scope='session', autouse=True)
def establish_seed():
    set_random_seed(42)
