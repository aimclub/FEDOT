import pytest

from fedot.api.main import Fedot
from test.unit.api.test_main_api import get_dataset


def test_correct_api_preprocessing():
    """ Check if dataset preprocessing was performed correctly """
    train_data, test_data, threshold = get_dataset('classification')

    fedot_model = Fedot(problem='classification', check_mode=True)
    with pytest.raises(SystemExit) as exc:
        assert fedot_model.fit(train_data)
    assert str(exc.value) == f'Initial pipeline were fitted successfully'
