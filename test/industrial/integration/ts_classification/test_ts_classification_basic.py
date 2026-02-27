import pytest
import golem.core.log

from fedot.industrial.api.main import FedotIndustrial
from fedot.industrial.api.utils.checkers_collections import ApiConfigCheck
from fedot.industrial.core.architecture.settings.computational import backend_methods as np
from fedot.industrial.core.repository.config_repository import DEFAULT_CLF_API_CONFIG
from fedot.industrial.tools.loader import DataLoader


def mock_message(self, msg: str, **kwargs):
    level = 40
    self.log(level, msg, **kwargs)


@pytest.mark.parametrize('dataset_name',
                         ['Lightning7', 'Epilepsy'],
                         ids=['univariate', 'multivariate'])
def test_basic_tsc_test(dataset_name, monkeypatch):
    # monkeypatch golem message function
    monkeypatch.setattr(golem.core.log.LoggerAdapter, 'message', mock_message)

    train_data, test_data = DataLoader(dataset_name=dataset_name).load_data()
    config = dict(task='classification',
                  timeout=0.1,
                  n_jobs=-1)
    api_config = ApiConfigCheck().update_config_with_kwargs(DEFAULT_CLF_API_CONFIG,
                                                            **config)

    industrial = FedotIndustrial(**api_config)

    industrial.fit(train_data)
    labels = industrial.predict(test_data)
    probs = industrial.predict_proba(test_data)
    assert labels is not None
    assert probs is not None
    assert np.mean(probs) > 0
