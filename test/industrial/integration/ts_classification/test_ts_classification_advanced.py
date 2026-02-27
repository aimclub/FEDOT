import golem.core.log
from fedot_ind.api.main import FedotIndustrial
from fedot_ind.api.utils.checkers_collections import ApiConfigCheck
from fedot_ind.core.repository.config_repository import DEFAULT_CLF_API_CONFIG
from fedot_ind.tools.synthetic.ts_datasets_generator import TimeSeriesDatasetsGenerator


def mock_message(self, msg: str, **kwargs):
    level = 40
    self.log(level, msg, **kwargs)


def test_federated_clf(monkeypatch):
    # monkeypatch golem message function
    monkeypatch.setattr(golem.core.log.LoggerAdapter, 'message', mock_message)

    config = dict(task='classification',
                  metric='f1',
                  timeout=5,
                  n_jobs=2,
                  industrial_strategy='federated_automl',
                  industrial_strategy_params={},
                  logging_level=20)
    api_config = ApiConfigCheck().update_config_with_kwargs(DEFAULT_CLF_API_CONFIG,
                                                            **config)

    # Huge synthetic dataset for experiment
    train_data, test_data = TimeSeriesDatasetsGenerator(num_samples=1800,
                                                        task='classification',
                                                        max_ts_len=50,
                                                        binary=True,
                                                        test_size=0.5,
                                                        multivariate=False).generate_data()

    industrial = FedotIndustrial(**api_config)
    industrial.fit(train_data)
    predict = industrial.predict(test_data)

    assert predict is not None
