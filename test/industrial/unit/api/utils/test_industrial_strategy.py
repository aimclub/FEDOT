import golem
import shutil
import os
import pytest
from fedot_ind.api.main import FedotIndustrial
from fedot_ind.core.repository.config_repository import DEFAULT_CLF_AUTOML_CONFIG, \
    DEFAULT_COMPUTE_CONFIG, DEFAULT_CLF_LEARNING_CONFIG, DEFAULT_AUTOML_LEARNING_CONFIG

from fedot_ind.tools.serialisation.path_lib import PROJECT_PATH
from tests.unit.api.fixtures import get_industrial_params, get_data_by_task

STRATEGY = ['federated_automl', 'lora_strategy',
            'kernel_automl',
            'forecasting_assumptions', 'forecasting_exogenous']


INDUSTRIAL_PARAMS = get_industrial_params()
DEFAULT_AUTOML_LEARNING_CONFIG['timeout'] = 0.1

CONFIGS = {
    # 'federated_automl': {'industrial_config': {'problem': 'classification',
    #                                            'strategy': 'federated_automl',
    #                                            'strategy_params': INDUSTRIAL_PARAMS},
    #                      'learning_config': DEFAULT_CLF_LEARNING_CONFIG,
    #                      'automl_config': DEFAULT_CLF_AUTOML_CONFIG,
    #                      'compute_config': DEFAULT_COMPUTE_CONFIG},

    'lora_strategy': {'industrial_config': {'problem': 'classification',
                                            'strategy': 'lora_strategy',
                                            'strategy_params': INDUSTRIAL_PARAMS},
                      'learning_config': DEFAULT_CLF_LEARNING_CONFIG,
                      'automl_config': DEFAULT_CLF_AUTOML_CONFIG,
                      'compute_config': DEFAULT_COMPUTE_CONFIG},

    'kernel_automl': {'industrial_config': {'problem': 'classification',
                                            'strategy': 'kernel_automl',
                                            'strategy_params': INDUSTRIAL_PARAMS},
                      'learning_config': DEFAULT_CLF_LEARNING_CONFIG,
                      'automl_config': DEFAULT_CLF_AUTOML_CONFIG,
                      'compute_config': DEFAULT_COMPUTE_CONFIG},

    # 'forecasting_assumptions': {'industrial_config': {'problem': 'ts_forecasting',
    #                                                   'strategy': 'forecasting_assumptions',
    #                                                   'strategy_params': INDUSTRIAL_PARAMS,
    #                                                   'task_params': {'forecast_length': NUM_SAMPLES}},
    #                             'learning_config': DEFAULT_REG_LEARNING_CONFIG,
    #                             'automl_config': {**DEFAULT_TSF_AUTOML_CONFIG,
    #                                               'task_params': {'forecast_length': NUM_SAMPLES}},
    #                             'compute_config': DEFAULT_COMPUTE_CONFIG},

    # 'forecasting_exogenous': {}
}


def mock_message(self, msg: str, **kwargs):
    level = 40
    self.log(level, msg, **kwargs)


@pytest.mark.parametrize('strategy', STRATEGY)
def test_custom_strategy(strategy, monkeypatch):
    monkeypatch.setattr(golem.core.log.LoggerAdapter, 'message', mock_message)
    if strategy in CONFIGS.keys():
        # clear cache before execution
        cache_folder = os.path.join(PROJECT_PATH, 'cache')
        if os.path.exists(cache_folder):
            shutil.rmtree(cache_folder)

        cnfg = CONFIGS[strategy]
        train_data, test_data = map(lambda x: (x[0].values, x[1]),
                                    get_data_by_task(cnfg['industrial_config']['problem']))
        n_samples = train_data[0].shape[0]

        industrial = FedotIndustrial(**cnfg)
        assert industrial.manager.industrial_config.strategy is not None
        train_data = (train_data[0], train_data[1].reshape(-1, 1))
        industrial.fit(train_data)
        predict = industrial.predict(test_data)

        assert predict is not None
        assert predict.shape[0] == n_samples
