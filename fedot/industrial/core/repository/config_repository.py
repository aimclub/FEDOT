from enum import Enum


class ComputeConfigConstant(Enum):
    DEFAULT_COMPUTE_CONFIG = {'backend': 'cpu',
                              'distributed': dict(processes=False,
                                                  n_workers=1,
                                                  threads_per_worker=1,
                                                  memory_limit=0.3
                                                  ),
                              'output_folder': './results',
                              'use_cache': None,
                              'automl_folder': {'optimisation_history': './results/opt_hist',
                                                'composition_results': './results/comp_res'}}
    DEFAULT_COMPUTE_CONFIG_GPU = {'backend': 'gpu',
                                  'distributed': dict(processes=False,
                                                      n_workers=1,
                                                      threads_per_worker=1,
                                                      memory_limit=0.3
                                                      ),
                                  'output_folder': './results',
                                  'use_cache': None,
                                  'automl_folder': {'optimisation_history': './results/opt_hist',
                                                    'composition_results': './results/comp_res'}}


class AutomlLearningConfigConstant(Enum):
    DEFAULT_AUTOML_CONFIG = dict(timeout=10,
                                 pop_size=5,
                                 early_stopping_iterations=10,
                                 early_stopping_timeout=10,
                                 with_tuning=False,
                                 n_jobs=-1)


class AutomlConfigConstant(Enum):
    DEFAULT_SUBCONFIG = {'use_automl': True,
                         'optimisation_strategy': {'optimisation_strategy':
                                                   {'mutation_agent': 'random',
                                                    'mutation_strategy': 'growth_mutation_strategy'},
                                                   'optimisation_agent': 'Industrial'}}
    DEFAULT_CLF_AUTOML_CONFIG = {'task': 'classification', **DEFAULT_SUBCONFIG}
    DEFAULT_REG_AUTOML_CONFIG = {'task': 'regression', **DEFAULT_SUBCONFIG}
    DEFAULT_TSF_AUTOML_CONFIG = {'task': 'ts_forecasting', 'task_params': {'forecast_length': 14}, **DEFAULT_SUBCONFIG}


class LearningConfigConstant(Enum):
    DEFAULT_SUBCONFIG = {'learning_strategy': 'from_scratch',
                         'learning_strategy_params': AutomlLearningConfigConstant.DEFAULT_AUTOML_CONFIG.value}
    DEFAULT_CLF_LEARNING_CONFIG = {'optimisation_loss': {'quality_loss': 'accuracy'}, **DEFAULT_SUBCONFIG}
    DEFAULT_REG_LEARNING_CONFIG = {'optimisation_loss': {'quality_loss': 'rmse'}, **DEFAULT_SUBCONFIG}
    DEFAULT_TSF_LEARNING_CONFIG = {'optimisation_loss': {'quality_loss': 'rmse'}, **DEFAULT_SUBCONFIG}
    TASK_MAPPING = {
        'classification': {
            'task': 'classification',
            'use_automl': True,
            'optimisation_strategy': {
                'optimisation_strategy': {
                    'mutation_agent': 'random',
                    'mutation_strategy': 'growth_mutation_strategy'},
                'optimisation_agent': 'Industrial'}},
        'regression': {
            'task': 'regression',
                    'use_automl': True,
                    'optimisation_strategy': {
                        'optimisation_strategy': {
                            'mutation_agent': 'random',
                            'mutation_strategy': 'growth_mutation_strategy'},
                        'optimisation_agent': 'Industrial'}},
        'ts_forecasting': {
            'task': 'ts_forecasting',
            'use_automl': True,
            'task_params': {
                'forecast_length': 14},
            'optimisation_strategy': {
                'optimisation_strategy': {
                    'mutation_agent': 'random',
                    'mutation_strategy': 'growth_mutation_strategy'},
                'optimisation_agent': 'Industrial'}}}


class IndustrialConfigConstant(Enum):
    DEFAULT_CLF_INDUSTRIAL_CONFIG = {'problem': 'classification'}
    DEFAULT_REG_INDUSTRIAL_CONFIG = {'problem': 'regression'}
    DEFAULT_TSF_INDUSTRIAL_CONFIG = {'problem': 'ts_forecasting',
                                     'task_params': {'forecast_length': 14}}


DEFAULT_AUTOML_LEARNING_CONFIG = AutomlLearningConfigConstant.DEFAULT_AUTOML_CONFIG.value
DEFAULT_COMPUTE_CONFIG = ComputeConfigConstant.DEFAULT_COMPUTE_CONFIG.value
DEFAULT_COMPUTE_CONFIG_GPU = ComputeConfigConstant.DEFAULT_COMPUTE_CONFIG_GPU.value
DEFAULT_CLF_AUTOML_CONFIG = AutomlConfigConstant.DEFAULT_CLF_AUTOML_CONFIG.value
DEFAULT_REG_AUTOML_CONFIG = AutomlConfigConstant.DEFAULT_REG_AUTOML_CONFIG.value
DEFAULT_TSF_AUTOML_CONFIG = AutomlConfigConstant.DEFAULT_TSF_AUTOML_CONFIG.value

DEFAULT_CLF_LEARNING_CONFIG = LearningConfigConstant.DEFAULT_CLF_LEARNING_CONFIG.value
DEFAULT_REG_LEARNING_CONFIG = LearningConfigConstant.DEFAULT_REG_LEARNING_CONFIG.value
DEFAULT_TSF_LEARNING_CONFIG = LearningConfigConstant.DEFAULT_TSF_LEARNING_CONFIG.value

DEFAULT_CLF_INDUSTRIAL_CONFIG = IndustrialConfigConstant.DEFAULT_CLF_INDUSTRIAL_CONFIG.value
DEFAULT_REG_INDUSTRIAL_CONFIG = IndustrialConfigConstant.DEFAULT_REG_INDUSTRIAL_CONFIG.value
DEFAULT_TSF_INDUSTRIAL_CONFIG = IndustrialConfigConstant.DEFAULT_TSF_INDUSTRIAL_CONFIG.value

DEFAULT_CLF_API_CONFIG = {'industrial_config': DEFAULT_CLF_INDUSTRIAL_CONFIG,
                          'automl_config': DEFAULT_CLF_AUTOML_CONFIG,
                          'learning_config': DEFAULT_CLF_LEARNING_CONFIG,
                          'compute_config': DEFAULT_COMPUTE_CONFIG}

DEFAULT_REG_API_CONFIG = {'industrial_config': DEFAULT_REG_INDUSTRIAL_CONFIG,
                          'automl_config': DEFAULT_REG_AUTOML_CONFIG,
                          'learning_config': DEFAULT_REG_LEARNING_CONFIG,
                          'compute_config': DEFAULT_COMPUTE_CONFIG}

DEFAULT_TSF_API_CONFIG = {'industrial_config': DEFAULT_TSF_INDUSTRIAL_CONFIG,
                          'automl_config': DEFAULT_TSF_AUTOML_CONFIG,
                          'learning_config': DEFAULT_TSF_LEARNING_CONFIG,
                          'compute_config': DEFAULT_COMPUTE_CONFIG}

TASK_MAPPING = LearningConfigConstant.TASK_MAPPING.value
