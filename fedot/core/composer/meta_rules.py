from fedot.api.api_utils.presets import PresetsEnum
from golem.core.log import LoggerAdapter
from typing import Dict, Optional

from fedot.api.api_utils.params import ApiParams
from fedot.core.data.data import InputData


def get_cv_folds_number(input_data: InputData, log: LoggerAdapter) -> Dict[str, Optional[int]]:
    """ Cross-validation folds are available from 1 to 10. """
    row_num = input_data.features.shape[0]
    if row_num < 1000:
        cv_folds = None
    elif row_num < 20000:
        cv_folds = 3
    else:
        cv_folds = 5
    log.info(f'number of cv_folds param was set to {cv_folds}')
    return {'cv_folds': cv_folds}


def get_recommended_preset(input_data: InputData, input_params: ApiParams, log: LoggerAdapter) \
        -> Dict[str, Optional[str]]:
    """ Get appropriate preset for `input_data` and `input_params`. """
    preset = None

    if input_params.timeout:
        # since there is enough time for optimization on such amount of data heavy models can be used
        if input_params.timeout >= 60 and \
                input_data.features.shape[0] * input_data.features.shape[1] < 300000:
            preset = PresetsEnum.BEST_QUALITY

        # to avoid stagnation due to too long evaluation of one population
        if input_params.timeout < 10 \
                and input_data.features.shape[0] * input_data.features.shape[1] > 300000:
            preset = PresetsEnum.FAST_TRAIN

    # to avoid overfitting for small datasets
    if input_data.features.shape[0] < 5000:
        preset = PresetsEnum.FAST_TRAIN

    if preset:
        log.info(f'preset was set to {preset}')
    return {'preset': preset}


def get_early_stopping_generations(input_params: ApiParams, log: LoggerAdapter) -> Dict[str, Optional[int]]:
    """ Get number of early stopping generations depending on timeout. """
    # If early_stopping_generations is not specified,
    # than estimate it as in time-based manner as: 0.33 * composing_timeout.
    # The minimal number of generations is 5.
    early_stopping_iterations = None
    if 'early_stopping_iterations' not in input_params:
        if input_params.timeout:
            depending_on_timeout = int(input_params.timeout / 3)
            early_stopping_iterations = depending_on_timeout if depending_on_timeout > 5 else 5
            log.info(f'early_stopping_iterations was set to {early_stopping_iterations}')
    return {'early_stopping_iterations': early_stopping_iterations}
