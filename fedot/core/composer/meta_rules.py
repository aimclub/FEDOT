from typing import Dict, Optional

from fedot.core.data.data import InputData


def get_cv_folds_number(input_data: InputData) -> Dict[str, Optional[int]]:
    """ Cross-validation folds are available from 1 to 10. """
    str_num = input_data.features.shape[0]
    available_cv_folds = list(reversed(range(1, 10)))
    for num in available_cv_folds:
        if str_num/num > 400:
            return {'cv_folds': num}
    return {'cv_folds': 1}


def get_recommended_preset(input_data: InputData, input_params: Dict) -> Dict[str, Optional[str]]:
    """ Get appropriate preset for `input_data` and `input_params`. """
    preset = None
    # to avoid overfitting for small datasets
    if input_data.features.shape[0] < 500:
        preset = 'fast_train'
    # to avoid stagnation due to too long evaluation of one population
    if input_params['timeout'] < 10 \
            and input_data.features.shape[0]*input_data.features.shape[1] > 300000:
        preset = 'fast_train'
    return {'preset': preset}


def get_early_stopping_generations(input_params: Dict) -> Dict[str, Optional[int]]:
    """ Get number of early stopping generations depending on timeout. """
    # If early_stopping_generations is not specified,
    # than estimate it as in time-based manner as: 0.33 * composing_timeout.
    # The minimal number of generations is 5.
    early_stopping_iterations = None
    if 'early_stopping_iterations' not in input_params:
        if input_params['timeout']:
            depending_on_timeout = int(input_params['timeout'] / 3)
            early_stopping_iterations = depending_on_timeout if depending_on_timeout > 5 else 5
    return {'early_stopping_iterations': early_stopping_iterations}
