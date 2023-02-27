from typing import Dict, Optional

from fedot.core.data.data import InputData


def get_cv_folds_number(input_data: InputData) -> Dict[str, Optional[int]]:
    if input_data.features.shape[0] < 200:
        cv_folds = 1
    else:
        cv_folds = None
    return {'cv_folds': cv_folds}


def get_recommended_preset(input_data: InputData) -> Dict[str, Optional[str]]:
    if input_data.features.shape[0] < 200:
        preset = 'fast_train'
    else:
        preset = None
    return {'preset': preset}
