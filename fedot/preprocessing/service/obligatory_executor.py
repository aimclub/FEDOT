from fedot.core.data.complex_types import ArrayType
from fedot.core.data.prepared_data import PreparedData
from fedot.preprocessing.tools.methods_mapping import PREPROCESSING_OBLIGATORY_MAPPING
from fedot.preprocessing.tools.preprocessor_types import (PreprocessingStep)
from fedot.preprocessing.tools.index_mapping_tools import update_index_mapping, update_indices
from fedot.core.data.tools import StateEnum

from typing import List, Optional, Dict


def apply_step(data: PreparedData, 
               step: PreprocessingStep):

    step.features_idx = update_indices(data.idx_mapping, step.features_idx)
    old_mapping = data.idx_mapping
    data.new_cols_dict = None

    if step.state == StateEnum.FIT:
        method = PREPROCESSING_OBLIGATORY_MAPPING[step.step][step.method](**step.step_args)
        result_data = method.fit_transform(data, step.features_idx)

        result_data.idx_mapping = update_index_mapping(
            old_mapping,
            step.features_idx,
            result_data.features,
            result_data.new_cols_dict
        )

        step.state = StateEnum.PREDICT
        # TODO: save model
    
    else:
        # TODO: get model from cache by hash and transform
        method = ...
        features = method.transform(data)

    return result_data, step


def apply_obligatory_steps(features: ArrayType, 
                           steps: Optional[List[PreprocessingStep]] = None,
                           idx_mapping: Optional[Dict[int, int]] = None):

    if steps is None:
        return features, None, idx_mapping
    
    if not isinstance(steps, List):
        steps = [steps]

    new_steps = [] # TODO: remove copy and make caching steps

    prepared_data = PreparedData(features=features)
    prepared_data.features = features
    prepared_data.idx_mapping = idx_mapping

    for step in steps:
        prepared_data, new_step = apply_step(prepared_data, step)
        new_steps.append(new_step)
    return prepared_data.features, new_steps, prepared_data.idx_mapping
