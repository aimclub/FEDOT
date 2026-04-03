from fedot.core.backend.backend import backend
from fedot.core.data.complex_types import ArrayType
from fedot.preprocessing.mapping import PREPROCESSING_OBLIGATORY_MAPPING
from fedot.preprocessing.preprocessor_types import (PreprocessingStep)
from fedot.core.data.tools import StateEnum


from typing import List, Optional



def apply_step(data: ArrayType, 
                               step: PreprocessingStep):
    

    if step.state == StateEnum.FIT:
        method = PREPROCESSING_OBLIGATORY_MAPPING[step.step][step.method]()
        features = method.fit_transform(data, step)
        step.state = StateEnum.PREDICT
        # TODO: save model
    
    else:
        # TODO: get model from cache by hash and transform
        method = ...
        features = method.transform(data)

    return features, step


def apply_obligatory_steps(data: ArrayType, 
                           steps: Optional[List[PreprocessingStep]] = None):

    if steps is None:
        return data, None
    
    if not isinstance(steps, List):
        steps = [steps]

    xp = backend.xp

    # if isinstance(features, torch.Tensor):
    #     features = xp.asnumpy(features)
    # constant_idx = get_constant_idx(data, steps)

    # result_data = data[:, constant_idx].copy()

    new_steps = [] # TODO: remove copy and make caching steps

    for step in steps:
        data, new_step = apply_step(data, step)
        # result_data = xp.hstack((result_data, prepared_features))
        new_steps.append(new_step)
    return data, new_steps
    # return result_data, new_steps
