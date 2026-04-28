import torch
from typing import Optional, Dict, List, Union

from fedot.core.backend.backend import Backend
from fedot.core.data.complex_types import ArrayType, IndexType
from fedot.core.data.tools import StateEnum
from fedot.preprocessing.tools.preprocessor_types import (PreprocessingStep,
                                                    PreprocessingStepEnum, EmbeddingMethodEnum, 
                                                    EncodingMethodEnum)
from fedot.core.data.data_tools import get_idx_from_features_names, convert_idx_to_list
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.preprocessing.tools.index_mapping_tools import update_indices
from fedot.preprocessing.planner.planner import PreprocessingPlan


def get_embedding_steps(parameters: Optional[List], 
                       features_names: Optional[List[str]] = None,
                       feature_type: Optional[DataTypesEnum] = None) -> PreprocessingStep:
    """
    """
    if feature_type == DataTypesEnum.ts:
        return None

    if parameters is None:
        return None
    
    features_idx_detected = all(param.get("features_idx") is not None for param in parameters)
    if not features_idx_detected or features_idx_detected is None:
        raise ValueError("Features indexes is required for embedding strategy")

    steps = []
    for param in parameters:
        DEFAULT_PARAMS = {
            "method": EmbeddingMethodEnum.transformer,
            "model_name": "all-distilroberta-v1",
            "batch_size": 32,
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            "state": StateEnum.FIT
        }
        params = {**DEFAULT_PARAMS, **param}

        features_idx = get_idx_from_features_names(params["features_idx"], features_names)
        features_idx = convert_idx_to_list(features_idx)

        if "model_hash" in params.keys():
            params["state"] = StateEnum.PREDICT

        step = PreprocessingStep(
            step=PreprocessingStepEnum.embedding,
            method=params["method"],
            features_idx=features_idx,
            state=params["state"],
            step_args={
                "model_name": params["model_name"],
                "batch_size": params["batch_size"],
                "device": params["device"]
            }
        )
        steps.append(step)

    return steps


def force_categorical_determination(table: ArrayType) -> IndexType:
    """
    Detect categorical feature columns.

    A column is treated as categorical if:
    1. it has object/string-like dtype or contains python string objects, and
    2. among non-missing values, it cannot be safely converted to numeric values.

    Missing values (None, np.nan, pd.NA, etc.) do not make a column categorical.
    """
    pd_backend = Backend().pd

    categorical_ids = []

    for column_id, column in enumerate(table.T):
        series = pd_backend.Series(column)

        if pd_backend.api.types.is_numeric_dtype(series):
            continue

        numeric_series = pd_backend.to_numeric(series, errors="coerce")

        original_notna = ~pd_backend.isna(series)
        converted_notna = ~pd_backend.isna(numeric_series)

        is_fully_numeric = converted_notna[original_notna].all()

        if is_fully_numeric:
            continue

        has_object_or_string_dtype = (
            str(series.dtype) in ("object", "string")
            or pd_backend.api.types.is_object_dtype(series)
            or pd_backend.api.types.is_string_dtype(series)
        )

        has_string_values = series.map(
            lambda x: isinstance(x, str) and x.strip().lower() not in {"nan", "none", "null", "na", "n/a", ""}
        ).any()

        if has_object_or_string_dtype or has_string_values:
            categorical_ids.append(column_id)

    if not categorical_ids:
        return []

    return convert_idx_to_list(categorical_ids)


def preprocess_encoding_params(params: Dict, 
                               features: ArrayType, 
                               features_names: Optional[List[str]]
                            ) -> PreprocessingStep:
    
    if params is not None and not isinstance(params, Dict):
        raise ValueError(f"Unsupported parameters type: {type(params)}")

    DEFAULT_PARAMS = {
        "method": EncodingMethodEnum.label,
        "features_idx": force_categorical_determination(features),
    }

    params = {**DEFAULT_PARAMS, **(params or {})}

    if params["features_idx"] is not None:
        features_idx = get_idx_from_features_names(
            params["features_idx"],
            features_names
        )
        features_idx = convert_idx_to_list(features_idx)
    else:
        features_idx = None

    if features_idx is None:
        return None

    step = PreprocessingStep(
        step=PreprocessingStepEnum.encoding,
        method=params["method"],
        features_idx=features_idx,
    )

    return step


def get_encoding_steps(features: ArrayType, 
                       parameters: Optional[List],
                       features_names: Optional[List[str]] = None,
                       feature_type: Optional[DataTypesEnum] = None,
                       used_idx: Optional[List[int]] = None) -> List[PreprocessingStep]:
    """
    """
    if feature_type == DataTypesEnum.ts:
        return None
    
    if parameters is None:
        parameters = list()
    parameters.append(None)

    if len(parameters) > 2:
        features_idx_detected = all(
            param.get("features_idx") is not None
            for param in parameters
            if param is not None
        )
        if (not features_idx_detected or features_idx_detected is None) and len(parameters) > 1:
            raise ValueError("More than one encoding step should have features_idx parameter")

    cat_idx = force_categorical_determination(features)
    steps = []
    for param in parameters:
        remainder_idx = list(set(cat_idx) - set(used_idx))

        DEFAULT_PARAMS = {
            "method": EncodingMethodEnum.label,
            "features_idx": remainder_idx
        }

        params = {**DEFAULT_PARAMS, **(param or {})}

        features_idx = get_idx_from_features_names(
            params["features_idx"],
            features_names
        )
        features_idx = convert_idx_to_list(features_idx)

        if len(features_idx) == 0:
            step = None
        else:
            step = PreprocessingStep(
                step=PreprocessingStepEnum.encoding,
                method=params["method"],
                features_idx=features_idx,
            )
            used_idx.extend(features_idx)

        steps.append(step)
    return steps


def target_has_strings(target) -> bool:
    """
    """

    xp = Backend().xp

    if xp.issubdtype(target.dtype, xp.number):
        return False

    if xp.issubdtype(target.dtype, xp.str_) or xp.issubdtype(target.dtype, xp.bytes_):
        return True

    if target.dtype == object:
        try:
            iterable = target.ravel()
        except Exception:
            iterable = xp.asnumpy(target).ravel()

        return any(isinstance(x, str) for x in iterable)

    return False


def get_target_encoding_step(target):

    if target is None:
        return None
    
    if target_has_strings(target):
        step = PreprocessingStep(
            step=PreprocessingStepEnum.target_encoding,
            method=EncodingMethodEnum.label,
            features_idx=[0],
            state=StateEnum.FIT
        )

    else:
        step = None
    return step


def get_constant_idx(features, steps):
    xp = Backend().xp

    changeable_idx = []

    if isinstance(steps, List):
        for step in steps:
            changeable_idx.extend(step["features_idx"])
    else:
        changeable_idx = steps["features_idx"]

    constant_idx = xp.setdiff1d(range(features.shape[1]), changeable_idx)

    return constant_idx


def check_idx(steps: List[PreprocessingStep], new_steps: List[PreprocessingStep]):
    old_idx = [idx for step in steps for idx in step.features_idx]
    
    new_idx = set()
    for step in new_steps:
        if len(new_idx.intersection(set(step.features_idx))) != 0:
            raise ValueError("Features idx should be unique")
        new_idx.update(set(step.features_idx))

    if len(new_idx.intersection(set(old_idx))) != 0:
        raise ValueError("Features idx should be unique")


def add_steps(steps: List[PreprocessingStep], new_steps: List[PreprocessingStep]):
    if new_steps is None:
        return steps

    new_steps = [step for step in new_steps if step is not None]
    if len(new_steps) == 0:
        return steps
    
    if (steps is None) or len(steps) == 0:
        return new_steps
    
    check_idx(steps, new_steps)

    steps.extend(new_steps)
    return steps


def get_custom_steps(parameters: List, features_names: Optional[List[str]] = None) -> List[PreprocessingStep]:
    if parameters is None:
        return None

    steps = []
    for param in parameters:
        features_idx = param.get("features_idx")
        if features_idx is None:
            raise ValueError("Features indexes is required for custom strategy")
        
        features_idx = get_idx_from_features_names(features_idx, features_names)
        features_idx = convert_idx_to_list(features_idx)

        step = PreprocessingStep(PreprocessingStepEnum.custom, 
                                 param['method'], 
                                 features_idx,
                                 param['implementation'])
        if param['step_args'] is not None:
            step.step_args = param['step_args']
        steps.append(step)
    return steps


def build_obligatory_plan(features: ArrayType, 
                          target: ArrayType,
                          params: dict) -> PreprocessingPlan:
    obligatory_plan = PreprocessingPlan()

    steps = []

    custom_steps = get_custom_steps(params['custom_strategy'], params['features_names'])
    steps = add_steps(steps, custom_steps)

    embedding_steps = get_embedding_steps(
                    params['embedding_strategy'],
                    params['features_names'],
                    params['data_type'])
    steps = add_steps(steps, embedding_steps)

    used_idx = [idx for step in steps for idx in step.features_idx]
    encoding_steps = get_encoding_steps(
        features,
        params['encoding_strategy'],
        params['features_names'],
        params['data_type'],
        used_idx
    )
    steps = add_steps(steps, encoding_steps)

    target_enc_step = get_target_encoding_step(target)
    if target_enc_step is not None:
        steps.insert(0, target_enc_step)

    obligatory_plan.add_step(steps)
    return obligatory_plan
