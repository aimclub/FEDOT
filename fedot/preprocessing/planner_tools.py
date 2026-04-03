import torch
from typing import Optional, Dict, List, Union

from golem.core.dag.convert import graph_structure_as_nx_graph

from fedot.core.backend.backend import backend
from fedot.core.data.complex_types import ArrayType, IndexType
# from fedot.core.data.tensordata import TensorData
from fedot.core.data.tools import StateEnum
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.preprocessing.structure import DEFAULT_SOURCE_NAME, PipelineStructureExplorer
from fedot.preprocessing.preprocessor_types import (PreprocessingStep, ImputationMethodEnum, 
                                                    PreprocessingStepEnum, EmbeddingMethodEnum, 
                                                    EncodingMethodEnum)
from fedot.core.data.data_tools import get_idx_from_features_names, convert_idx_to_array


# def data_type_is_tabular(data: TensorData) -> bool:
#     return data.data_type is DataTypesEnum.tabular


def has_nan_func(features: torch.Tensor) -> bool:
    """
    Check whether a torch.Tensor contains any NaN values.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor.

    Returns
    -------
    bool
        True if tensor contains at least one NaN, False otherwise.
    """
    return torch.isnan(features).any().item()


def is_imputation_needed(features: torch.Tensor, pipeline) -> bool:
    has_nan = has_nan_func(features)
    if pipeline is None:
        return has_nan
    has_imputation_operation = PipelineStructureExplorer.check_structure_by_tag(
            pipeline, tag_to_check='imputation')
    return has_nan and not has_imputation_operation


def get_nan_columns(features: torch.Tensor):
    """
    """
    nan_mask = torch.isnan(features)
    cols_with_nan = nan_mask.any(dim=0)
    indices = torch.where(cols_with_nan)[0]
    return indices


def preprocess_imputation_params(features: torch.Tensor, params: Optional[List]=None) -> PreprocessingStep:

    method = None

    if params is not None:
        steps = []
        for param in params:
                    
            method = param["method"]
            
            if "features_idx" in param.keys():
                features_idx = param["features_idx"]
                step = PreprocessingStep(PreprocessingStepEnum.imputation, method, features_idx)
            else:
                break

            steps.append(step)

        if len(steps) > 0:
            return steps
        
    # TODO: default method is simple
    method = ImputationMethodEnum.moda if method is None else method
    features_idx = get_nan_columns(features)
    step = PreprocessingStep(PreprocessingStepEnum.imputation, method, features_idx)

    return step


def get_imputation_step(features: torch.Tensor, pipeline=None, params=None) -> PreprocessingStep:
    if not is_imputation_needed(features, pipeline):
        return None
    
    imputation_step = preprocess_imputation_params(features, params)
    return imputation_step


def preprocess_optional_params(step_name: PreprocessingStepEnum, features: torch.Tensor, params: List) -> List[PreprocessingStep]:
    steps = []

    for param in params:
        step = PreprocessingStep(step_name, param['method'], param['features_idx'])
        steps.append(step)

    return steps


def get_optional_step(step_name: PreprocessingStepEnum, features: torch.Tensor, pipeline=None, params=None) -> PreprocessingStep:
    if step_name == PreprocessingStepEnum.imputation:
        return get_imputation_step(features, pipeline, params)
    
    steps = preprocess_optional_params(step_name,features, params)

    return steps


def get_embedding_step(parameters: Union[Dict], features_names: Optional[List[str]] = None) -> PreprocessingStep:
    """
    """

    if parameters is None:
        return None

    if isinstance(parameters, Dict):

        DEFAULT_PARAMS = {
            "method": EmbeddingMethodEnum.transformer,
            "model_name": "all-distilroberta-v1",
            "batch_size": 32,
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            "state": StateEnum.FIT
        }
        params = {**DEFAULT_PARAMS, **parameters}

        features_idx = get_idx_from_features_names(parameters["features_idx"], features_names)
        features_idx = convert_idx_to_array(features_idx)

        if "model_hash" in params.keys():
            params["state"] = StateEnum.PREDICT

        step = PreprocessingStep(
            step=PreprocessingStepEnum.embedding,
            method=params["method"],
            features_idx=features_idx,
            state=params["state"],
            model_name=params["model_name"],
            batch_size=params["batch_size"],
            device=params["device"]
        )
            
        return step
    
    elif isinstance(parameters, PreprocessingStep):
        if parameters.state == StateEnum.PREDICT and parameters.model_hash is None:
            raise ValueError("Model hash is required for prediction")
        return parameters
    
    else:
        raise ValueError(f"Unsupported parameters type: {type(parameters)}")


# def force_categorical_determination(table: ArrayType) -> IndexType:
#     """
#     Detect categorical feature columns by checking for string/object dtypes.

#     Args:
#         table (ArrayType): Feature matrix `(n_samples, n_features)`.

#     Returns:
#         IndexType: Indices of detected categorical columns, or `None` if no categorical
#             columns are found.
#     """
#     pd_backend = backend.pd

#     categorical_ids = []

#     for column_id, column in enumerate(table.T):
#         series = pd_backend.Series(column)
#         if str(series.dtype) in ("object", "string"):
#             categorical_ids.append(column_id)

#     if len(categorical_ids) == 0:
#         return None

#     categorical_ids = convert_idx_to_array(categorical_ids)
#     return categorical_ids


def force_categorical_determination(table: ArrayType) -> IndexType:
    """
    Detect categorical feature columns.

    A column is treated as categorical if:
    1. it has object/string-like dtype or contains python string objects, and
    2. it cannot be safely converted to numeric values.

    This handles the case when the whole numpy array has dtype=object because
    at least one column contains strings.

    Args:
        table (ArrayType): Feature matrix of shape (n_samples, n_features).

    Returns:
        IndexType: Indices of detected categorical columns, or None if no
            categorical columns are found.
    """
    pd_backend = backend.pd

    categorical_ids = []

    for column_id, column in enumerate(table.T):
        series = pd_backend.Series(column)

        # Fast path for explicitly numeric pandas dtypes
        if pd_backend.api.types.is_numeric_dtype(series):
            continue

        # Try to coerce the whole column to numeric.
        # If it succeeds, this is not a categorical column even if the source
        # numpy array had dtype=object.
        numeric_series = pd_backend.to_numeric(series, errors='coerce')
        is_fully_numeric = numeric_series.notna().all()

        if is_fully_numeric:
            continue

        # If conversion to numeric failed, mark as categorical only when the
        # source data is actually object/string-like.
        has_object_or_string_dtype = (
            str(series.dtype) in ("object", "string")
            or pd_backend.api.types.is_object_dtype(series)
            or pd_backend.api.types.is_string_dtype(series)
        )

        has_string_values = series.map(lambda x: isinstance(x, str)).any()

        if has_object_or_string_dtype or has_string_values:
            categorical_ids.append(column_id)

    if not categorical_ids:
        return None

    return convert_idx_to_array(categorical_ids)


def preprocess_encoding_params(params: Dict, features: ArrayType, features_names: Optional[List[str]]) -> PreprocessingStep:
    
    if isinstance(params, Dict):
        if "method" in params.keys():
            method = params["method"]
        else:
            method = EncodingMethodEnum.label

        if "features_idx" in params.keys():
            features_idx = get_idx_from_features_names(params["features_idx"], features_names)
            features_idx = convert_idx_to_array(features_idx)
        else:
            features_idx = force_categorical_determination(features)

        if features_idx is None:
            return None
        
        if "model_hash" in params.keys():
            state = StateEnum.PREDICT
            model_hash = params["model_hash"]
        else:
            state = StateEnum.FIT
            model_hash = None

        step = PreprocessingStep(
            step=PreprocessingStepEnum.encoding,
            method=method,
            features_idx=features_idx,
            state=state,
            model_hash=model_hash
        )

        return step
    
    elif isinstance(params, PreprocessingStep):
        if params.state == StateEnum.PREDICT and params.model_hash is None:
            raise ValueError("Model hash is required for prediction")
        return params
    elif params is None:
        method = EncodingMethodEnum.label
        features_idx = force_categorical_determination(features)
        if features_idx is None:
            return None
        else:
            step = PreprocessingStep(
                step=PreprocessingStepEnum.encoding,
                method=method,
                features_idx=features_idx
            )
            return step
    else:
        raise ValueError(f"Unsupported parameters type: {type(params)}")


def get_encoding_steps(parameters: Dict, features: ArrayType, features_names: Optional[List[str]] = None) -> List[PreprocessingStep]:
    """
    """

    steps = []

    xp = backend.xp

    # if isinstance(features, torch.Tensor):
    #     features = xp.asnumpy(features)

    if isinstance(parameters, List):
        for param in parameters:
            step = preprocess_encoding_params(param, features, features_names)
            steps.append(step)
    else:
        step = preprocess_encoding_params(parameters, features,features_names)
        steps.append(step)

    return steps


def target_has_strings(target) -> bool:
    """
    """

    xp = backend.xp

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
            step=PreprocessingStepEnum.encoding,
            method=EncodingMethodEnum.label,
            features_idx=[0],
            state=StateEnum.FIT
        )
    
    else:
        step = None
    return step


def get_constant_idx(features, steps):
    xp = backend.xp

    changeable_idx = []

    if isinstance(steps, List):
        for step in steps:
            changeable_idx.extend(step["features_idx"])
    else:
        changeable_idx = steps["features_idx"]

    constant_idx = xp.setdiff1d(range(features.shape[1]), changeable_idx)

    return constant_idx
