from dataclasses import dataclass
from typing import Any, Iterable, Tuple

from fedot.core.data.data import InputData
from fedot.core.data.multi_modal import MultiModalData


@dataclass(frozen=True)
class PreprocessingSourcePlan:
    source_names: Tuple[str, ...]


@dataclass(frozen=True)
class PreprocessorMergePlan:
    take_pipeline_encoders: bool
    take_pipeline_imputers: bool


@dataclass(frozen=True)
class OptionalPreprocessingPlan:
    apply_imputation: bool
    apply_encoding: bool


def resolve_source_names(data: Any, default_source_name: str) -> PreprocessingSourcePlan:
    if isinstance(data, InputData):
        return PreprocessingSourcePlan(source_names=(default_source_name,))
    if isinstance(data, MultiModalData):
        return PreprocessingSourcePlan(source_names=tuple(data.keys()))
    raise ValueError('Unknown type of data.')


def should_initialize_source_helpers(has_binary_processors: bool, has_type_correctors: bool) -> bool:
    return not (has_binary_processors and has_type_correctors)


def resolve_main_target_source_name(current_source_name, multi_data: MultiModalData):
    if current_source_name is not None:
        return current_source_name

    for data_source_name, input_data in multi_data.items():
        if input_data.supplementary_data.is_main_target:
            return data_source_name
    return None


def resolve_target_encoder_source_name(current_source_name, default_source_name: str) -> str:
    return current_source_name if current_source_name is not None else default_source_name


def iter_preprocessed_inputs(data: Any) -> Tuple[Any, ...]:
    if isinstance(data, InputData):
        return (data,)
    if isinstance(data, MultiModalData):
        return tuple(data.values())
    raise ValueError('Unknown type of data.')


def build_preprocessor_merge_plan(use_auto_preprocessing: bool,
                                  api_features_encoders,
                                  api_features_imputers) -> PreprocessorMergePlan:
    if use_auto_preprocessing:
        return PreprocessorMergePlan(
            take_pipeline_encoders=False,
            take_pipeline_imputers=False,
        )

    return PreprocessorMergePlan(
        take_pipeline_encoders=not bool(api_features_encoders),
        take_pipeline_imputers=not bool(api_features_imputers),
    )


def build_optional_preprocessing_plan(has_missing_values: bool,
                                      has_categorical_features: bool,
                                      has_imputation_operation: bool,
                                      has_encoding_operation: bool) -> OptionalPreprocessingPlan:
    return OptionalPreprocessingPlan(
        apply_imputation=has_missing_values and not has_imputation_operation,
        apply_encoding=has_categorical_features and not has_encoding_operation,
    )
