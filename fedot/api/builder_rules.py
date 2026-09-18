from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class BuilderParamsUpdate:
    applied_params: Dict[str, Any]


def normalize_builder_params(new_params: Dict[str, Any], default_value: Any) -> BuilderParamsUpdate:
    applied_params = {key: value for key,
                      value in new_params.items() if value != default_value}
    return BuilderParamsUpdate(applied_params=applied_params)


def merge_builder_params(current_params: Dict[str, Any],
                         new_params: Dict[str, Any],
                         default_value: Any) -> Dict[str, Any]:
    normalized_update = normalize_builder_params(new_params, default_value)
    merged = dict(current_params)
    merged.update(normalized_update.applied_params)
    return merged


def build_fedot_kwargs(api_params: Dict[str, Any]) -> Dict[str, Any]:
    return dict(api_params)
