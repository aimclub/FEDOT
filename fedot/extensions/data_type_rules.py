from dataclasses import dataclass
from typing import Tuple

from fedot.core.data.common.compatibility_rules import (
    to_input_compatible_data_type,
    to_tensor_canonical_data_type,
)
from fedot.core.repository.dataset_types import DataTypesEnum


@dataclass(frozen=True)
class ExtensionDataTypeView:
    input_types: Tuple[DataTypesEnum, ...]
    tensor_types: Tuple[DataTypesEnum, ...]
    preferred_output_type_name: str


def _deduplicate_preserving_order(items) -> Tuple[DataTypesEnum, ...]:
    result = []
    for item in items:
        if item not in result:
            result.append(item)
    return tuple(result)


def build_extension_data_type_view(data_types) -> ExtensionDataTypeView:
    if not data_types:
        input_types = (DataTypesEnum.table,)
        tensor_types = (DataTypesEnum.tabular,)
    else:
        input_types = _deduplicate_preserving_order(
            to_input_compatible_data_type(data_type)
            for data_type in data_types
        )
        tensor_types = _deduplicate_preserving_order(
            to_tensor_canonical_data_type(data_type)
            for data_type in data_types
        )

    preferred_output_type_name = input_types[0].name
    return ExtensionDataTypeView(
        input_types=input_types,
        tensor_types=tensor_types,
        preferred_output_type_name=preferred_output_type_name,
    )
