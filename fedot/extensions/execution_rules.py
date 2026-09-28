"""Pure compatibility decisions for the TensorData adapter."""
from dataclasses import dataclass

from pymonad.either import Left, Right

from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.extensions.contracts import ArrayBackend, ExtensionError, ExternalModelSpec, OperationKind
from fedot.extensions.data_type_rules import build_extension_data_type_view
from fedot.extensions.validation import validate_operation_spec


@dataclass(frozen=True)
class ExecutionPlan:
    operation_name: str
    kind: OperationKind
    backend: ArrayBackend
    output_data_type: DataTypesEnum


def plan_execution(spec, task_type, data_type, *, fitting=False, has_target=False):
    validation = validate_operation_spec(spec)
    if validation.is_left():
        return validation
    caps = spec.capabilities
    if task_type not in caps.tasks:
        return Left(ExtensionError('unsupported_task', 'Task is not supported by the extension.'))
    supported = build_extension_data_type_view(caps.data_types).tensor_types
    try:
        canonical = build_extension_data_type_view((data_type,)).tensor_types[0]
    except (TypeError, ValueError) as exc:
        return Left(ExtensionError('unsupported_data_type', 'Unknown input data type.', cause=exc))
    if canonical not in supported:
        return Left(ExtensionError('unsupported_data_type', 'Data type is not supported by the extension.'))
    if fitting and caps.requires_target and not has_target:
        return Left(ExtensionError('target_required', 'Operation requires a training target.'))
    output_type = caps.output_data_type if caps.output_data_type is not None else data_type
    output_type = build_extension_data_type_view((output_type,)).tensor_types[0]
    kind = OperationKind.model if isinstance(spec, ExternalModelSpec) else OperationKind.transform
    return Right(ExecutionPlan(spec.name, kind, caps.backend, output_type))
