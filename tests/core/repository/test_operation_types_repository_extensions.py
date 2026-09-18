from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.operation_types_repository import OperationTypesRepository, get_operations_for_task
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.extensions.contracts import ExtensionManifest, ExternalModelSpec, ModelCapabilities
from fedot.extensions.registry import clear_extension_registry, register_extension


def _make_manifest():
    return ExtensionManifest(
        name='repository_extension',
        version='1.0.0',
        models=(
            ExternalModelSpec(
                name='external_linear',
                factory=lambda params=None: object(),
                capabilities=ModelCapabilities(
                    tasks=(TaskTypesEnum.regression,),
                    data_types=(DataTypesEnum.table,),
                    tags=('linear',),
                ),
            ),
        ),
    )


def test_operation_repository_includes_registered_extension_models():
    clear_extension_registry()
    register_extension(_make_manifest())

    try:
        task = Task(TaskTypesEnum.regression)
        operations = OperationTypesRepository('model').suitable_operation(
            task_type=task.task_type,
            data_type=DataTypesEnum.table,
        )
        queried_operations = get_operations_for_task(
            task=task,
            data_type=DataTypesEnum.table,
            mode='model',
        )

        assert 'external_linear' in operations
        assert 'external_linear' in queried_operations
    finally:
        clear_extension_registry()
