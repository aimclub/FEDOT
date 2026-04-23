from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.pipeline_operation_repository import PipelineOperationRepository
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.extensions.contracts import ExtensionManifest, ExternalModelSpec, ModelCapabilities
from fedot.extensions.registry import clear_extension_registry, register_extension


def _make_manifest():
    return ExtensionManifest(
        name='pipeline_repository_extension',
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


def test_from_available_operations_returns_self_and_keeps_registered_extension():
    clear_extension_registry()
    register_extension(_make_manifest())

    try:
        repository = PipelineOperationRepository()
        returned_repository = repository.from_available_operations(
            task=Task(TaskTypesEnum.regression),
            preset='best_quality',
            available_operations=['external_linear', 'ridge'],
        )

        assert returned_repository is repository
        assert 'external_linear' in repository.get_all_operations()
    finally:
        clear_extension_registry()
