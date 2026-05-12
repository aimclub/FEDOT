from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.operation_query import RepositoryKind
from fedot.core.repository.tasks import TaskTypesEnum
from fedot.extensions.contracts import ExtensionManifest, ExternalModelSpec, ModelCapabilities
from fedot.extensions.operation_rules import (
    filter_extension_operation_views,
    get_extension_operation_names,
    should_include_extensions,
)
from fedot.extensions.registry import clear_extension_registry, register_extension


def _make_manifest():
    return ExtensionManifest(
        name='demo_extension',
        version='1.0.0',
        models=(
            ExternalModelSpec(
                name='external_rf',
                factory=lambda params=None: object(),
                capabilities=ModelCapabilities(
                    tasks=(TaskTypesEnum.classification,),
                    data_types=(DataTypesEnum.table,),
                    tags=('tree', 'external'),
                ),
            ),
        ),
    )


def test_extension_operation_rules_filter_registered_models():
    clear_extension_registry()
    register_extension(_make_manifest())

    try:
        assert should_include_extensions(RepositoryKind.model) is True
        assert should_include_extensions(
            RepositoryKind.data_operation) is False

        views = filter_extension_operation_views(
            task_type=TaskTypesEnum.classification,
            data_type=DataTypesEnum.table,
            tags=('tree',),
        )
        names = get_extension_operation_names(
            task_type=TaskTypesEnum.classification,
            data_type=DataTypesEnum.table,
            tags=('tree',),
        )

        assert len(views) == 1
        assert views[0].name == 'external_rf'
        assert names == ['external_rf']
    finally:
        clear_extension_registry()
