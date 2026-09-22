from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.constants import BEST_QUALITY_PRESET_NAME
from fedot.core.repository.operation_query import OperationQuery, RepositoryKind
from fedot.core.repository.operation_types_repository import OperationTypesRepository
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
            RepositoryKind.data_operation) is True

        views = filter_extension_operation_views(
            OperationQuery(
                repository_kind=RepositoryKind.model,
                task_type=TaskTypesEnum.classification,
                data_type=DataTypesEnum.table,
                tags=('tree',),
            )
        )
        names = get_extension_operation_names(
            OperationQuery(
                repository_kind=RepositoryKind.model,
                task_type=TaskTypesEnum.classification,
                data_type=DataTypesEnum.table,
                tags=('tree',),
            )
        )

        assert len(views) == 1
        assert views[0].name == 'external_rf'
        assert names == ['external_rf']
    finally:
        clear_extension_registry()


def test_extension_operation_rules_apply_default_exclusions_and_presets():
    manifest = _make_manifest()
    model = manifest.models[0]
    hidden_model = ExternalModelSpec(
        name='external_hidden',
        factory=model.factory,
        capabilities=ModelCapabilities(
            tasks=model.capabilities.tasks,
            data_types=model.capabilities.data_types,
            tags=('non-default',),
        ),
    )
    clear_extension_registry()
    register_extension(ExtensionManifest(
        name=manifest.name,
        version=manifest.version,
        models=(model, hidden_model),
    ))

    try:
        base_query = dict(
            repository_kind=RepositoryKind.model,
            task_type=TaskTypesEnum.classification,
            data_type=DataTypesEnum.table,
            default_excluded_tags=('non-default', 'deprecated'),
        )

        assert get_extension_operation_names(
            OperationQuery(**base_query)) == ['external_rf']
        assert get_extension_operation_names(OperationQuery(
            **base_query,
            tags=('non-default',),
        )) == ['external_hidden']
        assert get_extension_operation_names(OperationQuery(
            **base_query,
            forbidden_tags=('tree',),
        )) == []
        assert get_extension_operation_names(OperationQuery(
            **base_query,
            preset='fast_train',
        )) == []
        assert get_extension_operation_names(OperationQuery(
            **base_query,
            preset=BEST_QUALITY_PRESET_NAME,
        )) == ['external_rf']

        repository = OperationTypesRepository('model')
        assert 'external_hidden' not in repository.suitable_operation(
            TaskTypesEnum.classification,
            DataTypesEnum.table,
        )
        assert 'external_hidden' in repository.suitable_operation(
            TaskTypesEnum.classification,
            DataTypesEnum.table,
            tags=['non-default'],
        )
        assert 'external_rf' not in repository.suitable_operation(
            TaskTypesEnum.classification,
            DataTypesEnum.table,
            preset='fast_train',
        )
    finally:
        clear_extension_registry()
