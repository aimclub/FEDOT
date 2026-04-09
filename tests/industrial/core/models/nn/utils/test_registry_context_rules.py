from fedot.industrial.core.models.nn.utils.registry_context_rules import (
    ResolvedRegistryContext,
    build_resolved_registry_context,
)


def test_build_resolved_registry_context_prefers_explicit_fedcore_id():
    context = build_resolved_registry_context(
        explicit_fedcore_id='explicit_fedcore',
        trainer_fedcore_id='trainer_fedcore',
        thread_local_context=('thread_fedcore', 'thread_model'),
    )

    assert isinstance(context, ResolvedRegistryContext)
    assert context.fedcore_id == 'explicit_fedcore'
    assert context.model_id is None
    assert context.source == 'explicit'


def test_build_resolved_registry_context_falls_back_to_trainer_then_thread_local():
    trainer_context = build_resolved_registry_context(
        explicit_fedcore_id=None,
        trainer_fedcore_id='trainer_fedcore',
        thread_local_context=('thread_fedcore', 'thread_model'),
    )
    assert trainer_context.fedcore_id == 'trainer_fedcore'
    assert trainer_context.source == 'trainer'

    thread_context = build_resolved_registry_context(
        explicit_fedcore_id=None,
        trainer_fedcore_id=None,
        thread_local_context=('thread_fedcore', 'thread_model'),
    )
    assert thread_context.fedcore_id == 'thread_fedcore'
    assert thread_context.model_id == 'thread_model'
    assert thread_context.source == 'thread_local'


def test_build_resolved_registry_context_returns_none_source_when_unavailable():
    context = build_resolved_registry_context()
    assert context.fedcore_id is None
    assert context.model_id is None
    assert context.source == 'none'