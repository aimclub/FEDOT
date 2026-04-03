from types import SimpleNamespace

from fedot.industrial.core.models.nn.utils.hook_runtime_rules import (
    build_hook_runtime_payload,
    resolve_stage_hooks,
    should_stop_training,
)


def test_build_hook_runtime_payload_contains_required_context():
    trainer_objects = {'optimizer': object(), 'stop': False}
    history = {'train_loss': []}

    payload = build_hook_runtime_payload(
        trainer_objects=trainer_objects,
        history=history,
        learning_rate=0.01,
        val_loader='val',
        criterion='criterion',
        extra={'custom': 'value'},
    )

    assert payload == {
        'trainer_objects': trainer_objects,
        'history': history,
        'learning_rate': 0.01,
        'val_loader': 'val',
        'criterion': 'criterion',
        'custom': 'value',
    }


def test_build_hook_runtime_payload_skips_optional_none_fields():
    payload = build_hook_runtime_payload(
        trainer_objects={'stop': False},
        history={'train_loss': []},
    )

    assert payload == {
        'trainer_objects': {'stop': False},
        'history': {'train_loss': []},
    }


def test_resolve_stage_hooks_reads_start_and_end_properties():
    collection = SimpleNamespace(start=['start_a', 'start_b'], end=['end_a'])

    assert resolve_stage_hooks(collection, 'start') == ['start_a', 'start_b']
    assert resolve_stage_hooks(collection, 'end') == ['end_a']


def test_resolve_stage_hooks_rejects_unknown_stage():
    collection = SimpleNamespace(start=[], end=[])

    try:
        resolve_stage_hooks(collection, 'validation')
    except ValueError as exc:
        assert 'Unsupported hook stage' in str(exc)
    else:
        raise AssertionError('Expected ValueError for unsupported stage')


def test_should_stop_training_reads_stop_flag():
    assert should_stop_training({'stop': True}) is True
    assert should_stop_training({'stop': False}) is False
    assert should_stop_training({}) is False