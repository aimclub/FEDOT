from fedot.industrial.tools.registry.registry_persistence_rules import (
    RegistryPersistenceRequest,
    build_registry_checkpoint_target_plan,
    build_registry_persistence_request,
    execute_registry_persistence,
)


def test_build_registry_checkpoint_target_plan_reuses_existing_model_path():
    plan = build_registry_checkpoint_target_plan(
        model_path='models/model_1.pt',
        generated_checkpoint_path='checkpoints/model_1_generated.pt',
        model_path_exists=True,
    )

    assert plan.checkpoint_path == 'models/model_1.pt'
    assert plan.should_save_file is False



def test_build_registry_checkpoint_target_plan_uses_generated_checkpoint_when_needed():
    plan = build_registry_checkpoint_target_plan(
        model_path='models/model_1.pt',
        generated_checkpoint_path='checkpoints/model_1_generated.pt',
        model_path_exists=False,
    )

    assert plan.checkpoint_path == 'checkpoints/model_1_generated.pt'
    assert plan.should_save_file is True



def test_execute_registry_persistence_skips_serialize_and_save_for_existing_model_path():
    calls = {'serialize': 0, 'save': 0, 'append': 0}

    request = build_registry_persistence_request(
        fedcore_id='fedcore_1',
        checkpoint_path='models/model_1.pt',
        cleanup_after_save=True,
        should_save_file=False,
        record={'record': 'record_1'},
    )

    execute_registry_persistence(
        request=request,
        serialize_checkpoint=lambda: calls.__setitem__('serialize', calls['serialize'] + 1),
        save_checkpoint=lambda *args, **kwargs: calls.__setitem__('save', calls['save'] + 1),
        append_record=lambda fedcore_id, record: calls.__setitem__('append', calls['append'] + 1),
    )

    assert calls == {'serialize': 0, 'save': 0, 'append': 1}



def test_execute_registry_persistence_uses_fake_collaborators_for_generated_checkpoint():
    calls = {}

    request = RegistryPersistenceRequest(
        fedcore_id='fedcore_1',
        checkpoint_path='checkpoints/model_1.pt',
        cleanup_after_save=True,
        should_save_file=True,
        record={'record': 'record_1'},
    )

    def _serialize_checkpoint():
        calls['serialize'] = True
        return b'checkpoint-bytes'

    def _save_checkpoint(checkpoint_bytes, checkpoint_path, cleanup_after_save=None):
        calls['save'] = (checkpoint_bytes, checkpoint_path, cleanup_after_save)

    def _append_record(fedcore_id, record):
        calls['append'] = (fedcore_id, record)

    execute_registry_persistence(
        request=request,
        serialize_checkpoint=_serialize_checkpoint,
        save_checkpoint=_save_checkpoint,
        append_record=_append_record,
    )

    assert calls['serialize'] is True
    assert calls['save'] == (b'checkpoint-bytes', 'checkpoints/model_1.pt', True)
    assert calls['append'] == ('fedcore_1', {'record': 'record_1'})