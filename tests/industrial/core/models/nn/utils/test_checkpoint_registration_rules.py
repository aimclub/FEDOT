from fedot.industrial.core.models.nn.utils.checkpoint_registration_rules import (
    CheckpointRegistrationRequest,
    build_checkpoint_registration_request,
    execute_checkpoint_registration,
)


def test_build_checkpoint_registration_request_requires_model_and_resolved_context():
    request = build_checkpoint_registration_request(
        model_present=True,
        stage='after',
        explicit_fedcore_id='fedcore_explicit',
        trainer_fedcore_id='fedcore_trainer',
        thread_local_context=('fedcore_thread', 'model_thread'),
    )

    assert isinstance(request, CheckpointRegistrationRequest)
    assert request.fedcore_id == 'fedcore_explicit'
    assert request.stage == 'after'
    assert request.should_register is True

    missing_model_request = build_checkpoint_registration_request(
        model_present=False,
        explicit_fedcore_id='fedcore_explicit',
    )
    assert missing_model_request.should_register is False

    missing_context_request = build_checkpoint_registration_request(
        model_present=True,
        explicit_fedcore_id=None,
        trainer_fedcore_id=None,
        thread_local_context=None,
    )
    assert missing_context_request.should_register is False


def test_execute_checkpoint_registration_uses_fake_registry_collaborators():
    calls = {}

    def _register_model(**kwargs):
        calls['register_model'] = kwargs
        return 'model_1'

    def _get_checkpoint_path(fedcore_id, model_id):
        calls['get_checkpoint_path'] = (fedcore_id, model_id)
        return 'checkpoints/model_1.pt'

    request = build_checkpoint_registration_request(
        model_present=True,
        stage='after',
        explicit_fedcore_id='fedcore_explicit',
    )
    context = execute_checkpoint_registration(
        request=request,
        model='model_object',
        register_model=_register_model,
        get_checkpoint_path=_get_checkpoint_path,
    )

    assert calls['register_model'] == {
        'fedcore_id': 'fedcore_explicit',
        'model': 'model_object',
        'stage': 'after',
        'delete_model_after_save': False,
    }
    assert calls['get_checkpoint_path'] == ('fedcore_explicit', 'model_1')
    assert context.model_id == 'model_1'
    assert context.checkpoint_path == 'checkpoints/model_1.pt'
    assert context.fedcore_id == 'fedcore_explicit'


def test_execute_checkpoint_registration_is_noop_when_request_disables_registration():
    calls = {'registers': 0, 'paths': 0}

    def _register_model(**kwargs):
        calls['registers'] += 1
        return 'model_ignored'

    def _get_checkpoint_path(fedcore_id, model_id):
        calls['paths'] += 1
        return 'ignored.pt'

    context = execute_checkpoint_registration(
        request=CheckpointRegistrationRequest(
            fedcore_id='fedcore_only',
            stage='after',
            should_register=False,
        ),
        model='model_object',
        register_model=_register_model,
        get_checkpoint_path=_get_checkpoint_path,
    )

    assert calls == {'registers': 0, 'paths': 0}
    assert context.model_id is None
    assert context.checkpoint_path is None
    assert context.fedcore_id == 'fedcore_only'