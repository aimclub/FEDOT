from fedot.industrial.core.models.nn.utils.trainer_output_rules import (
    TrainerOutputAssemblyRequest,
    assemble_registered_trainer_output,
)


class _OutputData:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class _NestedFeatures:
    def __init__(self):
        self.features = 'nested_features'
        self.train_dataloader = 'nested_train_loader'
        self.val_dataloader = 'nested_val_loader'
        self.num_classes = 4


class _InputData:
    def __init__(self):
        self.features = _NestedFeatures()


def test_assemble_registered_trainer_output_runs_full_fake_registry_flow():
    calls = {}

    def _register_model(**kwargs):
        calls['register_model'] = kwargs
        return 'model_1'

    def _get_checkpoint_path(fedcore_id, model_id):
        calls['get_checkpoint_path'] = (fedcore_id, model_id)
        return 'checkpoint_1.pt'

    output = assemble_registered_trainer_output(
        output_factory=_OutputData,
        input_data=_InputData(),
        request=TrainerOutputAssemblyRequest(
            task='task',
            predict='predict',
            data_type='table',
            stage='after',
            explicit_fedcore_id='fedcore_explicit',
        ),
        model='model_object',
        register_model=_register_model,
        get_checkpoint_path=_get_checkpoint_path,
        thread_local_context=('thread_fedcore', 'thread_model'),
    )

    assert calls['register_model'] == {
        'fedcore_id': 'fedcore_explicit',
        'model': 'model_object',
        'stage': 'after',
        'delete_model_after_save': False,
    }
    assert calls['get_checkpoint_path'] == ('fedcore_explicit', 'model_1')
    assert output.features == 'nested_features'
    assert output.num_classes == 4
    assert output.train_dataloader == 'nested_train_loader'
    assert output.val_dataloader == 'nested_val_loader'
    assert output.model == 'model_object'
    assert output.model_id == 'model_1'
    assert output.checkpoint_path == 'checkpoint_1.pt'
    assert output.fedcore_id == 'fedcore_explicit'



def test_assemble_registered_trainer_output_skips_registration_without_model():
    calls = {'registers': 0, 'paths': 0}

    def _register_model(**kwargs):
        calls['registers'] += 1
        return 'ignored_model'

    def _get_checkpoint_path(fedcore_id, model_id):
        calls['paths'] += 1
        return 'ignored.pt'

    output = assemble_registered_trainer_output(
        output_factory=_OutputData,
        input_data=_InputData(),
        request=TrainerOutputAssemblyRequest(
            task='task',
            predict='predict',
            data_type='table',
            stage='after',
            explicit_fedcore_id='fedcore_only',
        ),
        model=None,
        register_model=_register_model,
        get_checkpoint_path=_get_checkpoint_path,
    )

    assert calls == {'registers': 0, 'paths': 0}
    assert output.fedcore_id == 'fedcore_only'
    assert not hasattr(output, 'model_id')
    assert not hasattr(output, 'checkpoint_path')