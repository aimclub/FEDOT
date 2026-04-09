from fedot.industrial.core.models.nn.utils.checkpoint_registration_rules import (
    build_checkpoint_registration_request,
    execute_checkpoint_registration,
)
from fedot.industrial.core.models.nn.utils.output_assembly_rules import (
    assemble_output_container,
    build_output_container_request,
)
from fedot.industrial.core.models.nn.utils.registry_context_rules import (
    build_resolved_registry_context,
)
from fedot.industrial.core.models.nn.utils.runtime_metadata_rules import (
    OutputCompatibilityContext,
)


class _OutputData:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def test_nn_runtime_planning_flow_preserves_explicit_context_priority():
    registry_context = build_resolved_registry_context(
        explicit_fedcore_id='explicit_fedcore',
        trainer_fedcore_id='trainer_fedcore',
        thread_local_context=('thread_fedcore', 'thread_model'),
    )
    request = build_checkpoint_registration_request(
        model_present=True,
        stage='after',
        explicit_fedcore_id=registry_context.fedcore_id,
    )
    checkpoint_context = execute_checkpoint_registration(
        request=request,
        model='model_object',
        register_model=lambda **kwargs: 'model_1',
        get_checkpoint_path=lambda fedcore_id, model_id: 'checkpoint_1.pt',
    )
    output = assemble_output_container(
        factory=_OutputData,
        request=build_output_container_request(
            features='features',
            task='task',
            predict='predict',
            data_type='table',
        ),
        compatibility_context=OutputCompatibilityContext(
            features='features',
            num_classes=2,
            train_dataloader='train_loader',
        ),
        checkpoint_context=checkpoint_context,
        model='model_object',
    )

    assert registry_context.source == 'explicit'
    assert output.num_classes == 2
    assert output.train_dataloader == 'train_loader'
    assert output.model == 'model_object'
    assert output.model_id == 'model_1'
    assert output.fedcore_id == 'explicit_fedcore'


def test_nn_runtime_planning_flow_skips_missing_optional_fields():
    output = assemble_output_container(
        factory=_OutputData,
        request=build_output_container_request(
            features='features_only',
            task='task',
            predict='predict',
            data_type='table',
        ),
        compatibility_context=OutputCompatibilityContext(features='features_only'),
        checkpoint_context=execute_checkpoint_registration(
            request=build_checkpoint_registration_request(
                model_present=False,
                explicit_fedcore_id='fedcore_only',
            ),
            model=None,
            register_model=lambda **kwargs: 'ignored_model',
            get_checkpoint_path=lambda fedcore_id, model_id: 'ignored.pt',
        ),
        model=None,
    )

    assert not hasattr(output, 'num_classes')
    assert not hasattr(output, 'model')
    assert output.fedcore_id == 'fedcore_only'