from fedot.industrial.core.models.nn.utils.registry_context_rules import (
    build_resolved_registry_context,
)
from fedot.industrial.core.models.nn.utils.runtime_metadata_rules import (
    OutputCompatibilityContext,
    attach_output_runtime_context,
    build_output_runtime_attachment_plan,
    build_registry_checkpoint_context,
)


class _OutputData:
    pass


def test_nn_runtime_planning_flow_preserves_explicit_context_priority():
    registry_context = build_resolved_registry_context(
        explicit_fedcore_id='explicit_fedcore',
        trainer_fedcore_id='trainer_fedcore',
        thread_local_context=('thread_fedcore', 'thread_model'),
    )
    checkpoint_context = build_registry_checkpoint_context(
        model_id='model_1',
        checkpoint_path='checkpoint_1.pt',
        fedcore_id=registry_context.fedcore_id,
    )
    plan = build_output_runtime_attachment_plan(
        compatibility_context=OutputCompatibilityContext(
            features='features',
            num_classes=2,
            train_dataloader='train_loader',
        ),
        checkpoint_context=checkpoint_context,
        model='model_object',
    )
    output = attach_output_runtime_context(_OutputData(), plan)

    assert registry_context.source == 'explicit'
    assert output.num_classes == 2
    assert output.train_dataloader == 'train_loader'
    assert output.model == 'model_object'
    assert output.model_id == 'model_1'
    assert output.fedcore_id == 'explicit_fedcore'


def test_nn_runtime_planning_flow_skips_missing_optional_fields():
    plan = build_output_runtime_attachment_plan(
        compatibility_context=OutputCompatibilityContext(features='features_only'),
        checkpoint_context=build_registry_checkpoint_context(fedcore_id='fedcore_only'),
        model=None,
    )
    output = attach_output_runtime_context(_OutputData(), plan)

    assert not hasattr(output, 'num_classes')
    assert not hasattr(output, 'model')
    assert output.fedcore_id == 'fedcore_only'