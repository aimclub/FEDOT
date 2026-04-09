from fedot.industrial.core.models.nn.utils.runtime_metadata_rules import (
    OutputCompatibilityContext,
    RegistryCheckpointContext,
    attach_output_runtime_context,
    build_output_compatibility_context,
    build_output_runtime_attachment_plan,
    build_registry_checkpoint_context,
)


class _OutputData:
    pass


class _NestedFeatures:
    def __init__(self):
        self.features = 'nested_features'
        self.train_dataloader = 'nested_train_loader'
        self.val_dataloader = 'nested_val_loader'
        self.num_classes = 4


class _InputData:
    def __init__(self):
        self.features = _NestedFeatures()


def test_build_registry_checkpoint_context_returns_typed_context():
    context = build_registry_checkpoint_context(
        model_id='model_1',
        checkpoint_path='checkpoints/model_1.pt',
        fedcore_id='fedcore_1',
    )

    assert isinstance(context, RegistryCheckpointContext)
    assert context.model_id == 'model_1'
    assert context.checkpoint_path == 'checkpoints/model_1.pt'
    assert context.fedcore_id == 'fedcore_1'


def test_build_output_compatibility_context_extracts_nested_fields():
    context = build_output_compatibility_context(_InputData())

    assert isinstance(context, OutputCompatibilityContext)
    assert context.features == 'nested_features'
    assert context.train_dataloader == 'nested_train_loader'
    assert context.val_dataloader == 'nested_val_loader'
    assert context.num_classes == 4


def test_build_output_runtime_attachment_plan_skips_none_metadata():
    plan = build_output_runtime_attachment_plan(
        compatibility_context=OutputCompatibilityContext(
            num_classes=2,
            train_dataloader='train_loader',
            val_dataloader=None,
        ),
        checkpoint_context=build_registry_checkpoint_context(
            model_id=None,
            checkpoint_path='checkpoint.pt',
            fedcore_id='fedcore_1',
        ),
        model='model_object',
    )

    assert plan.context_attrs == {
        'num_classes': 2,
        'train_dataloader': 'train_loader',
    }
    assert plan.metadata_attrs == {
        'model': 'model_object',
        'checkpoint_path': 'checkpoint.pt',
        'fedcore_id': 'fedcore_1',
    }


def test_attach_output_runtime_context_applies_context_and_metadata_attrs():
    output = _OutputData()
    plan = build_output_runtime_attachment_plan(
        compatibility_context=OutputCompatibilityContext(num_classes=3),
        checkpoint_context=build_registry_checkpoint_context(
            model_id='model_2',
            checkpoint_path='checkpoint_2.pt',
            fedcore_id='fedcore_2',
        ),
        model='model_object',
    )

    attach_output_runtime_context(output, plan)

    assert output.num_classes == 3
    assert output.model == 'model_object'
    assert output.model_id == 'model_2'
    assert output.checkpoint_path == 'checkpoint_2.pt'
    assert output.fedcore_id == 'fedcore_2'