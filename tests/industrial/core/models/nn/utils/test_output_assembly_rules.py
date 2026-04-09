from fedot.industrial.core.models.nn.utils.output_assembly_rules import (
    OutputContainerRequest,
    assemble_output_container,
    build_output_container_request,
)
from fedot.industrial.core.models.nn.utils.runtime_metadata_rules import (
    OutputCompatibilityContext,
    build_registry_checkpoint_context,
)


class _OutputData:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)


def test_build_output_container_request_preserves_base_kwargs():
    request = build_output_container_request(
        features='features',
        task='task',
        predict='predict',
        data_type='table',
        extra_flag='extra',
    )

    assert isinstance(request, OutputContainerRequest)
    assert request.base_kwargs == {
        'features': 'features',
        'task': 'task',
        'predict': 'predict',
        'data_type': 'table',
        'extra_flag': 'extra',
    }


def test_assemble_output_container_builds_object_and_attaches_runtime_context():
    output = assemble_output_container(
        factory=_OutputData,
        request=build_output_container_request(
            features='features',
            task='task',
            predict='predict',
            data_type='table',
        ),
        compatibility_context=OutputCompatibilityContext(
            num_classes=3,
            train_dataloader='train_loader',
        ),
        checkpoint_context=build_registry_checkpoint_context(
            model_id='model_1',
            checkpoint_path='checkpoint_1.pt',
            fedcore_id='fedcore_1',
        ),
        model='model_object',
    )

    assert output.features == 'features'
    assert output.task == 'task'
    assert output.predict == 'predict'
    assert output.data_type == 'table'
    assert output.num_classes == 3
    assert output.train_dataloader == 'train_loader'
    assert output.model == 'model_object'
    assert output.model_id == 'model_1'
    assert output.checkpoint_path == 'checkpoint_1.pt'
    assert output.fedcore_id == 'fedcore_1'