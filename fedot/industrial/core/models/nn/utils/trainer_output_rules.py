"""Effect-light trainer output assembly rules."""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

from fedot.industrial.core.models.nn.utils.checkpoint_registration_rules import (
    build_checkpoint_registration_request,
    execute_checkpoint_registration,
)
from fedot.industrial.core.models.nn.utils.output_assembly_rules import (
    assemble_output_container,
    build_output_container_request,
)
from fedot.industrial.core.models.nn.utils.runtime_metadata_rules import (
    build_output_compatibility_context,
)


@dataclass(frozen=True)
class TrainerOutputAssemblyRequest:
    task: Any
    predict: Any
    data_type: Any
    stage: Optional[str] = None
    explicit_fedcore_id: Optional[str] = None
    trainer_fedcore_id: Optional[str] = None
    extra_output_kwargs: Dict[str, Any] = field(default_factory=dict)


def assemble_registered_trainer_output(output_factory: Callable[..., Any],
                                       input_data: Any,
                                       request: TrainerOutputAssemblyRequest,
                                       model: Any,
                                       register_model: Callable[..., str],
                                       get_checkpoint_path: Callable[[str, str], Optional[str]],
                                       thread_local_context: Optional[Tuple[Optional[str], Optional[str]]] = None
                                       ) -> Any:
    compatibility_context = build_output_compatibility_context(input_data)
    checkpoint_request = build_checkpoint_registration_request(
        model_present=model is not None,
        stage=request.stage,
        explicit_fedcore_id=request.explicit_fedcore_id,
        trainer_fedcore_id=request.trainer_fedcore_id,
        thread_local_context=thread_local_context,
    )
    checkpoint_context = execute_checkpoint_registration(
        request=checkpoint_request,
        model=model,
        register_model=register_model,
        get_checkpoint_path=get_checkpoint_path,
    )

    return assemble_output_container(
        factory=output_factory,
        request=build_output_container_request(
            features=compatibility_context.features,
            task=request.task,
            predict=request.predict,
            data_type=request.data_type,
            **request.extra_output_kwargs,
        ),
        compatibility_context=compatibility_context,
        checkpoint_context=checkpoint_context,
        model=model,
    )
