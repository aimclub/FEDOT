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
from fedot.industrial.core.models.nn.utils.registry_context_rules import (
    build_resolved_registry_context,
)
from fedot.industrial.core.models.nn.utils.runtime_metadata_rules import (
    build_registry_checkpoint_context,
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


@dataclass(frozen=True)
class TrainerRegistryBindings:
    register_model: Callable[..., str]
    get_checkpoint_path: Callable[[str, str], Optional[str]]
    thread_local_context: Optional[Tuple[Optional[str], Optional[str]]] = None


def build_fallback_checkpoint_context(request: TrainerOutputAssemblyRequest) -> Any:
    resolved_context = build_resolved_registry_context(
        explicit_fedcore_id=request.explicit_fedcore_id,
        trainer_fedcore_id=request.trainer_fedcore_id,
        thread_local_context=None,
    )
    return build_registry_checkpoint_context(fedcore_id=resolved_context.fedcore_id)


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


def assemble_trainer_output_with_registry_fallback(output_factory: Callable[..., Any],
                                                   input_data: Any,
                                                   request: TrainerOutputAssemblyRequest,
                                                   model: Any,
                                                   registry_provider: Callable[[], TrainerRegistryBindings]
                                                   ) -> Any:
    compatibility_context = build_output_compatibility_context(input_data)
    output_request = build_output_container_request(
        features=compatibility_context.features,
        task=request.task,
        predict=request.predict,
        data_type=request.data_type,
        **request.extra_output_kwargs,
    )

    try:
        bindings = registry_provider()
        return assemble_registered_trainer_output(
            output_factory=output_factory,
            input_data=input_data,
            request=request,
            model=model,
            register_model=bindings.register_model,
            get_checkpoint_path=bindings.get_checkpoint_path,
            thread_local_context=bindings.thread_local_context,
        )
    except Exception:
        return assemble_output_container(
            factory=output_factory,
            request=output_request,
            compatibility_context=compatibility_context,
            checkpoint_context=build_fallback_checkpoint_context(request),
            model=model,
        )
