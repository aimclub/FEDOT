"""Pure rules for resolving registry context in NN trainer shells."""

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

RegistryContextSource = Literal['explicit', 'trainer', 'thread_local', 'none']


@dataclass(frozen=True)
class ResolvedRegistryContext:
    fedcore_id: Optional[str]
    model_id: Optional[str]
    source: RegistryContextSource


def build_resolved_registry_context(explicit_fedcore_id: Optional[str] = None,
                                    trainer_fedcore_id: Optional[str] = None,
                                    thread_local_context: Optional[Tuple[Optional[str], Optional[str]]] = None
                                    ) -> ResolvedRegistryContext:
    if explicit_fedcore_id is not None:
        return ResolvedRegistryContext(
            fedcore_id=explicit_fedcore_id,
            model_id=None,
            source='explicit',
        )

    if trainer_fedcore_id is not None:
        return ResolvedRegistryContext(
            fedcore_id=trainer_fedcore_id,
            model_id=None,
            source='trainer',
        )

    if thread_local_context is not None:
        thread_fedcore_id, thread_model_id = thread_local_context
        if thread_fedcore_id is not None:
            return ResolvedRegistryContext(
                fedcore_id=thread_fedcore_id,
                model_id=thread_model_id,
                source='thread_local',
            )

    return ResolvedRegistryContext(
        fedcore_id=None,
        model_id=None,
        source='none',
    )
