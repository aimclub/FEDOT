from dataclasses import dataclass
from enum import Enum
from typing import Optional


class TopologicalBackend(Enum):
    GPH = 'gph'
    RIPSER = 'ripser'


@dataclass(frozen=True)
class TopologicalBackendResolution:
    backend: Optional[TopologicalBackend]

    @property
    def is_available(self) -> bool:
        return self.backend is not None


def resolve_topological_backend(gph_available: bool,
                                ripser_available: bool) -> TopologicalBackendResolution:
    backend: Optional[TopologicalBackend]
    if gph_available:
        backend = TopologicalBackend.GPH
    elif ripser_available:
        backend = TopologicalBackend.RIPSER
    else:
        backend = None
    return TopologicalBackendResolution(backend=backend)
