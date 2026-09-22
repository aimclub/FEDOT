"""Content-complete TensorData identities for evaluation-scoped cache entries."""
from dataclasses import dataclass, field, fields
from hashlib import sha256

from fedot.core.caching.normalization import stable_hash
from fedot.core.data.tensor_data import TensorData
from fedot.extensions.registry import registered_extensions_identity


def preparation_identity(data: TensorData) -> str:
    state = data.preparation_state
    if state is None:
        return 'unprepared'
    return stable_hash({
        'schema': state.schema, 'plan_hash': state.plan_hash,
        'plan_state': sha256(state.plan_state).hexdigest(),
        'backend': state.backend_name, 'input_hash': state.input_hash,
        'steps': tuple((step.step, step.indices, sha256(step.handler_state).hexdigest())
                       for step in state.steps),
    })


def tensor_data_identity(data: TensorData) -> str:
    """Hash all rows, labels and semantic metadata; ignore derived trace ids.

    Unlike the fast legacy Hasher, this boundary must not sample feature rows.
    Storage device belongs to the evaluation context, not portable file identity.
    """
    data.validate()
    payload = {field.name: getattr(data, field.name) for field in fields(data)
               if field.name not in {'fingerprint', 'trace_uuid', 'preparation_state'}}
    payload['preparation'] = preparation_identity(data)
    return stable_hash(payload, digest_size=32)


@dataclass(frozen=True)
class TensorDataCacheContext:
    data_id: str
    preparation_id: str
    fold_id: int
    candidate_id: str
    namespace: str = 'fedot-evaluation-v1'
    backend: str = 'cpu'
    extensions_id: str = field(default_factory=registered_extensions_identity)

    def __post_init__(self):
        if isinstance(self.fold_id, bool) or not isinstance(self.fold_id, int) or self.fold_id < 0:
            raise ValueError('fold_id must be a nonnegative integer')
        for name in ('data_id', 'preparation_id', 'candidate_id', 'namespace', 'backend', 'extensions_id'):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f'{name} must be a nonempty string')

    @property
    def key(self) -> str:
        return stable_hash(self, digest_size=32)

    @classmethod
    def from_fold(cls, train: TensorData, test: TensorData, fold_id: int,
                  candidate_id: str, namespace: str = 'fedot-evaluation-v1'):
        return cls(
            data_id=stable_hash((tensor_data_identity(train), tensor_data_identity(test))),
            preparation_id=stable_hash((preparation_identity(train), preparation_identity(test))),
            fold_id=fold_id, candidate_id=candidate_id, namespace=namespace,
            backend=f'{train.device}/{test.device}',
        )

    def operation_key(self, operation_hash: str, state: str) -> str:
        return stable_hash((self.key, operation_hash, state), digest_size=32)
