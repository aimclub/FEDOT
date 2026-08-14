from abc import ABC, abstractmethod
from typing import Optional

from golem.core.log import default_log

from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.utilities.custom_errors import AbstractMethodNotImplementError


class TensorDataOperationImplementation(ABC):
    """Contract for TensorData transform operations used as pipeline nodes.

    Mirror of :class:`DataOperationImplementation` for the ``TensorData`` path:
    one searchable operation / node with ``fit`` + ``transform``.

    Fit-graph vs predict-graph behaviour is owned by
    :class:`~fedot.core.operations.evaluation.evaluation_interfaces.EvaluationStrategy.predict_for_fit`
    (see :class:`~fedot.core.operations.evaluation.tensor_transform.TensorTransformStrategy`),
    not by a second method on this contract. Override strategy-level
    ``predict_for_fit`` only when fit-stage transform must differ (leakage-prone
    ops); until then it delegates to ``transform`` via ``predict``.
    """

    def __init__(self, params: Optional[OperationParameters] = None):
        self.params = params or OperationParameters()
        self.log = default_log(self)

    @abstractmethod
    def fit(self, data: TensorData):
        """Fit the operation on ``data``."""
        raise AbstractMethodNotImplementError()

    @abstractmethod
    def transform(self, data: TensorData) -> TensorData:
        """Apply the fitted operation."""
        raise AbstractMethodNotImplementError()

    def get_params(self) -> OperationParameters:
        """Detached params snapshot for node updates (does not share mutable state).

        Uses :meth:`OperationParameters.copy` (shallow value copy + changed-keys).
        Hyperparameters are scalars/strings; nested mutables are not expected.
        """
        return self.params.copy()
