from fedot.core.operations.data_operation import DataOperation
from fedot.core.operations.evaluation.extensions import ExtensionTransformStrategy
from fedot.core.operations.extension_model import ExtensionOperationMixin


class ExtensionTransform(ExtensionOperationMixin, DataOperation):
    """Manifest-registered transform; node output carries features, not predictions."""

    strategy_type = ExtensionTransformStrategy
