"""Prepare parent outputs without storing graph context in TensorData."""
from dataclasses import replace

from fedot.core.data.merge.data_merger import TensorDataMerger
from fedot.core.operations.implementations.data_operations.tools import replace_projected_features
from fedot.core.operations.model import Model
from fedot.core.pipelines.tensor_input_rules import (
    DECOMPOSE_OPERATIONS, resolve_decompose_parents, resolve_tensor_target_source,
)
from fedot.core.repository.operation_types_repository import get_operation_type_from_id


def prepare_tensor_input(operation_type, parents, outputs):
    """Resolve the objective from the graph, then align the actual parent data.

    Ordinary operations concatenate semantic parent outputs. Decompose keeps
    the feature parent's own schema and places the baseline in predict, avoiding
    a concatenation followed by reconstruction of that same feature block.
    """
    is_model = [isinstance(parent.operation, Model) for parent in parents]
    merger = TensorDataMerger(outputs, parent_is_models=is_model,
                             main_output_index=resolve_tensor_target_source(
                                 parents, [output.target is not None for output in outputs]))
    if get_operation_type_from_id(operation_type) not in DECOMPOSE_OPERATIONS:
        return merger.merge()
    prediction_index, feature_index = resolve_decompose_parents(parents)
    features = merger.aligned_output(feature_index)
    if is_model[feature_index]:
        features = replace_projected_features(features, features.predict.reshape(len(features.features), -1))
    prediction = merger.aligned_output(prediction_index).predict
    return replace(features, target=merger.merge_target(), task=merger.main_output.task,
                   predict=prediction, fingerprint=None)
