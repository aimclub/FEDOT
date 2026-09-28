import pytest

from fedot.core.pipelines.schemas import (
    validate_cv_folds,
    validate_pipeline_is_fitted,
    validate_pipeline_node_has_parent_nodes,
    validate_pipeline_node_parent_operation,
    validate_single_root_node,
)
from fedot.validation.errors import FedotValidationError


def test_validate_cv_folds_accepts_valid_value():
    validate_cv_folds(5)


def test_validate_cv_folds_accepts_none():
    validate_cv_folds(None)


def test_validate_cv_folds_rejects_one():
    with pytest.raises(FedotValidationError, match='must be 2 or more'):
        validate_cv_folds(1)


def test_validate_cv_folds_rejects_zero():
    with pytest.raises(FedotValidationError, match='must be 2 or more'):
        validate_cv_folds(0)


def test_validate_pipeline_is_fitted_accepts_true():
    validate_pipeline_is_fitted(True)


def test_validate_pipeline_is_fitted_rejects_false():
    with pytest.raises(FedotValidationError, match='Pipeline is not fitted yet'):
        validate_pipeline_is_fitted(False)


def test_validate_single_root_node_accepts_single_root():
    validate_single_root_node(1)


def test_validate_single_root_node_rejects_multiple_roots():
    with pytest.raises(FedotValidationError, match='More than 1 root_nodes in pipeline'):
        validate_single_root_node(2)


def test_validate_pipeline_node_has_parent_nodes_accepts_non_empty_parents():
    validate_pipeline_node_has_parent_nodes(1)


def test_validate_pipeline_node_has_parent_nodes_rejects_empty_parents():
    with pytest.raises(FedotValidationError, match='No parent nodes found'):
        validate_pipeline_node_has_parent_nodes(0)


def test_validate_pipeline_node_parent_operation_accepts_supported_values():
    assert validate_pipeline_node_parent_operation('fit') == 'fit'
    assert validate_pipeline_node_parent_operation('predict') == 'predict'


def test_validate_pipeline_node_parent_operation_rejects_unknown_value():
    with pytest.raises(FedotValidationError, match='parent_operation'):
        validate_pipeline_node_parent_operation('unknown')
