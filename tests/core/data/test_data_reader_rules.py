import numpy as np
import pytest

from fedot.core.data.reader.data_reader_rules import (
    infer_arff_target_idx,
    resolve_arff_target_idx,
    split_arff_features_and_target,
)


@pytest.mark.unit
def test_infer_arff_target_idx_prefers_last_string_column():
    data_array = np.asarray([
        np.array([1.0, 2.0]),
        np.array([b'a', b'b']),
    ], dtype=object)

    assert infer_arff_target_idx(data_array) == -1


@pytest.mark.unit
def test_infer_arff_target_idx_falls_back_to_first_string_column():
    data_array = np.asarray([
        np.array([b'a', b'b']),
        np.array([1.0, 2.0]),
    ], dtype=object)

    assert infer_arff_target_idx(data_array) == 0


@pytest.mark.unit
def test_resolve_arff_target_idx_accepts_named_target():
    resolution = resolve_arff_target_idx(
        target_idx='target',
        field_names=['feature', 'target'],
        data_array=np.asarray([], dtype=object),
    )

    assert resolution.target_idx == 1


@pytest.mark.unit
def test_resolve_arff_target_idx_rejects_unknown_target_name():
    with pytest.raises(ValueError, match='Unknown ARFF target column'):
        resolve_arff_target_idx(
            target_idx='missing',
            field_names=['feature', 'target'],
            data_array=np.asarray([], dtype=object),
        )


@pytest.mark.unit
def test_split_arff_features_and_target_extracts_selected_column():
    data_array = np.asarray([
        np.array([1.0, 2.0]),
        np.array([3.0, 4.0]),
        np.array([b'a', b'b']),
    ], dtype=object)

    features, target = split_arff_features_and_target(data_array=data_array, target_idx=-1)

    assert features.shape == (2, 2)
    assert target.shape == (2,)
    assert target.tolist() == ['a', 'b']
