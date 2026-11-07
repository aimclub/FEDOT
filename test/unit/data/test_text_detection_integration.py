import pandas as pd
import pytest

import fedot.core.data.multi_modal as multi_modal
from fedot.core.data.data_detection import TextDataDetector
from fedot.core.data.multi_modal import MultiModalData


@pytest.fixture
def csv_path(tmp_path):
    descriptions = [
        'This product works very well and is simple to use',
        'The package arrived quickly and was in good condition',
        'I would recommend this device because of its quality',
        'The instructions were clear and installation was easy',
    ]

    dataframe = pd.DataFrame({
        'review': descriptions * 10,
        'sparse': [
            'This is a detailed review of the product quality',
            'The delivery was quick and the item worked well',
            'The product was comfortable and easy to install',
        ] + [None] * 37,
        'quantity': list(range(40)),
        'target': [0, 1] * 20,
    })

    path = tmp_path / 'reviews.csv'
    dataframe.to_csv(path, index=False)

    return path


def test_sparse_text_is_not_prepared(csv_path, monkeypatch):
    prepared_columns = []

    original = TextDataDetector.prepare_multimodal_data

    def record_columns(dataframe, columns):
        prepared_columns.extend(columns)
        return original(dataframe, columns)

    monkeypatch.setattr(
        TextDataDetector,
        'prepare_multimodal_data',
        staticmethod(record_columns),
    )

    result = MultiModalData.from_csv(
        file_path=str(csv_path),
        target_columns='target',
        index_col=None,
    )

    assert 'review' in prepared_columns
    assert 'sparse' not in prepared_columns

    assert 'data_source_text/review' in result
    assert 'data_source_text/sparse' not in result


def test_target_processing_called_once(csv_path, monkeypatch):
    original = multi_modal.process_target_and_features
    calls = []

    def record_call(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(
        multi_modal,
        'process_target_and_features',
        record_call,
    )

    MultiModalData.from_csv(
        file_path=str(csv_path),
        target_columns='target',
        index_col=None,
    )

    assert len(calls) == 1
