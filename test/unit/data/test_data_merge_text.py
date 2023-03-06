import os

import pandas as pd
import pytest
import numpy as np

from fedot.core.data.data import OutputData
from fedot.core.data.merge.data_merger import DataMerger
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.utils import fedot_project_root


def load_sample_text(file_path=None, label_col='label'):
    base_dir = str(fedot_project_root())
    file_path = file_path or os.path.join(base_dir, 'cases', 'data', 'spam', 'spamham.csv')
    df_text = pd.read_csv(file_path)
    df_text = df_text.sample(frac=1).reset_index(drop=True)

    features = df_text['text'].values.reshape(-1, 1)
    target = np.array(df_text[label_col])
    return features, target


all_features, all_classes = load_sample_text()


def generate_output_texts(length=10, num_columns=1):
    task = Task(TaskTypesEnum.classification)
    data_type = DataTypesEnum.text
    features = all_features[:length]
    if num_columns > 1:
        features = np.hstack([np.expand_dims(features, axis=-1)] * num_columns)
    idx = np.arange(0, length)

    return OutputData(idx,  task, data_type, features=features, predict=features, target=None)


# len(params) is a number of input arrays, each element of params - number of columns in the corresponding array
# FEDOT supports merging of text data with 1 column only, so ValueError is expected for (2, 1) and (2, 3, 1) cases
@pytest.fixture(params=[(1,), (1, 1), (1, 1, 1), (2, 1), (2, 3, 1)],
                ids=lambda cols: f'texts with {cols} columns')
def output_texts(request):
    all_num_columns = request.param
    outputs = [generate_output_texts(num_columns=num_columns) for num_columns in all_num_columns]
    return outputs


def test_data_merge_texts(output_texts):
    first_output = output_texts[0]

    def get_num_columns(data: np.array):
        return data.shape[1] if data.ndim > 1 else 1

    if len(output_texts[0].features.shape) > 2:
        with pytest.raises(ValueError, match="not supported"):
            DataMerger.get(output_texts).merge()
    else:
        merged_data = DataMerger.get(output_texts).merge()

        assert np.equal(merged_data.idx, first_output.idx).all()
        expected_num_columns = sum(get_num_columns(output.predict) for output in output_texts)
        assert merged_data.features.shape[0] == len(first_output.predict)
        assert get_num_columns(merged_data.features) == 1
        assert len(merged_data.features[0][0]) >= len(output_texts[0].features[0][0]) * expected_num_columns
