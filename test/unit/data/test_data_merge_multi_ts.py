import numpy as np
import pytest

from fedot.core.data.data import OutputData
from fedot.core.data.merge.data_merger import DataMerger
from fedot.core.data.supplementary_data import SupplementaryData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum


@pytest.fixture()
def equal_outputs_multi_ts():
    idx_form = list(range(0, 30, 2)) * 5
    idx_1 = np.array(idx_form)
    idx_2 = np.array(idx_form)

    task = Task(TaskTypesEnum.ts_forecasting)
    generated_target = np.random.sample((len(idx_1), 1))
    generated_features = np.random.sample((len(idx_1), 2))

    list_with_outputs = []
    for idx, data_flow_len in zip([idx_1, idx_2], [1, 0]):
        data_info = SupplementaryData(data_flow_length=data_flow_len)
        output_data = OutputData(idx=idx,
                                 features=generated_features[idx, :],
                                 predict=generated_target[idx, :],
                                 task=task,
                                 target=generated_target[idx, :],
                                 data_type=DataTypesEnum.multi_ts,
                                 supplementary_data=data_info)
        list_with_outputs.append(output_data)

    return list_with_outputs


@pytest.fixture()
def unequal_outputs_multi_ts():
    nrepeats = 5
    idx_1 = np.array(list(range(0, 30, 2)) * nrepeats)
    idx_2 = np.array(list(range(0, 30, 3)) * nrepeats)
    common_idx = np.array(list(range(0, 30, 6)) * nrepeats)

    task = Task(TaskTypesEnum.ts_forecasting)
    generated_target = np.random.sample((len(idx_1), 1))
    generated_features = np.random.sample((len(idx_1), 2))

    list_with_outputs = []
    for idx, data_flow_len in zip([idx_1, idx_2, common_idx], [1, 0, 0]):
        data_info = SupplementaryData(data_flow_length=data_flow_len)
        output_data = OutputData(idx=idx,
                                 features=generated_features[idx, :],
                                 predict=generated_target[idx, :],
                                 task=task,
                                 target=generated_target[idx, :],
                                 data_type=DataTypesEnum.multi_ts,
                                 supplementary_data=data_info)
        list_with_outputs.append(output_data)

    return list_with_outputs


def test_data_merge_multi_ts_equal(equal_outputs_multi_ts):
    output1, output2 = equal_outputs_multi_ts

    merged_data = DataMerger.get([output1, output2]).merge()

    assert np.equal(merged_data.idx, output1.idx).all()
    true_shape = (len(output1.idx), output1.predict.shape[1] + output2.predict.shape[1])
    assert merged_data.features.shape == true_shape


def test_data_merge_multi_ts_unequal(unequal_outputs_multi_ts):
    output1, output2, output_true_idx = unequal_outputs_multi_ts

    merged_data = DataMerger.get([output1, output2]).merge()

    assert np.equal(merged_data.idx, output_true_idx.idx).all()
    true_shape = (len(output_true_idx.predict), output1.predict.shape[1] + output2.predict.shape[1])
    assert merged_data.features.shape == true_shape
