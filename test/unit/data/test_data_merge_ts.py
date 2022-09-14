from itertools import product

import numpy as np
import pytest

from fedot.core.data.data import OutputData
from fedot.core.data.merge.data_merger import DataMerger
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.utilities.synth_dataset_generator import generate_synthetic_data

from test.unit.data_operations.test_data_operations_implementations import get_time_series

np.random.seed(2021)

testing_pipeline_builders = {
    'lagged-ridge':
        PipelineBuilder()
        .add_sequence('lagged', 'ridge'),
    'parallel lagged':
        PipelineBuilder()
        .add_node('smoothing')
        .add_branch(('lagged', {'window_size': 3}), ('lagged', {'window_size': 5}))
        .join_branches('ridge')
}


@pytest.fixture(params=list(testing_pipeline_builders.keys()))
def ts_pipelines(request):
    return testing_pipeline_builders[request.param]


def get_output_timeseries(len_forecast=5, length=100, num_variables=1, for_predict=False):
    """ Function returns time series for time series forecasting task """
    synthetic_ts = generate_synthetic_data(length=length)

    features = synthetic_ts
    if num_variables > 1:
        # Multivariate timeseries
        features = np.hstack([features.reshape(-1, 1)] * num_variables)
    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=len_forecast))

    if for_predict:
        start_forecast = len(synthetic_ts)
        end_forecast = start_forecast + len_forecast
        idx = np.arange(start_forecast, end_forecast)
        predict = features[-len_forecast:]
    else:
        idx = np.arange(0, length)
        predict = features

    predict_output = OutputData(idx, task, DataTypesEnum.ts, features=features, predict=predict)
    return predict_output


def drop_elements(output: OutputData, fraction_dropped=0.2, with_repetitions=False) -> OutputData:
    num_left = int(len(output.idx) * (1 - fraction_dropped))
    idx_short = np.sort(np.random.choice(output.idx, size=num_left, replace=with_repetitions))
    output_short = OutputData(idx=idx_short,
                              features=output.features[idx_short],
                              predict=output.predict[idx_short],
                              target=output.target[idx_short] if output.target else None,
                              task=output.task, data_type=output.data_type)
    return output_short


def get_output_ts_different_idx(length=100, fraction_dropped=0.2, with_repetitions=False, equal=False):
    output = get_output_timeseries(length=length, for_predict=False)
    output_long = drop_elements(output, fraction_dropped, with_repetitions)
    if not equal:
        fraction_dropped = fraction_dropped / 2
    output_short = drop_elements(output, fraction_dropped, with_repetitions)
    return output_short, output_long


@pytest.fixture(params=list(product(range(0, 100, 20), (False,), (True, False))),
                ids=lambda param: f'dropped %: {param[0]}, with repeats: {param[1]}, equal length: {param[2]}')
def output_ts_different_idx(request):
    percent_dropped, with_repetitions, equal = request.param
    fraction_dropped = percent_dropped / 100
    length = 100
    return get_output_ts_different_idx(length, fraction_dropped, with_repetitions, equal)


def test_data_merge_ts_pipelines(ts_pipelines):
    train_input, predict_input, test_data = get_time_series()

    pipeline = ts_pipelines.to_pipeline()

    pipeline.fit_from_scratch(train_input)
    predicted_output = pipeline.predict(predict_input)

    assert predicted_output is not None


def test_data_merge_ts_multivariate():
    ts1 = get_output_timeseries(for_predict=False)
    ts2 = get_output_timeseries(for_predict=False)
    merged_ts = DataMerger.get([ts1, ts2]).merge()

    assert merged_ts.data_type == DataTypesEnum.ts
    expected_shape = (len(ts1.idx), 2)
    assert merged_ts.features.shape == expected_shape

    ts1 = get_output_timeseries(num_variables=2, for_predict=False)
    ts2 = get_output_timeseries(num_variables=3, for_predict=False)
    merged_ts = DataMerger.get([ts1, ts2]).merge()

    assert merged_ts.data_type == DataTypesEnum.ts
    expected_shape = (len(ts1.idx), 5)
    assert merged_ts.features.shape == expected_shape


def test_data_merge_ts_different_forecast_lengths():
    output_short = get_output_timeseries(len_forecast=5, for_predict=True)
    output_long = get_output_timeseries(len_forecast=12, for_predict=True)
    outputs = [output_short, output_long]

    merged_data = DataMerger.get(outputs).merge()

    assert np.equal(merged_data.idx, output_short.idx).all()
    assert merged_data.features.shape == (output_short.predict.shape[0], 2)


def test_data_merge_ts_different_fit_lengths(output_ts_different_idx):
    output_short, output_long = output_ts_different_idx
    shared_idx = np.intersect1d(output_short.idx, output_long.idx)

    merged_data = DataMerger.get([output_short, output_long]).merge()

    # test index validity
    assert np.isin(merged_data.idx, output_short.idx).all()
    assert np.isin(merged_data.idx, output_long.idx).all()
    assert np.isin(shared_idx, merged_data.idx).all()
    assert len(merged_data.idx) == len(merged_data.features)

    # test features validity
    assert merged_data.features.shape[1] == 2
    assert np.isin(merged_data.features[:, 0], output_short.predict).all()
    assert np.isin(merged_data.features[:, 1], output_long.predict).all()
