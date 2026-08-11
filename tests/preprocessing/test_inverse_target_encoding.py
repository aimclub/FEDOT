import numpy as np
import pytest
import torch

from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.data.tensor_data.tensor_data_creator import TensorDataCreator
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.pipeline_rules import OutputModeEnum
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams


def _classification_train_with_string_target():
    return np.array([
        [1.0, 0.1, 'cat'],
        [2.0, 0.2, 'dog'],
        [3.0, 0.3, 'cat'],
        [4.0, 0.4, 'dog'],
    ], dtype=object)


def _result_from_train_td(train_td: TensorData, predict: torch.Tensor) -> TensorData:
    """Build a predict-like TensorData that reuses train obligatory trace."""
    return TensorData(
        task=train_td.task,
        data_type=train_td.data_type,
        features=train_td.features,
        target=train_td.target,
        predict=predict,
        trace_uuid=train_td.trace_uuid,
        idx=train_td.idx,
        categorical_idx=train_td.categorical_idx,
        numerical_idx=train_td.numerical_idx,
    )


@pytest.mark.unit
def test_postprocess_decoded_restores_string_labels_from_cached_encoder(isolated_cache_dir):
    train_td = TensorDataCreator.create(
        _classification_train_with_string_target(),
        backend_name='cpu',
    )
    assert train_td.trace_uuid is not None

    encoded_predict = train_td.target.detach().clone().reshape(-1)
    result = _result_from_train_td(train_td, encoded_predict)

    postprocessed = Pipeline()._postprocess(result, output_mode=OutputModeEnum.DECODED)

    assert list(postprocessed.predict) == ['cat', 'dog', 'cat', 'dog']


@pytest.mark.unit
def test_postprocess_auto_classification_decodes_like_decoded(isolated_cache_dir):
    train_td = TensorDataCreator.create(
        _classification_train_with_string_target(),
        backend_name='cpu',
    )
    encoded_predict = train_td.target.detach().clone().reshape(-1)
    result = _result_from_train_td(train_td, encoded_predict)

    postprocessed = Pipeline()._postprocess(result, output_mode=OutputModeEnum.AUTO)

    assert list(postprocessed.predict) == ['cat', 'dog', 'cat', 'dog']


@pytest.mark.unit
def test_postprocess_raw_keeps_encoded_numeric_labels(isolated_cache_dir):
    train_td = TensorDataCreator.create(
        _classification_train_with_string_target(),
        backend_name='cpu',
    )
    encoded_predict = train_td.target.detach().clone().reshape(-1)
    result = _result_from_train_td(train_td, encoded_predict)

    postprocessed = Pipeline()._postprocess(result, output_mode=OutputModeEnum.RAW)

    assert torch.allclose(postprocessed.predict, encoded_predict)


@pytest.mark.unit
def test_postprocess_decoded_noop_when_target_was_numeric(isolated_cache_dir):
    train = np.array([
        [1.0, 0.1, 0],
        [2.0, 0.2, 1],
        [3.0, 0.3, 0],
    ], dtype=object)
    train_td = TensorDataCreator.create(train, backend_name='cpu')

    encoded_predict = torch.tensor([0.0, 1.0, 0.0])
    result = _result_from_train_td(train_td, encoded_predict)

    postprocessed = Pipeline()._postprocess(result, output_mode=OutputModeEnum.DECODED)

    assert torch.allclose(postprocessed.predict, encoded_predict)


@pytest.mark.unit
def test_postprocess_flattened_ravels_prediction(isolated_cache_dir):
    train = np.array([
        [1.0, 0.1, 0.0],
        [2.0, 0.2, 1.0],
    ], dtype=np.float32)
    train_td = TensorDataCreator.create(
        train,
        backend_name='cpu',
        task=Task(TaskTypesEnum.regression),
    )

    result = _result_from_train_td(
        train_td,
        predict=torch.tensor([[0.0], [1.0]]),
    )

    postprocessed = Pipeline()._postprocess(result, output_mode=OutputModeEnum.FLATTENED)

    assert postprocessed.predict.shape == (2,)
    assert torch.allclose(postprocessed.predict, torch.tensor([0.0, 1.0]))


@pytest.mark.unit
def test_postprocess_auto_ts_flattens_without_decode(isolated_cache_dir):
    series = np.arange(12, dtype=np.float32).reshape(-1, 1)
    train_td = TensorDataCreator.create(
        series,
        backend_name='cpu',
        task=Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(forecast_length=2)),
        data_type=DataTypesEnum.ts,
    )

    result = TensorData(
        task=train_td.task,
        data_type=train_td.data_type,
        features=train_td.features,
        target=train_td.target,
        predict=torch.tensor([[1.5], [2.5]]),
        trace_uuid=train_td.trace_uuid,
    )

    postprocessed = Pipeline()._postprocess(result, output_mode=OutputModeEnum.AUTO)

    assert postprocessed.predict.shape == (2,)
    assert torch.allclose(postprocessed.predict, torch.tensor([1.5, 2.5]))


@pytest.mark.unit
def test_postprocess_auto_classification_with_predict_state_trace(isolated_cache_dir):
    """Train + predict TD creation keeps the same trace; postprocess still decodes."""
    train = _classification_train_with_string_target()
    train_td = TensorDataCreator.create(train, backend_name='cpu')

    test = np.array([
        [5.0, 0.5],
        [6.0, 0.6],
    ], dtype=object)
    test_td = TensorDataCreator.create(
        test,
        backend_name='cpu',
        state='predict',
        without_target=True,
        trace_uuid=train_td.trace_uuid,
    )
    assert test_td.trace_uuid == train_td.trace_uuid

    # Simulate model outputs for two rows using train-encoded label ids.
    # train targets after encode: cat/dog/cat/dog -> typically 0/1/0/1
    encoded_train = train_td.target.detach().cpu().numpy().reshape(-1)
    cat_id = float(encoded_train[0])
    dog_id = float(encoded_train[1])
    test_td.predict = torch.tensor([cat_id, dog_id], dtype=torch.float32)

    postprocessed = Pipeline()._postprocess(test_td, output_mode=OutputModeEnum.AUTO)

    assert list(postprocessed.predict) == ['cat', 'dog']
