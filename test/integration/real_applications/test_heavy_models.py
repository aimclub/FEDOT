import pytest

from examples.simple.time_series_forecasting.api_forecasting import get_ts_data
from examples.simple.time_series_forecasting.ts_pipelines import cgru_pipeline
from fedot.core.pipelines.pipeline_builder import PipelineBuilder


@pytest.mark.parametrize('pipeline', (
    PipelineBuilder().add_node('lagged', params={'window_size': 200}).add_node('cgru').build(),
    cgru_pipeline(),
), ids=str)
def test_cgru_forecasting(pipeline):
    horizon = 5
    train_data, test_data, _ = get_ts_data('salaries', horizon)
    pipeline.fit(train_data)
    predicted = pipeline.predict(test_data).predict

    assert predicted is not None
    assert len(predicted) == horizon
