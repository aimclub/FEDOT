import pytest

from examples.simple.time_series_forecasting.api_forecasting import get_ts_data
from examples.simple.time_series_forecasting.ts_pipelines import cgru_pipeline
from fedot.core.pipelines.pipeline_builder import PipelineBuilder


@pytest.mark.skip(reason="Fails due to the https://github.com/aimclub/FEDOT/issues/1240")
def test_cgru_forecasting():
    horizon = 5
    window_size = 200
    train_data, test_data = get_ts_data('salaries', horizon)

    pipeline = PipelineBuilder().add_node('lagged', params={'window_size': window_size}).add_node('cgru').build()
    pipeline.fit(train_data)
    predicted = pipeline.predict(test_data).predict[0]

    assert len(predicted) == horizon


@pytest.mark.skip(reason="Fails due to the https://github.com/aimclub/FEDOT/issues/1240")
def test_cgru_in_pipeline():
    horizon = 5
    train_data, test_data = train_data, test_data = get_ts_data('salaries', horizon)

    pipeline = cgru_pipeline()
    pipeline.fit(train_data)
    predicted = pipeline.predict(test_data).predict[0]

    assert len(predicted) == horizon
