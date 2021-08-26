from fedot.core.pipelines.node import PrimaryNode
from fedot.core.pipelines.pipeline import Pipeline
from data.data_manager import text_input_data


def test_clean_text_preprocessing():
    input_data = text_input_data()

    preprocessing_pipeline = Pipeline(PrimaryNode('text_clean'))
    preprocessing_pipeline.fit(input_data)

    predicted_output = preprocessing_pipeline.predict(input_data)
    cleaned_text = predicted_output.predict
    text = input_data.features

    assert len(text) == len(cleaned_text)
