from fedot.api.main import Fedot
from test.unit.multimodal.data_generators import get_single_task_multimodal_tabular_data


def test_multimodal_predict_correct():
    """ Test if multimodal data can be processed with pipeline preprocessing correctly """
    mm_data, pipeline = get_single_task_multimodal_tabular_data()

    pipeline.fit(mm_data)
    predicted_labels = pipeline.predict(mm_data, output_mode='labels')
    predicted = pipeline.predict(mm_data)

    # Union of several tables into one feature table
    assert predicted.features.shape == (9, 24)
    assert predicted.predict[0, 0] > 0.5
    assert predicted_labels.predict[0, 0] == 'true'


def test_multimodal_api():
    """ Test if multimodal data can be processed correctly through API """
    mm_data, _ = get_single_task_multimodal_tabular_data()

    automl_model = Fedot(problem='classification', timeout=0.1)
    pipeline = automl_model.fit(features=mm_data,
                                target=mm_data.target,
                                predefined_model='auto')
    prediction = automl_model.predict(mm_data)

    assert pipeline is not None
    assert (9, 1) == prediction.shape
