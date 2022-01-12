from test.unit.multimodal.data_generators import get_single_task_multimodal_tabular_data


def test_multimodal_predict_correct():
    """ Test if multimodal data can be processed with pipeline preprocessing correctly """
    mm_data, pipeline = get_single_task_multimodal_tabular_data()

    pipeline.fit(mm_data)
