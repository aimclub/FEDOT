from fedot import Fedot
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.tasks import TaskTypesEnum
from test.unit.multimodal.data_generators import get_single_task_multimodal_tabular_data


def test_multimodal_predict_correct():
    """ Test if multimodal data can be processed with pipeline preprocessing correctly """
    mm_data, pipeline = get_single_task_multimodal_tabular_data()

    pipeline.fit(mm_data)
    predicted_labels = pipeline.predict(mm_data, output_mode='labels')
    predicted = pipeline.predict(mm_data)

    # Union of several tables into one feature table
    if pipeline.preprocessor.use_label_encoder:
        assert predicted.features.shape == (9, 4)
    else:
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


def generate_multi_task_pipeline():
    ds_regr = PipelineNode('data_source_table/regr')
    ds_class = PipelineNode('data_source_table/class')

    scaling_node_regr = PipelineNode('scaling', nodes_from=[ds_regr])
    scaling_node_class = PipelineNode('scaling', nodes_from=[ds_class])

    dt_class_node = PipelineNode('dt', nodes_from=[scaling_node_class])

    scaling_node_class_2 = PipelineNode('scaling', nodes_from=[dt_class_node])

    root_regr = PipelineNode('dtreg', nodes_from=[scaling_node_regr, scaling_node_class_2])

    initial_pipeline = Pipeline(root_regr)

    return initial_pipeline


def test_finding_side_root_node_in_multi_modal_pipeline():
    reg_root_node = 'dtreg'
    class_root_node = 'dt'

    pipeline = generate_multi_task_pipeline()

    reg_pipeline = pipeline.pipeline_for_side_task(task_type=TaskTypesEnum.regression)
    class_pipeline = pipeline.pipeline_for_side_task(task_type=TaskTypesEnum.classification)

    assert reg_pipeline.root_node.operation.operation_type == reg_root_node
    assert class_pipeline.root_node.operation.operation_type == class_root_node
