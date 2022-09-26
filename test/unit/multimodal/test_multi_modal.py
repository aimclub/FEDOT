import os

from examples.advanced.multi_modal_pipeline import prepare_multi_modal_data
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.utils import fedot_project_root
from fedot.core.data.multi_modal import MultiModalData


def generate_multi_modal_pipeline(data: MultiModalData):
    # image
    images_size = data['data_source_img'].features.shape[1:4]
    ds_image = PrimaryNode('data_source_img')
    image_node = SecondaryNode('cnn', nodes_from=[ds_image])
    image_node.parameters = {'image_shape': images_size,
                             'architecture': 'simplified',
                             'epochs': 15,
                             'batch_size': 128}

    # table
    ds_table = PrimaryNode('data_source_table')
    scaling_node = SecondaryNode('scaling', nodes_from=[ds_table])
    numeric_node = SecondaryNode('rf', nodes_from=[scaling_node])

    # text
    ds_text = PrimaryNode('data_source_text')
    node_text_clean = SecondaryNode('text_clean', nodes_from=[ds_text])
    text_node = SecondaryNode('tfidf', nodes_from=[node_text_clean])

    pipeline = Pipeline(SecondaryNode('logit', nodes_from=[numeric_node, image_node, text_node]))

    return pipeline


def generate_multi_task_pipeline():
    ds_regr = PrimaryNode('data_source_table/regr')
    ds_class = PrimaryNode('data_source_table/class')

    scaling_node_regr = SecondaryNode('scaling', nodes_from=[ds_regr])
    scaling_node_class = SecondaryNode('scaling', nodes_from=[ds_class])

    dt_class_node = SecondaryNode('dt', nodes_from=[scaling_node_class])

    scaling_node_class_2 = SecondaryNode('scaling', nodes_from=[dt_class_node])

    root_regr = SecondaryNode('dtreg', nodes_from=[scaling_node_regr, scaling_node_class_2])

    initial_pipeline = Pipeline(root_regr)

    return initial_pipeline


def test_multi_modal_pipeline():
    files_path = os.path.join('test', 'data', 'multi_modal')
    path = os.path.join(str(fedot_project_root()), files_path)
    task = Task(TaskTypesEnum.classification)
    images_size = (128, 128)

    fit_data = prepare_multi_modal_data(path, task, images_size)
    pipeline = generate_multi_modal_pipeline(fit_data)

    pipeline.fit(fit_data)
    prediction = pipeline.predict(fit_data)

    assert prediction is not None


def test_finding_side_root_node_in_multi_modal_pipeline():
    reg_root_node = 'dtreg'
    class_root_node = 'dt'

    pipeline = generate_multi_task_pipeline()

    reg_pipeline = pipeline.pipeline_for_side_task(task_type=TaskTypesEnum.regression)
    class_pipeline = pipeline.pipeline_for_side_task(task_type=TaskTypesEnum.classification)

    assert reg_pipeline.root_node.operation.operation_type == reg_root_node
    assert class_pipeline.root_node.operation.operation_type == class_root_node
