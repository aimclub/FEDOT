import os

from examples.advanced.multi_modal_pipeline import (prepare_multi_modal_data)
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.utils import fedot_project_root


def generate_multi_modal_pipeline():
    images_size = (128, 128)

    # image
    image_node = PrimaryNode('cnn')
    image_node.custom_params = {'image_shape': (images_size[0], images_size[1], 1),
                                'architecture': 'simplified',
                                'num_classes': 2,
                                'epochs': 1,
                                'batch_size': 128}

    # image
    ds_image = PrimaryNode('data_source_img')
    image_node = SecondaryNode('cnn', nodes_from=[ds_image])
    image_node.custom_params = {'image_shape': (images_size[0], images_size[1], 1),
                                'architecture': 'simplified',
                                'num_classes': 2,
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

    pca_node_regr = SecondaryNode('pca', nodes_from=[scaling_node_regr])
    pca_node_regr.custom_params = {'n_components': 0.2}

    pca_node_class = SecondaryNode('pca', nodes_from=[scaling_node_class])
    pca_node_class.custom_params = {'n_components': 0.2}

    class_node = SecondaryNode('dt', nodes_from=[scaling_node_class])

    root_regr = SecondaryNode('dtreg', nodes_from=[scaling_node_regr, class_node])

    initial_pipeline = Pipeline(root_regr)

    return initial_pipeline


def test_multi_modal_pipeline():
    pipeline = generate_multi_modal_pipeline()

    files_path = os.path.join('test', 'data', 'multi_modal')
    path = os.path.join(str(fedot_project_root()), files_path)
    task = Task(TaskTypesEnum.classification)
    images_size = (128, 128)

    train_num, _, train_img, _, train_text, _ = \
        prepare_multi_modal_data(path, task, images_size, with_split=False)

    fit_data = MultiModalData({
        'data_source_img': train_img,
        'data_source_table': train_num,
        'data_source_text': train_text
    })

    pipeline.fit(fit_data)
    prediction = pipeline.predict(fit_data)

    assert prediction is not None


def test_finding_side_root_node_in_multi_modal_pipeline():
    pipeline = generate_multi_task_pipeline()

    class_pipeline = pipeline.pipeline_for_side_task(task_type=TaskTypesEnum.classification)
    reg_pipeline = pipeline.pipeline_for_side_task(task_type=TaskTypesEnum.regression)

    assert reg_pipeline.nodes[0] is pipeline.nodes[0]
    assert class_pipeline.nodes[0] is pipeline.nodes[3]
