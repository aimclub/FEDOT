import os

from examples.multi_modal_chain import (prepare_multi_modal_data)
from fedot.core.chains.chain import Chain
from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.utils import fedot_project_root


def test_multi_modal_chain():
    task = Task(TaskTypesEnum.classification)
    images_size = (128, 128)

    files_path = os.path.join('test', 'data', 'multi_modal')
    path = os.path.join(str(fedot_project_root()), files_path)

    train_num, _, train_img, _, train_text, _ = \
        prepare_multi_modal_data(path, task, images_size, with_split=False)

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

    chain = Chain(SecondaryNode('logit', nodes_from=[numeric_node, image_node, text_node]))

    fit_data = MultiModalData({
        'data_source_img': train_img,
        'data_source_table': train_num,
        'data_source_text': train_text
    })

    chain.fit(fit_data)
    prediction = chain.predict(fit_data)

    assert prediction is not None
