import os
from typing import Union

from fedot.api.main import Fedot

from sklearn.metrics import f1_score as f1

from cases.dataset_preparation import unpack_archived_data
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.data.data import InputData, OutputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.utils import fedot_project_root


def calculate_validation_metric(valid: Union[InputData, MultiModalData], pred: OutputData) -> float:
    """
    Calculates F1 score for predicted data

    :param valid: dataclass with true target
    :param pred: dataclass with model's prediction
    """

    real = valid.target
    predicted = pred.predict

    err = f1(y_true=real,
             y_pred=predicted, average='micro')

    return round(err, 2)


def prepare_multi_modal_data(files_path: str, task: Task, images_size: tuple = (128, 128)) -> MultiModalData:
    """
    Imports data from 3 different sources (table, images and text)

    :param files_path: path to data
    :param task: task to solve
    :param images_size: the requested size in pixels, as a 2-tuple of (width, height)
    :return: MultiModalData object which contains table, text and image data
    """

    path = os.path.join(str(fedot_project_root()), files_path)
    # unpacking of data archive
    unpack_archived_data(path)
    # import of table data
    data_num = InputData.from_json_files(path, fields_to_use=['votes', 'rating'],
                                         label='genres', task=task, is_multilabel=True, shuffle=False)

    class_labels = data_num.target

    img_files_path = f'{files_path}/*.jpeg'
    img_path = os.path.join(str(fedot_project_root()), img_files_path)

    # import of image data
    data_img = InputData.from_image(images=img_path, labels=class_labels, task=task, target_size=images_size)
    # import of text data
    data_text = InputData.from_json_files(path, fields_to_use=['plot'],
                                          label='genres', task=task,
                                          data_type=DataTypesEnum.text, is_multilabel=True, shuffle=False)

    data = MultiModalData({
        # 'data_source_img': data_img,
        'data_source_table': data_num,
        'data_source_text': data_text
    })

    return data


def generate_initial_pipeline_and_data(data: Union[InputData, MultiModalData],
                                       with_split=True) -> tuple:
    """
    Generates initial pipeline for data from 3 different sources (table, images and text)
    Each source is the primary node for its subpipeline

    :param data: multimodal data (from 3 different sources: table, text, image)
    :param with_split: if True, splits the sample on train/test
    :return: pipeline object, 2 multimodal data objects (fit and predict)
    """

    # Identifying a number of classes for CNN params
    if data.target.shape[1] > 1:
        num_classes = data.target.shape[1]
    else:
        num_classes = data.num_classes
    # image
    # images_size = data['data_source_img'].features.shape[1:4]
    # ds_image = PrimaryNode('data_source_img')
    # image_node = SecondaryNode('cnn', nodes_from=[ds_image])
    # image_node.custom_params = {'image_shape': images_size,
    #                            'architecture_type': 'simplified',
    #                            'num_classes': num_classes,
    #                            'epochs': 2,
    #                            'batch_size': 16,
    #                            'optimizer_parameters': {'loss': "binary_crossentropy",
    #                                                     'optimizer': "adam",
    #                                                     'metrics': 'categorical_crossentropy'}
    #                            }

    # table
    ds_table = PrimaryNode('data_source_table')
    numeric_node = SecondaryNode('scaling', nodes_from=[ds_table])

    # text
    ds_text = PrimaryNode('data_source_text')
    node_text_clean = SecondaryNode('text_clean', nodes_from=[ds_text])
    text_node = SecondaryNode('tfidf', nodes_from=[node_text_clean])
    text_node.custom_params = {'ngram_range': (1, 3), 'min_df': 0.001, 'max_df': 0.9}

    # combining all sources together
    logit_node = SecondaryNode('logit', nodes_from=[numeric_node, text_node])
    logit_node.custom_params = {'max_iter': 100000, 'random_state': 42}
    pipeline = Pipeline(logit_node)

    # train/test ratio
    ratio = 0.6
    if with_split:
        fit_data, predict_data = train_test_data_setup(data, shuffle_flag=True, split_ratio=ratio)
    else:
        fit_data, predict_data = data, data

    return pipeline, fit_data, predict_data


def run_multi_modal_pipeline(files_path: str, is_visualise=True) -> float:
    task = Task(TaskTypesEnum.classification)
    images_size = (224, 224)

    data = prepare_multi_modal_data(files_path, task, images_size)

    initial_pipeline, fit_data, predict_data = generate_initial_pipeline_and_data(data, with_split=True)

    automl_model = Fedot(problem='classification', timeout=0.1)
    pipeline = automl_model.fit(features=fit_data,
                                target=fit_data.target,
                                predefined_model='auto')

    if is_visualise:
        pipeline.show()

    prediction = pipeline.predict(predict_data, output_mode='labels')

    err = calculate_validation_metric(predict_data, prediction)

    print(f'F1 micro for validation sample is {err}')

    return err


if __name__ == '__main__':
    run_multi_modal_pipeline(files_path='examples/data/multimodal', is_visualise=True)
