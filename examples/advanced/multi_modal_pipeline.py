import os

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


def calculate_validation_metric(valid: InputData, pred: OutputData) -> float:
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


def prepare_multi_modal_data(files_path: str, task: Task, images_size: tuple = (128, 128), with_split=True) -> tuple:
    """
    Imports data from 3 different sources (table, images and text)

    :param files_path: path to data
    :param task: task to solve
    :param images_size: the requested size in pixels, as a 2-tuple of (width, height)
    :param with_split: if True, splits the sample on train/test
    :return: 6 OutputData objects (2 with table data, 2 with images, 2 with text)
    """

    path = os.path.join(str(fedot_project_root()), files_path)
    # unpacking of data archive
    unpack_archived_data(path)
    # import of table data
    data_num = InputData.from_json_files(path, fields_to_use=['votes', 'rating'],
                                         label='genres', task=task, is_multilabel=True, shuffle=False)

    class_labels = data_num.target
    # train/test ratio
    ratio = 0.6

    img_files_path = f'{files_path}/*.jpeg'
    img_path = os.path.join(str(fedot_project_root()), img_files_path)

    # import of image data
    data_img = InputData.from_image(images=img_path, labels=class_labels, task=task, target_size=images_size)
    # import of text data
    data_text = InputData.from_json_files(path, fields_to_use=['plot'],
                                          label='genres', task=task,
                                          data_type=DataTypesEnum.text, is_multilabel=True, shuffle=False)

    if with_split:

        train_num, test_num = train_test_data_setup(data_num, shuffle_flag=True, split_ratio=ratio)
        train_img, test_img = train_test_data_setup(data_img, shuffle_flag=True, split_ratio=ratio)
        train_text, test_text = train_test_data_setup(data_text, shuffle_flag=True, split_ratio=ratio)
    else:

        train_num, test_num = data_num, data_num
        train_img, test_img = data_img, data_img
        train_text, test_text = data_text, data_text

    return train_num, test_num, train_img, test_img, train_text, test_text


def generate_initial_pipeline_and_data(images_size: tuple,
                                       train_num: InputData, test_num: InputData,
                                       train_img: InputData, test_img: InputData,
                                       train_text: InputData, test_text: InputData) -> tuple:
    """
    Generates initial pipeline for data from 3 different sources (table, images and text)
    Each source is the primary node for its subpipeline

    :param images_size: the requested size in pixels, as a 2-tuple of (width, height)
    :param train_num: train sample of table data
    :param test_num: test sample of table data
    :param train_img: train sample of image data
    :param test_img: test sample of image data
    :param train_text: train sample of text data
    :param test_text: test sample of text data
    :return: pipeline object, 2 multimodal data objects (fit and predict)
    """

    # image
    ds_image = PrimaryNode('data_source_img/1')
    image_node = SecondaryNode('cnn', nodes_from=[ds_image])
    image_node.custom_params = {'image_shape': (images_size[0], images_size[1], 1),
                                'architecture': 'simplified',
                                'num_classes': 5,
                                'epochs': 10,
                                'batch_size': 16,
                                'optimizer_parameters': {'loss': "binary_crossentropy",
                                                         'optimizer': "adam",
                                                         'metrics': 'categorical_crossentropy'}
                                }

    # table
    ds_table = PrimaryNode('data_source_table/2')
    numeric_node = SecondaryNode('scaling', nodes_from=[ds_table])

    # text
    ds_text = PrimaryNode('data_source_text/3')
    node_text_clean = SecondaryNode('text_clean', nodes_from=[ds_text])
    text_node = SecondaryNode('tfidf', nodes_from=[node_text_clean])
    text_node.custom_params = {'ngram_range': (1, 3), 'min_df': 0.001, 'max_df': 0.9}

    # combining all sources together
    logit_node = SecondaryNode('logit', nodes_from=[image_node, numeric_node, text_node])
    logit_node.custom_params = {'max_iter': 100000, 'random_state': 42}
    pipeline = Pipeline(logit_node)

    fit_data = MultiModalData({
        'data_source_img/1': train_img,
        'data_source_table/2': train_num,
        'data_source_text/3': train_text
    })
    predict_data = MultiModalData({
        'data_source_img/1': test_img,
        'data_source_table/2': test_num,
        'data_source_text/3': test_text
    })

    return pipeline, fit_data, predict_data


def run_multi_modal_pipeline(files_path: str, is_visualise=False) -> float:
    task = Task(TaskTypesEnum.classification)
    images_size = (128, 128)

    train_num, test_num, train_img, test_img, train_text, test_text = \
        prepare_multi_modal_data(files_path, task, images_size)

    pipeline, fit_data, predict_data = generate_initial_pipeline_and_data(images_size,
                                                                          train_num, test_num,
                                                                          train_img, test_img,
                                                                          train_text, test_text)

    pipeline.fit(input_data=fit_data)

    if is_visualise:
        pipeline.show()

    prediction = pipeline.predict(predict_data, output_mode='labels')

    err = calculate_validation_metric(test_text, prediction)

    print(f'F1 micro for validation sample is {err}')

    return err


if __name__ == '__main__':
    run_multi_modal_pipeline(files_path='examples/data/multimodal', is_visualise=True)
