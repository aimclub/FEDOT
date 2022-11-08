import os
from typing import Union

from sklearn.metrics import f1_score as f1

from cases.dataset_preparation import unpack_archived_data
from fedot.api.main import Fedot
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
        'data_source_img': data_img,
        'data_source_table': data_num,
        'data_source_text': data_text
    })

    return data


def run_multi_modal_pipeline(files_path: str, visualization=False) -> float:
    task = Task(TaskTypesEnum.classification)
    images_size = (224, 224)

    data = prepare_multi_modal_data(files_path, task, images_size)

    fit_data, predict_data = train_test_data_setup(data, shuffle_flag=True, split_ratio=0.6)

    automl_model = Fedot(problem='classification', timeout=15)
    pipeline = automl_model.fit(features=fit_data,
                                target=fit_data.target)

    if visualization:
        pipeline.show()

    prediction = pipeline.predict(predict_data, output_mode='labels')

    err = calculate_validation_metric(predict_data, prediction)

    print(f'F1 micro for validation sample is {err}')

    return err


if __name__ == '__main__':
    run_multi_modal_pipeline(files_path='examples/data/multimodal', visualization=True)
