import os

from fedot.api.main import Fedot

from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.utils import fedot_project_root


def prepare_multi_modal_data(files_path: str, task: Task) -> MultiModalData:
    """
    Imports data from 2 different sources (table and text)

    :param files_path: path to data
    :param task: task to solve
    :return: MultiModalData object which contains table and text data
    """

    path = os.path.join(str(fedot_project_root()), files_path)

    # import of table data
    path_table = os.path.join(path, 'multimodal_wine_table.csv')
    data_num = InputData.from_csv(path_table, task=task, target_columns='variety')

    # import of text data
    path_text = os.path.join(path, 'multimodal_wine_text.csv')
    data_text = InputData.from_csv(path_text, data_type=DataTypesEnum.text, task=task, target_columns='variety')

    data = MultiModalData({
        'data_source_table': data_num,
        'data_source_text': data_text
    })

    return data


def run_multi_modal_pipeline(files_path: str, is_visualise=True) -> float:
    task = Task(TaskTypesEnum.classification)

    data = prepare_multi_modal_data(files_path, task)
    fit_data, predict_data = train_test_data_setup(data, shuffle_flag=True, split_ratio=0.7)

    automl_model = Fedot(problem='classification', timeout=10)
    automl_model.fit(features=fit_data,
                     target=fit_data.target)

    prediction = automl_model.predict(predict_data)
    metrics = automl_model.get_metrics()

    if is_visualise:
        automl_model.current_pipeline.show()

    print(f'F1 for validation sample is {round(metrics["f1"], 3)}')

    return metrics["f1"]


if __name__ == '__main__':
    run_multi_modal_pipeline(files_path='examples/data/multimodal_wine', is_visualise=True)
