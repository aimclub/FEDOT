import os

from fedot.api.main import Fedot

from fedot.core.data.data_split import train_test_data_setup
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.utils import fedot_project_root


def run_multi_modal_example(files_path: str, is_visualise=True) -> float:
    task = Task(TaskTypesEnum.classification)

    path = os.path.join(str(fedot_project_root()), files_path, 'multimodal_wine.csv')
    data = MultiModalData.from_csv(file_path=path, task=task, target_columns='variety', index_col=None)
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
    run_multi_modal_example(files_path='examples/data/multimodal_wine', is_visualise=True)
