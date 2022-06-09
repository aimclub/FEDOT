from fedot.api.main import Fedot

from examples.advanced.multi_modal_pipeline import calculate_validation_metric, prepare_multi_modal_data
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.repository.tasks import Task, TaskTypesEnum


def run_multi_modal_case(files_path, is_visualise=True):
    task = Task(TaskTypesEnum.classification)
    images_size = (224, 224)

    data = prepare_multi_modal_data(files_path, task, images_size)

    fit_data, predict_data = train_test_data_setup(data, shuffle_flag=True, split_ratio=0.6)

    # tuner on image data is not implemented yet, timeout increase can cause unstable work
    automl_model = Fedot(problem='classification', timeout=0.1)
    pipeline = automl_model.fit(features=fit_data,
                                target=fit_data.target)

    if is_visualise:
        pipeline.show()

    prediction = pipeline.predict(predict_data, output_mode='labels')

    err = calculate_validation_metric(predict_data, prediction)

    print(f'F1 micro for validation sample is {err}')

    return err


def download_mmdb_dataset():
    # TODO change to uploadable full dataset
    pass


if __name__ == '__main__':
    download_mmdb_dataset()

    run_multi_modal_case('cases/data/mm_imdb', is_visualise=True)
