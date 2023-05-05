import os
import shutil
import zipfile
from pathlib import Path

import pytest

from fedot.api.main import Fedot
from fedot.core.utils import fedot_project_root
from fedot.utilities.project_import_export import export_project_to_zip, import_project_from_zip, DEFAULT_PROJECTS_PATH
from test.integration.models.test_atomized_model import create_pipeline
from test.unit.validation.test_table_cv import get_classification_data

PATHS_TO_DELETE_AFTER_TEST = []

LOG_NAME = 'log.log'


@pytest.fixture(scope="session", autouse=True)
def creation_model_files_before_after_tests(request):
    request.addfinalizer(delete_files_folders)


def delete_files_folders():
    for path_to_del in PATHS_TO_DELETE_AFTER_TEST:
        absolute_path = DEFAULT_PROJECTS_PATH.joinpath(path_to_del)
        if absolute_path.is_file():
            os.remove(absolute_path)
        elif absolute_path.is_dir():
            shutil.rmtree(absolute_path)


def test_export_project_correctly():
    folder_name = 'iris_classification'
    zip_name = Path(folder_name).with_suffix('.zip')
    path_to_zip = DEFAULT_PROJECTS_PATH.joinpath(zip_name)
    path_to_folder = DEFAULT_PROJECTS_PATH.joinpath(folder_name)

    PATHS_TO_DELETE_AFTER_TEST.append(zip_name)
    PATHS_TO_DELETE_AFTER_TEST.append(path_to_folder)

    pipeline = create_pipeline()
    train_data = test_data = get_classification_data()
    export_project_to_zip(zip_name=path_to_zip, opt_history=None,
                          pipeline=pipeline, train_data=train_data, test_data=test_data, log_file_name=LOG_NAME)

    assert path_to_zip.exists()

    with zipfile.ZipFile(path_to_zip) as zip_object:
        actual = {file.filename for file in zip_object.infolist()}
        expected = {LOG_NAME, 'train_data.csv', 'pipeline/', 'pipeline/pipeline.json', 'test_data.csv'}
        assert actual == expected


def test_import_project_correctly():
    folder_path = Path(fedot_project_root(), 'test', 'data', 'project', 'iris_classification')
    zip_path = Path(folder_path).with_suffix('.zip')

    assert zip_path.exists()

    pipeline, train_data, test_data, opt_history = import_project_from_zip(zip_path)

    assert pipeline is not None
    assert train_data is not None
    assert test_data is not None

    assert opt_history is None

    assert pipeline.fit(train_data)
    assert pipeline.predict(test_data)


def test_export_import_api_correctly():
    folder_name = 'api_classification'
    zip_name = Path(folder_name).with_suffix('.zip')
    path_to_zip = DEFAULT_PROJECTS_PATH.joinpath(zip_name)
    path_to_folder = DEFAULT_PROJECTS_PATH.joinpath(folder_name)

    PATHS_TO_DELETE_AFTER_TEST.append(zip_name)
    PATHS_TO_DELETE_AFTER_TEST.append(path_to_folder)

    train_data = test_data = get_classification_data()

    api = Fedot(problem='classification', timeout=-1,
                with_tuning=False,
                num_of_generations=1,
                pop_size=3,
                show_progress=False)

    api.fit(train_data)
    api.predict(test_data)

    api.export_as_project(path_to_zip)

    new_api = Fedot(problem='classification')
    new_api.import_as_project(path_to_zip)

    assert len(new_api.get_metrics()) > 0
    assert new_api.history is not None
    assert new_api.current_pipeline is not None
    assert new_api.train_data is not None
    assert new_api.test_data is not None
