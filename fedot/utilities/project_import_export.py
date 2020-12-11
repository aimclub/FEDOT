import os
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

from fedot.core.data.data import InputData
from fedot.core.log import Log, default_log
from fedot.core.optimisers.opt_history import OptHistory
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.utils import default_fedot_data_dir

DEFAULT_PATH = default_fedot_data_dir()
DEFAULT_PROJECTS_PATH = os.path.join(DEFAULT_PATH, 'projects')


def export_project_to_zip(zip_name: str, pipeline: Pipeline, train_data: InputData, test_data: InputData,
                          opt_history: Optional[OptHistory] = None, log_file_name: str = 'log.txt'):
    """
    Convert pipeline to JSON, data to csv, compress them to zip
    archive and save to 'DEFAULT_PROJECTS_PATH/projects' with logs.

    :param zip_name: name of the zip file with exported project
    :param pipeline: pipeline object to export
    :param train_data: train InputData object to export
    :param test_data: test InputData object to export
    :param opt_history: history of model optimisation
    :param log_file_name: name of the file with log to export
    """

    log = default_log('fedot.utilities.project_import_export')
    absolute_folder_path, absolute_zip_path, folder_name, zip_name = _prepare_paths(zip_name)
    _check_for_existing_project(absolute_folder_path)

    # Converts python objects to files for compression
    pipeline.save(os.path.join(absolute_folder_path, 'pipeline.json'), datetime_in_path=False)
    train_data.to_csv(os.path.join(absolute_folder_path, 'train_data.csv'))
    test_data.to_csv(os.path.join(absolute_folder_path, 'test_data.csv'))
    if opt_history is not None:
        opt_history.save(Path(os.path.join(absolute_folder_path, 'opt_history.json')))

    _copy_log_file(log_file_name, absolute_folder_path)

    shutil.make_archive(absolute_folder_path, 'zip', absolute_folder_path)
    shutil.rmtree(absolute_folder_path)

    log.info(f'The exported project was saved on the path: {absolute_folder_path}')


def import_project_from_zip(zip_path: str) -> Tuple[Pipeline, InputData, InputData, OptHistory]:
    """
    Unzipping zip file. Zip file should contains:
    - pipeline.json: json performance,
    - train_data.csv: csv with first line which contains task_type and data_type of train InputData object,
    - test_data.csv: csv with first line which contains task_type and data_type of test InputData object.

    Created Pipeline and InputData objects. Ready to work with it.

    :param zip_path: path to zip archive
    :return [Pipeline, InputData, InputData]: return array of Pipeline and InputData objects.
    """
    log = default_log('fedot.utilities.project_import_export')

    zip_path = _check_zip_path(zip_path, log)
    zip_name = _get_zip_name(zip_path)
    folder_path = os.path.join(DEFAULT_PROJECTS_PATH, zip_name)

    shutil.unpack_archive(zip_path, folder_path)

    message = f"The project '{zip_name}' was unpacked to the '{folder_path}'."
    log.info(message)
    pipeline, train_data, test_data, opt_history = None, None, None, None
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file == 'pipeline.json':
                pipeline = Pipeline()
                pipeline.load(os.path.join(root, file))
            elif file == 'train_data.csv':
                train_data = InputData.from_csv(os.path.join(root, file))
            elif file == 'test_data.csv':
                test_data = InputData.from_csv(os.path.join(root, file))
            elif file == 'opt_history.json':
                opt_history = OptHistory.load(os.path.join(root, file))

    return pipeline, train_data, test_data, opt_history


def _get_zip_name(zip_path: str) -> str:
    zip_path_split = os.path.split(zip_path)
    zip_name = zip_path_split[-1].split('.')
    return zip_name[0]


def _check_zip_path(zip_path: str, log: Log):
    """Check 'zip_path' for correctness."""

    if '.zip' not in zip_path:
        zip_path = f'{zip_path}.zip'

    if not os.path.exists(zip_path):
        message = f"File with the path '{zip_path}' could not be found."
        log.error(message)
        raise FileExistsError(message)

    zip_path_split = os.path.split(zip_path)

    if zip_path_split[-1].split('.')[-1] != 'zip':
        message = f"Zipfile must be with 'zip' extension."
        log.error(message)
        raise FileExistsError(message)
    return zip_path


def _copy_log_file(log_file_name: str, absolute_folder_path: str):
    """Copy log file to folder which will be compressed."""

    if log_file_name is not None:
        if not os.path.isabs(log_file_name):
            log_file_name = os.path.abspath(os.path.join(DEFAULT_PATH, log_file_name))

        if os.path.exists(log_file_name):
            shutil.copy2(log_file_name, os.path.join(absolute_folder_path, os.path.split(log_file_name)[-1]))


def _prepare_paths(zip_name: str) -> List[str]:
    """Prepared absolute paths for zip and project's folder."""

    name_split = zip_name.split('.')
    folder_name = zip_name

    if len(name_split) == 2:
        folder_name = name_split[0]
    else:
        zip_name = zip_name + '.zip'

    absolute_folder_path = os.path.join(DEFAULT_PROJECTS_PATH, folder_name)
    absolute_zip_path = os.path.join(absolute_folder_path, zip_name)

    return [absolute_folder_path, absolute_zip_path, folder_name, zip_name]


def _check_for_existing_project(absolute_folder_path):
    """Check for existing folder and zipfile of project. Create it, if it is no exists."""

    if os.path.exists(absolute_folder_path + '.zip'):
        message = f"Zipfile with the name '{absolute_folder_path + '.zip'}' exists."
        raise FileExistsError(message)

    if os.path.exists(absolute_folder_path):
        message = f"Project with the name '{absolute_folder_path}' exists."
        raise FileExistsError(message)
    else:
        os.makedirs(absolute_folder_path)
