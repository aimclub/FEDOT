import os
import shutil
from pathlib import Path
from typing import Optional, Tuple, Union

from fedot.core.data.data import InputData
from fedot.core.log import LoggerAdapter, default_log
from fedot.core.optimisers.opt_history_objects.opt_history import OptHistory
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.utils import default_fedot_data_dir

DEFAULT_PATH = Path(default_fedot_data_dir())
DEFAULT_PROJECTS_PATH = DEFAULT_PATH.joinpath('projects')


def export_project_to_zip(zip_name: Union[str, Path], pipeline: Pipeline, train_data: InputData, test_data: InputData,
                          opt_history: Optional[OptHistory] = None, log_file_name: str = 'log.txt'):
    """
    Convert pipeline to JSON, data to csv, compress them to zip
    archive and save to 'DEFAULT_PROJECTS_PATH/projects' with logs.

    :param zip_name: absolute or relative path the zip file with exported project
    :param pipeline: pipeline object to export
    :param train_data: train InputData object to export
    :param test_data: test InputData object to export
    :param opt_history: history of model optimisation to export (if available)
    :param log_file_name: name of the file with log to export
    """

    log = default_log(prefix='fedot.utilities.project_import_export')
    absolute_folder_path, absolute_zip_path, folder_name, zip_name = _prepare_paths(zip_name)
    _check_for_existing_project(absolute_folder_path, absolute_zip_path)

    # Converts python objects to files for compression
    pipeline_path = os.path.join(absolute_folder_path, 'pipeline', 'pipeline.json')
    pipeline.save(pipeline_path, is_datetime_in_path=False, create_subdir=False)
    train_data.to_csv(absolute_folder_path.joinpath('train_data.csv'))
    test_data.to_csv(absolute_folder_path.joinpath('test_data.csv'))
    if opt_history is not None:
        opt_history.save(Path(absolute_folder_path, 'opt_history.json'))

    _copy_log_file(log_file_name, absolute_folder_path)

    shutil.make_archive(base_name=absolute_zip_path.with_suffix(''), format='zip', root_dir=absolute_folder_path)
    folder_to_delete = DEFAULT_PROJECTS_PATH.joinpath(
        absolute_folder_path.relative_to(DEFAULT_PROJECTS_PATH).parts[0])
    shutil.rmtree(folder_to_delete)

    log.info(f'The exported project was saved on the path: {absolute_zip_path}')


def import_project_from_zip(zip_path: str) -> Tuple[Pipeline, InputData, InputData, OptHistory]:
    """
    Unzipping zip file. Zip file should contain:
    - pipeline.json: json performance,
    - train_data.csv: csv with first line which contains task_type and data_type of train InputData object,
    - test_data.csv: csv with first line which contains task_type and data_type of test InputData object.

    Created Pipeline and InputData objects. Ready to work with it.

    :param zip_path: path to zip archive
    :return imported classes
    """
    log = default_log(prefix='fedot.utilities.project_import_export')

    folder_path, absolute_zip_path, _, zip_name = _prepare_paths(zip_path)

    zip_path = _check_zip_path(absolute_zip_path, log)

    if folder_path.exists():
        # ensure temporary folder is clear
        shutil.rmtree(folder_path)
    shutil.unpack_archive(zip_path, folder_path)

    message = f'The project "{zip_name}" was unpacked to the "{folder_path}".'
    log.info(message)
    pipeline, train_data, test_data, opt_history = [None] * 4
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file == 'pipeline.json':
                pipeline = Pipeline.from_serialized(os.path.join(root, file))
            elif file == 'train_data.csv':
                train_data = InputData.from_csv(os.path.join(root, file))
            elif file == 'test_data.csv':
                test_data = InputData.from_csv(os.path.join(root, file))
            elif file == 'opt_history.json':
                opt_history = OptHistory.load(os.path.join(root, file))

    shutil.rmtree(folder_path)
    return pipeline, train_data, test_data, opt_history


def _check_zip_path(zip_path: Path, log: LoggerAdapter) -> Path:
    """Check 'zip_path' for correctness."""

    zip_path = zip_path.with_suffix('.zip')

    if not zip_path.exists():
        message = f'File with the path "{zip_path}" could not be found.'
        log.error(message)
        raise FileNotFoundError(message)
    return zip_path


def _copy_log_file(log_file_name: Optional[str], absolute_folder_path: Path):
    """Copy log file to folder which will be compressed."""

    if log_file_name is None:
        return

    log_file_name = Path(log_file_name)

    if not log_file_name.is_absolute():
        log_file_name = DEFAULT_PATH.joinpath(log_file_name).resolve()

    if log_file_name.exists():
        shutil.copy2(log_file_name,
                     absolute_folder_path.joinpath(log_file_name.stem).with_suffix(log_file_name.suffix))


def _prepare_paths(zip_path: Union[str, Path]) -> Tuple[Path, Path, Path, Path]:
    """Prepared paths and names: absolute folder path, absolute zip path, folder name, zip name."""

    zip_path = Path(zip_path)
    if Path(zip_path).suffix:
        folder_name = Path(zip_path.with_suffix('').name)
    else:
        folder_name = Path(zip_path.name)
        zip_path = zip_path.with_suffix('.zip')

    absolute_folder_path = DEFAULT_PROJECTS_PATH.joinpath(folder_name)
    if not zip_path.is_absolute():
        absolute_zip_path = Path.cwd().joinpath(zip_path)
    else:
        absolute_zip_path = zip_path
        zip_path = Path(zip_path.name)
    return absolute_folder_path, absolute_zip_path, folder_name, zip_path


def _check_for_existing_project(absolute_folder_path: Path, zip_path: Path):
    """Check for existing folder and zipfile of project. Create it, if it is no exists."""
    if zip_path.exists():
        message = f'Zipfile with the name "{zip_path}" exists.'
        raise FileExistsError(message)

    if absolute_folder_path.exists():
        message = f'Project with the name "{absolute_folder_path}" exists.'
        raise FileExistsError(message)
    else:
        absolute_folder_path.mkdir(parents=True)
