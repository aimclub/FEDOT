import json
import os
import pickle
import traceback
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:
    from fedot.core.data.data import InputData
    from fedot.core.pipelines.pipeline import Pipeline

from fedot.core.utils import default_fedot_data_dir

FIT_ERROR_REPORT_DIR_ENV = 'FEDOT_FIT_ERROR_REPORT_DIR'


def is_recording_mode():
    return 'FEDOT_RECORDING_MODE' in os.environ


def is_analytic_mode():
    return 'FEDOT_ANALYTIC_MODE' in os.environ


def report_pipeline_fit_error(graph_id: str, exception: Exception) -> bool:
    """Write a lightweight fit-error report when a collector is configured."""
    report_dir = os.environ.get(FIT_ERROR_REPORT_DIR_ENV)
    if not report_dir:
        return False

    try:
        report_path = Path(report_dir)
        report_path.mkdir(parents=True, exist_ok=True)
        error_type = type(exception)
        report = {
            'exception_type': f'{error_type.__module__}.{error_type.__qualname__}',
            'message': str(exception),
            'graph_id': graph_id,
            'stack_trace': traceback.format_exc(),
        }
        report_file = report_path.joinpath(f'{os.getpid()}-{uuid4()}.json')
        report_file.write_text(json.dumps(report), encoding='utf-8')
        return True
    except Exception as ex:
        print(f'Failed to write pipeline fit-error report: {ex}')
        return False


def save_debug_info_for_pipeline(pipeline: 'Pipeline', train_data: 'InputData', test_data: 'InputData',
                                 exception: Exception, stack_trace: str):
    try:
        tmp_folder = Path(default_fedot_data_dir(), 'debug_info')
        if not tmp_folder.exists():
            os.mkdir(tmp_folder)

        pipeline_id = str(uuid4())
        base_path = Path(tmp_folder, pipeline_id)
        pipeline.save(f'{base_path}_pipeline', is_datetime_in_path=False)

        with open(f'{base_path}_train_data.pkl', 'wb') as file:
            pickle.dump(train_data, file)
        with open(f'{base_path}_test_data.pkl', 'wb') as file:
            pickle.dump(test_data, file)
        with open(f'{base_path}_exception.txt', 'w') as file:
            print(exception, file=file)
            print(stack_trace, file=file)
    except Exception as ex:
        print(ex)


def reproduce_fitting_error(pipeline_id: str, base_path=None, use_input_preprocessing: bool = True):
    from fedot.core.pipelines.pipeline import Pipeline

    if not base_path:
        base_path = Path(default_fedot_data_dir(), 'debug_info')

    pipeline = Pipeline(use_input_preprocessing=use_input_preprocessing)
    pipeline.load(str(Path(base_path, f'{pipeline_id}_pipeline', f'{pipeline_id}_pipeline.json')))
    with open(Path(base_path, f'{pipeline_id}_train_data.pkl'), 'rb') as file:
        train_data = pickle.load(file)
    with open(Path(base_path, f'{pipeline_id}_test_data.pkl'), 'rb') as file:
        test_data = pickle.load(file)
    pipeline.unfit()
    pipeline.fit(train_data)
    pipeline.predict(test_data)
