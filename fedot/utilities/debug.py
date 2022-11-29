import os
import pickle
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:
    from fedot.core.data.data import InputData
    from fedot.core.pipelines.pipeline import Pipeline

from fedot.core.utils import default_fedot_data_dir


def is_test_session():
    return 'PYTEST_CURRENT_TEST' in os.environ


def is_recording_mode():
    return 'FEDOT_RECORDING_MODE' in os.environ


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


def reproduce_fitting_error(pipeline_id: str, base_path=None):
    from fedot.core.pipelines.pipeline import Pipeline

    if not base_path:
        base_path = Path(default_fedot_data_dir(), 'debug_info')

    pipeline = Pipeline()
    pipeline.load(str(Path(base_path, f'{pipeline_id}_pipeline', f'{pipeline_id}_pipeline.json')))
    with open(Path(base_path, f'{pipeline_id}_train_data.pkl'), 'rb') as file:
        train_data = pickle.load(file)
    with open(Path(base_path, f'{pipeline_id}_test_data.pkl'), 'rb') as file:
        test_data = pickle.load(file)
    pipeline.unfit()
    pipeline.fit(train_data)
    pipeline.predict(test_data)
