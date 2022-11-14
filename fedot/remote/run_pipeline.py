import json
import os
import sys
from typing import Union

from fedot.core.data.data import InputData
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.log import default_log
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.verification import verifier_for_task
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TaskTypesEnum
from fedot.core.utilities.random import RandomStateHandler
from fedot.remote.pipeline_run_config import PipelineRunConfig


def new_key_name(data_part_key):
    if data_part_key == 'idx':
        return 'idx'
    return f'data_source_ts/{data_part_key}'


def _load_ts_data(config):
    task = config.task
    if config.is_multi_modal:
        train_data = MultiModalData.from_csv_time_series(
            file_path=config.input_data,
            task=task, target_column=config.target,
            var_names=config.var_names)
    else:
        train_data = InputData.from_csv_time_series(
            file_path=config.input_data,
            task=task, target_column=config.target)
    return train_data


def _load_data(config):
    data_type = DataTypesEnum.table
    if config.task.task_type == TaskTypesEnum.ts_forecasting:
        train_data = _load_ts_data(config)
    else:
        train_data = InputData.from_csv(file_path=config.input_data,
                                        task=config.task, data_type=data_type)
    return train_data


def fit_pipeline(config_file: Union[str, bytes]) -> bool:
    logger = default_log(prefix='pipeline_fitting_logger')

    config = \
        PipelineRunConfig().load_from_file(config_file)

    verifier = verifier_for_task(config.task.task_type)

    pipeline = pipeline_from_json(config.pipeline_template)

    train_data = _load_data(config)

    # subset data using indices
    if config.train_data_idx not in [None, []]:
        train_data = train_data.subset_indices(config.train_data_idx)

    if not verifier(pipeline):
        logger.error('Pipeline not valid')
        return False

    try:
        RandomStateHandler.MODEL_FITTING_SEED = 0
        pipeline.fit_from_scratch(train_data)
    except Exception as ex:
        logger.error(ex)
        return False

    if config.test_data_path:
        test_data = InputData.from_csv(config.test_data_path)
        pipeline.predict(test_data)

    pipeline.save(path=os.path.join(config.output_path, 'fitted_pipeline'), create_subdir=False,
                  is_datetime_in_path=False)

    return True


def pipeline_from_json(json_str: str):
    json_dict = json.loads(json_str)
    pipeline = Pipeline.from_serialized(json_dict)

    return pipeline


if __name__ == '__main__':
    config_file_name_from_argv = sys.argv[1]
    fit_pipeline(config_file_name_from_argv)
