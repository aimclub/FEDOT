import json
import os
import sys
from functools import partial

import numpy as np
import pandas as pd

from cases.industrial.processing import prepare_multimodal_data
from fedot.api.api_utils.data_definition import array_to_input_data
from fedot.core.data.data import InputData
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.validation import validate
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TaskTypesEnum
from fedot.remote.pipeline_run_config import PipelineRunConfig


def _load_data(config):
    data_type = DataTypesEnum.table
    if config.task.task_type == TaskTypesEnum.ts_forecasting:
        data_type = DataTypesEnum.ts
        df = pd.read_csv(config.input_data,
                         parse_dates=['datetime'])
        idx = [str(d) for d in df['datetime']]

        if config.is_multi_modal:
            var_names = config.var_names
            if not var_names:
                var_names = list(set(df.columns) - set('datetime'))
            train_data, _ = \
                prepare_multimodal_data(dataframe=df,
                                        features=var_names,
                                        forecast_length=0)

            if config.target is not None:
                target = np.array(df[config.target])
            else:
                target = np.array(df[df.columns[-1]])

            # create labels for data sources
            data_part_transformation_func = partial(array_to_input_data, idx=idx,
                                                    target_array=target, task=config.task)

            def new_key_name(data_part_key):
                if data_part_key == 'idx':
                    return 'idx'
                return f'data_source_ts/{data_part_key}'

            sources = dict((new_key_name(data_part_key),
                            data_part_transformation_func(features_array=data_part))
                           for (data_part_key, data_part) in train_data.items())
            train_data = MultiModalData(sources)
        else:
            train_data = InputData.from_csv(file_path=config.input_data,
                                            task=config.task, data_type=data_type)
            train_data.features = np.squeeze(train_data.features, axis=1)
            train_data.idx = idx
    else:
        train_data = InputData.from_csv(file_path=config.input_data,
                                        task=config.task, data_type=data_type)
    return train_data


def fit_pipeline(config_file) -> bool:
    status = True

    config = \
        PipelineRunConfig().load_from_file(config_file)

    pipeline = pipeline_from_json(config.pipeline_template)

    train_data = _load_data(config)

    # subset data using indices
    if config.train_data_idx not in [None, []]:
        train_data = train_data.subset_indices(config.train_data_idx)

    try:
        validate(pipeline, task=config.task)
    except ValueError:
        print('Pipeline not valid')
        return False

    try:
        pipeline.fit_from_scratch(train_data)
    except Exception as ex:
        print(ex)
        return False

    if config.test_data_path:
        test_data = InputData.from_csv(config.test_data_path)
        pipeline.predict(test_data)

    pipeline.save(path=os.path.join(config.output_path, 'fitted_pipeline'), datetime_in_path=False)

    return status


def pipeline_from_json(json_str: str):
    json_dict = json.loads(json_str)
    pipeline = Pipeline()
    pipeline.load(json_dict)

    return pipeline


if __name__ == '__main__':
    config_file = sys.argv[1]
    fit_pipeline(config_file)
