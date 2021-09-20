import ast
import configparser
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
from fedot.core.pipelines.template import PipelineTemplate
from fedot.core.pipelines.validation import validate
# required for the import of task from file
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams

tmp_task = Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(forecast_length=1))


def extract_data_from_config_file(file):
    config = configparser.ConfigParser()
    if not os.path.exists(file):
        raise ValueError('Config not found')
    config.read(file, encoding='utf-8')
    pipeline_description = config['DEFAULT']['pipeline_description']
    train_data_idx = ast.literal_eval(config['DEFAULT']['train_data_idx'])
    input_data = config['DEFAULT']['train_data']
    task = eval(config['DEFAULT']['task'])
    output_path = config['DEFAULT']['output_path']
    is_multi_modal = False
    if 'is_multi_modal' in config['DEFAULT']:
        is_multi_modal = ast.literal_eval(config['DEFAULT']['is_multi_modal'])
    var_names = False
    if 'var_names' in config['DEFAULT']:
        var_names = ast.literal_eval(config['DEFAULT']['var_names'])

    test_data_path = config['OPTIONAL'].get('test_data')

    return pipeline_description, input_data, train_data_idx, task, \
           test_data_path, output_path, is_multi_modal, var_names


def fit_pipeline(config_file) -> bool:
    status = True
    try:
        (pipeline_description, train_data_path, train_data_idx,
         task, test_data_path, output_path, is_multi_modal, var_names) = \
            extract_data_from_config_file(config_file)

        pipeline = pipeline_from_json(pipeline_description)

        data_type = DataTypesEnum.table
        if task.task_type == TaskTypesEnum.ts_forecasting:
            data_type = DataTypesEnum.ts

        if is_multi_modal and task.task_type == TaskTypesEnum.ts_forecasting:
            df = pd.read_csv(train_data_path,
                             parse_dates=['datetime'])
            idx = [str(d) for d in df['datetime']]
            if not var_names:
                var_names = list(set(df.columns) - set('datetime'))
            train_data, _ = \
                prepare_multimodal_data(dataframe=df,
                                        features=var_names,
                                        forecast_length=0)

            target = np.array(df['diesel_fuel_kWh'])

            # create labels for data sources
            data_part_transformation_func = partial(array_to_input_data, idx=idx,
                                                    target_array=target, task=task)
            sources = dict((f'data_source_ts/{data_part_key}', data_part_transformation_func(features_array=data_part))
                           for (data_part_key, data_part) in train_data.items())
            train_data = MultiModalData(sources)
        else:
            train_data = InputData.from_csv(file_path=train_data_path,
                                            task=task, data_type=data_type)

        train_data = train_data.subset_list(train_data_idx)
        # pipeline.show()
        if not validate(pipeline, task=task):
            return False

        pipeline.fit_from_scratch(train_data)
        pipeline.show()
        if test_data_path:
            test_data = InputData.from_csv(test_data_path)
            pipeline.predict(test_data)

        pipeline.save(path=output_path)
    except Exception as ex:
        status = False
        print(f'Pipeline processing failed: {ex}')
    return status


def pipeline_from_json(json_str: str):
    json_dict = json.loads(json_str)
    pipeline = Pipeline()
    pipeline.nodes = []
    pipeline.template = PipelineTemplate(pipeline, pipeline.log)

    pipeline.template._extract_operations(json_dict, None)
    pipeline.template.convert_to_pipeline(pipeline.template.link_to_empty_pipeline, None)
    pipeline.template.depth = pipeline.template.link_to_empty_pipeline.depth

    return pipeline


if __name__ == '__main__':
    config_file = sys.argv[1]
    fit_pipeline(config_file)
