import configparser

import pytest

from fedot.core.repository.tasks import TaskTypesEnum, TsForecastingParams
from fedot.remote.pipeline_run_config import PipelineRunConfig, parse_pipeline_run_config_dict


def _base_config(task='Task(TaskTypesEnum.classification)'):
    return {
        'DEFAULT': {
            'pipeline_template': '{}',
            'train_data': '{fedot_base_path}/test/data/advanced_classification.csv',
            'task': task,
            'output_path': './out',
            'train_data_idx': '[1, 2, 3]',
            'is_multi_modal': 'False',
            'var_names': 'None',
        },
        'OPTIONAL': {},
    }


def test_parse_pipeline_run_config_dict_parses_classification_task():
    result = parse_pipeline_run_config_dict(_base_config())

    assert result.is_right()
    assert result.value.task.task_type == TaskTypesEnum.classification
    assert result.value.train_data_idx == [1, 2, 3]
    assert result.value.var_names is None


def test_parse_pipeline_run_config_dict_parses_forecasting_task_with_params():
    result = parse_pipeline_run_config_dict(
        _base_config(task='Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(forecast_length=3))')
    )

    assert result.is_right()
    assert result.value.task.task_type == TaskTypesEnum.ts_forecasting
    assert isinstance(result.value.task.task_params, TsForecastingParams)
    assert result.value.task.task_params.forecast_length == 3


def test_parse_pipeline_run_config_dict_rejects_eval_like_task_payload():
    config = _base_config(task='__import__("os").system("echo hacked")')

    result = parse_pipeline_run_config_dict(config)

    assert result.is_left()
    assert result.monoid[0].code == 'unsupported_task_format'


@pytest.mark.parametrize('raw_value, expected', [('False', False), ('True', True), ('"True"', True), ('None', False)])
def test_pipeline_run_config_parses_bool_literals_compatibly(raw_value, expected):
    config = _base_config()
    config['DEFAULT']['is_multi_modal'] = raw_value

    result = parse_pipeline_run_config_dict(config)

    assert result.is_right()
    assert result.value.is_multi_modal is expected


def test_pipeline_run_config_from_parser_keeps_oop_factory_style():
    parser = configparser.ConfigParser()
    parser.read_dict(_base_config())

    config = PipelineRunConfig.from_parser(parser)

    assert config.task.task_type == TaskTypesEnum.classification
    assert config.train_data_idx == [1, 2, 3]
    assert config.input_data.endswith('test/data/advanced_classification.csv')


def test_pipeline_run_config_load_from_file_compatibility_wrapper(tmp_path):
    config_path = tmp_path / 'remote.ini'
    parser = configparser.ConfigParser()
    parser.read_dict(_base_config())
    with config_path.open('w', encoding='utf-8') as file:
        parser.write(file)

    config = PipelineRunConfig().load_from_file(str(config_path))

    assert config.output_path == './out'
    assert config.task.task_type == TaskTypesEnum.classification
