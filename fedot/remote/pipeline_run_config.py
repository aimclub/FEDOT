import ast
import configparser
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

from pymonad.either import Left, Right

from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.core.utils import fedot_project_root

_TASK_PATTERN = re.compile(
    r'^Task\(TaskTypesEnum\.(?P<task_type>[a-z_]+)'
    r'(?:,\s*TsForecastingParams\(forecast_length=(?P<forecast_length>\d+)\))?\)$'
)


@dataclass(frozen=True)
class PipelineRunConfigError:
    code: str
    message: str
    details: Dict[str, Any]


@dataclass(frozen=True)
class PipelineRunConfigPayload:
    pipeline_template: str
    input_data: str
    task: Task
    output_path: str
    train_data_idx: Optional[list] = None
    is_multi_modal: bool = False
    var_names: Optional[list] = None
    target: Optional[Union[str, list]] = None
    test_data_path: Optional[str] = None


class PipelineRunConfig:
    """
    OOP config object for external pipeline fitting with an immutable typed payload inside.
    """

    def __init__(self, payload: Optional[PipelineRunConfigPayload] = None):
        self._payload = payload

    @classmethod
    def try_from_dict(cls, config_dict: Dict[str, Dict[str, str]]):
        payload_result = parse_pipeline_run_config_dict(config_dict)
        if payload_result.__class__ is Left:
            return payload_result
        return Right(cls(payload_result.value))

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Dict[str, str]]):
        result = cls.try_from_dict(config_dict)
        if result.__class__ is Left:
            raise ValueError(result.value.message)
        return result.value

    @classmethod
    def try_from_parser(cls, config: configparser.ConfigParser):
        sections_result = _config_parser_to_dict(config)
        if sections_result.__class__ is Left:
            return sections_result
        return cls.try_from_dict(sections_result.value)

    @classmethod
    def from_parser(cls, config: configparser.ConfigParser):
        result = cls.try_from_parser(config)
        if result.__class__ is Left:
            raise ValueError(result.value.message)
        return result.value

    @classmethod
    def try_from_file(cls, file: Union[str, bytes]):
        parser_result = _read_config_parser(file)
        if parser_result.__class__ is Left:
            return parser_result
        return cls.try_from_parser(parser_result.value)

    @classmethod
    def from_file(cls, file: Union[str, bytes]):
        result = cls.try_from_file(file)
        if result.__class__ is Left:
            raise ValueError(result.value.message)
        return result.value

    def load_from_file(self, file: Union[str, bytes]):
        return type(self).from_file(file)

    @property
    def pipeline_template(self) -> str:
        return self._require_payload().pipeline_template

    @property
    def input_data(self) -> str:
        return self._require_payload().input_data

    @property
    def task(self) -> Task:
        return self._require_payload().task

    @property
    def output_path(self) -> str:
        return self._require_payload().output_path

    @property
    def train_data_idx(self) -> Optional[list]:
        return self._require_payload().train_data_idx

    @property
    def is_multi_modal(self) -> bool:
        return self._require_payload().is_multi_modal

    @property
    def var_names(self) -> Optional[list]:
        return self._require_payload().var_names

    @property
    def target(self) -> Optional[Union[str, list]]:
        return self._require_payload().target

    @property
    def test_data_path(self) -> Optional[str]:
        return self._require_payload().test_data_path

    def as_payload(self) -> PipelineRunConfigPayload:
        return self._require_payload()

    def _require_payload(self) -> PipelineRunConfigPayload:
        if self._payload is None:
            raise ValueError('PipelineRunConfig payload is not initialized.')
        return self._payload


def parse_pipeline_run_config_dict(config_dict: Dict[str, Dict[str, str]]):
    default_section = config_dict.get('DEFAULT')
    optional_section = config_dict.get('OPTIONAL', {})
    if default_section is None:
        return Left(_error('missing_default_section', 'Config must contain DEFAULT section.'))

    required_fields = ('pipeline_template', 'train_data', 'task', 'output_path')
    for field in required_fields:
        if field not in default_section:
            return Left(_error('missing_required_field',
                               f'Config DEFAULT section must contain "{field}".',
                               field=field))

    task_result = _parse_task(default_section['task'])
    if task_result.__class__ is Left:
        return task_result

    train_data_idx_result = _parse_optional_literal(default_section.get('train_data_idx'), 'train_data_idx')
    if train_data_idx_result.__class__ is Left:
        return train_data_idx_result

    is_multi_modal_result = _parse_optional_bool(default_section.get('is_multi_modal'), default=False)
    if is_multi_modal_result.__class__ is Left:
        return is_multi_modal_result

    var_names_result = _parse_optional_literal(default_section.get('var_names'), 'var_names')
    if var_names_result.__class__ is Left:
        return var_names_result

    target_result = _parse_target(default_section.get('target'))
    if target_result.__class__ is Left:
        return target_result

    input_data = _expand_base_path(default_section['train_data'])
    test_data_path = _expand_base_path(optional_section.get('test_data')) if optional_section else None

    payload = PipelineRunConfigPayload(
        pipeline_template=default_section['pipeline_template'],
        input_data=input_data,
        task=task_result.value,
        output_path=default_section['output_path'],
        train_data_idx=train_data_idx_result.value,
        is_multi_modal=is_multi_modal_result.value,
        var_names=var_names_result.value,
        target=target_result.value,
        test_data_path=test_data_path,
    )
    return Right(payload)


def _config_parser_to_dict(config: configparser.ConfigParser):
    if 'DEFAULT' not in config:
        return Left(_error('missing_default_section', 'Config must contain DEFAULT section.'))

    return Right({
        'DEFAULT': dict(config['DEFAULT']),
        'OPTIONAL': dict(config['OPTIONAL']) if 'OPTIONAL' in config else {},
    })


def _read_config_parser(file: Union[str, bytes]):
    config = configparser.ConfigParser()

    if isinstance(file, bytes):
        config.read_string(file.decode('utf-8'))
        return Right(config)

    if not os.path.exists(file):
        return Left(_error('config_not_found', 'Config not found.', path=file))

    config.read(file, encoding='utf-8')
    return Right(config)


def _parse_task(raw_task: str):
    if not isinstance(raw_task, str):
        return Left(_error('invalid_task_type', 'Task field must be a string representation.'))

    match = _TASK_PATTERN.fullmatch(raw_task.strip())
    if match is None:
        return Left(_error('unsupported_task_format',
                           'Task field must use supported Task(TaskTypesEnum.*) format.',
                           task=raw_task))

    task_type_name = match.group('task_type')
    try:
        task_type = TaskTypesEnum(task_type_name)
    except ValueError:
        return Left(_error('unknown_task_type', 'Unknown task type in config.', task_type=task_type_name))

    forecast_length = match.group('forecast_length')
    if forecast_length is None:
        return Right(Task(task_type))

    return Right(Task(task_type, TsForecastingParams(forecast_length=int(forecast_length))))


def _parse_optional_literal(raw_value: Optional[str], field_name: str):
    if raw_value is None:
        return Right(None)

    normalized = raw_value.strip()
    if normalized in ('', 'None'):
        return Right(None)

    try:
        return Right(ast.literal_eval(normalized))
    except (ValueError, SyntaxError) as ex:
        return Left(_error('invalid_literal',
                           f'Field "{field_name}" must be a valid Python literal.',
                           field=field_name,
                           exception=str(ex)))


def _parse_optional_bool(raw_value: Optional[str], default: bool):
    if raw_value is None:
        return Right(default)

    normalized = raw_value.strip()
    if normalized in ('', 'None'):
        return Right(default)

    try:
        value = ast.literal_eval(normalized)
    except (ValueError, SyntaxError):
        value = normalized

    if isinstance(value, bool):
        return Right(value)

    if isinstance(value, str) and value.lower() in ('true', 'false'):
        return Right(value.lower() == 'true')

    return Left(_error('invalid_bool', 'Boolean field must be True/False.', value=raw_value))


def _parse_target(raw_value: Optional[str]):
    if raw_value is None:
        return Right(None)

    normalized = raw_value.strip()
    if normalized in ('', 'None'):
        return Right(None)

    try:
        return Right(ast.literal_eval(normalized))
    except (ValueError, SyntaxError):
        return Right(normalized)


def _expand_base_path(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    if '{fedot_base_path}' not in path:
        return path
    return path.format(fedot_base_path=fedot_project_root())


def _error(code: str, message: str, **details: Any) -> PipelineRunConfigError:
    return PipelineRunConfigError(code=code, message=message, details=details)
