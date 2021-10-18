from inspect import signature
from pathlib import Path
from typing import Any, Dict

from fedot.shared.interfaces.serializable import Serializable


class LogSerializer(Serializable):

    def to_json(self) -> Dict[str, Any]:
        basic_serialization = super().to_json()
        del basic_serialization['logger']  # cause it will be automatically generated in __init__
        basic_serialization['logger_name'] = basic_serialization['name']
        basic_serialization['config_json_file'] = basic_serialization['config_file']
        del basic_serialization['config_file']
        basic_serialization['output_verbosity_level'] = basic_serialization['verbosity_level']
        del basic_serialization['verbosity_level']
        return basic_serialization

    @classmethod
    def from_json(cls, json_obj: Dict[str, Any]):
        init_data = {
            k: v
            for k, v in json_obj.items()
            if k in signature(cls.__init__).parameters
        }
        config_json_file, log_file = init_data['config_json_file'], init_data['log_file']
        if config_json_file != 'default' and not Path(config_json_file).exists():
            init_data['config_json_file'] = 'default'
        if init_data['config_json_file'] == 'default' and not Path(log_file).exists():
            init_data['log_file'] = None
        return super().from_json(init_data)
