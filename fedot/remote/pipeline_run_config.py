import ast
import configparser
import os
from typing import Union

from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.core.utils import fedot_project_root

tmp_task = Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(forecast_length=1))


class PipelineRunConfig:
    """
    Quasi-dataclass for the input parameters of external pipeline fitting
    """

    def __init__(self, config=None):
        if config is None:
            return

        self.pipeline_template = config['DEFAULT']['pipeline_template']
        self.input_data = config['DEFAULT']['train_data']
        self.task = eval(config['DEFAULT']['task'])
        self.output_path = config['DEFAULT']['output_path']

        self.train_data_idx = None
        if config['DEFAULT']['train_data_idx'] != 'None':
            self.train_data_idx = ast.literal_eval(config['DEFAULT']['train_data_idx'])

        self.is_multi_modal = False
        if config['DEFAULT']['is_multi_modal'] != 'None':
            self.is_multi_modal = ast.literal_eval(config['DEFAULT']['is_multi_modal'])

        self.var_names = False
        if config['DEFAULT']['var_names'] != 'None':
            self.var_names = ast.literal_eval(config['DEFAULT']['var_names'])

        self.target = None
        if 'target' in config['DEFAULT'] and config['DEFAULT']['target'] != 'None':
            try:
                # list of target values
                self.target = ast.literal_eval(config['DEFAULT']['target'])
            except ValueError:
                # name of target column
                self.target = config['DEFAULT']['target']

        self.test_data_path = config['OPTIONAL'].get('test_data')

    def load_from_file(self, file: Union[str, bytes]):
        config = configparser.ConfigParser()

        if isinstance(file, bytes):
            config.read_string(file.decode('utf-8'))
        else:
            if not os.path.exists(file):
                raise ValueError('Config not found')
            config.read(file, encoding='utf-8')

        processed_config = PipelineRunConfig(config)

        if '{fedot_base_path}' in processed_config.input_data:
            processed_config.input_data = \
                processed_config.input_data.format(fedot_base_path=fedot_project_root())

        return processed_config
