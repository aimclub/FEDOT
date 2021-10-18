import ast

from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams

tmp_task = Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(forecast_length=1))


class PipelineRunConfig:
    """
    Quasi-dataclass for the input parameters of external pipeline fitting
    """

    def __init__(self, config):
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
            self.target = ast.literal_eval(config['DEFAULT']['target'])

        self.test_data_path = config['OPTIONAL'].get('test_data')
