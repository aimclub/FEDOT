from fedot.core.optimisers.stages_builder.ts_stages import ts_stages
from fedot.core.repository.tasks import TaskTypesEnum


class OptimizerStagesBuilder:
    stages = None

    def __init__(self, task_type: TaskTypesEnum):
        if task_type is TaskTypesEnum.ts_forecasting:
            self.stages = ts_stages
        else:
            raise NotImplementedError('Stages are prepared only for time series forecasting task')

    def build(self):
        return self.stages
