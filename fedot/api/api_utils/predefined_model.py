import traceback
from typing import Union

from fedot.api.api_utils.assumptions.assumptions_builder import AssumptionsBuilder
from fedot.core.data.data import InputData
from fedot.core.log import Log
from fedot.core.pipelines.node import PrimaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.verification import verify_pipeline, verifier_for_task


class PredefinedModel:
    def __init__(self, predefined_model: Union[str, Pipeline], data: InputData, log: Log):
        self.predefined_model = predefined_model
        self.data = data
        self.log = log
        self.pipeline = self._get_pipeline()

    def _get_pipeline(self) -> Pipeline:
        if isinstance(self.predefined_model, Pipeline):
            pipelines = self.predefined_model
        elif self.predefined_model == 'auto':
            # Generate initial assumption automatically
            pipelines = AssumptionsBuilder.get(self.data).from_operations().build()[0]
        elif isinstance(self.predefined_model, str):
            model = PrimaryNode(self.predefined_model)
            pipelines = Pipeline(model)
        else:
            raise ValueError(f'{type(self.predefined_model)} is not supported as Fedot model')

        verify_pipeline(pipelines, task_type=self.data.task.task_type, raise_on_failure=True)

        return pipelines

    def fit(self):
        try:
            self.pipeline.fit(self.data)
        except Exception as ex:
            fit_failed_info = f'Predefined model fit was failed due to: {ex}.'
            advice_info = f'{fit_failed_info} Check pipeline structure and the correctness of the data'
            self.log.message(fit_failed_info)
            print(traceback.format_exc())
            raise ValueError(advice_info)
        return self.pipeline
