from fedot.core.pipelines.node import PrimaryNode
from fedot.core.pipelines.pipeline import Pipeline
from infrastructure.remote_fit import RemoteFitter

pipeline_1 = Pipeline(PrimaryNode('linear'))
pipeline_2 = Pipeline(PrimaryNode('ridge'))

RemoteFitter.remote_eval_params = {
    'use': True,
    'dataset_name': 'cholesterol',
    'task_type': 'Task(TaskTypesEnum.regression)'
}

fitter = RemoteFitter()
if fitter.is_use:
    fitted_pipelines = fitter.fit([pipeline_1, pipeline_2])
    print(fitted_pipelines[0].is_fitted)
    print(fitted_pipelines[1].is_fitted)
