from fedot.core.pipelines.node import PrimaryNode
from fedot.core.pipelines.pipeline import Pipeline
from remote.remote_fit import ComputationalSetup

pipeline_1 = Pipeline(PrimaryNode('linear'))
pipeline_2 = Pipeline(PrimaryNode('ridge'))

ComputationalSetup.remote_eval_params = {
    'mode': 'remote',
    'dataset_name': 'cholesterol',
    'task_type': 'Task(TaskTypesEnum.regression)'
}

fitter = ComputationalSetup()
if fitter.is_remote:
    fitted_pipelines = fitter.fit([pipeline_1, pipeline_2])
    print(fitted_pipelines[0].is_fitted)
    print(fitted_pipelines[1].is_fitted)
