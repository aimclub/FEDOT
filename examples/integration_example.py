from fedot.core.pipelines.node import PrimaryNode
from fedot.core.pipelines.pipeline import Pipeline
from infrastructure.remote_fit import remote_pipeline_fit

pipeline = Pipeline(PrimaryNode('linear'))

remote_eval_params = {
    'dataset_name': 'cholesterol',
    'task_type': 'Task(TaskTypesEnum.regression)'
}

fitted_pipeline = remote_pipeline_fit(pipeline, remote_eval_params)
print(fitted_pipeline.is_fitted)
