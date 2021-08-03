from fedot.core.pipelines.node import PrimaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.remote.remote_evaluator import RemoteEvaluator, RemoteTaskParams

pipeline_1 = Pipeline(PrimaryNode('linear'))
pipeline_2 = Pipeline(PrimaryNode('ridge'))

fitter = RemoteEvaluator(RemoteTaskParams(
    mode='remote',
    dataset_name='cholesterol',
    task_type='Task(TaskTypesEnum.regression)',
    train_data_idx=None,
    is_multi_modal=False,
    var_names=None,
    max_parallel=2,
    access_params={
        'FEDOT_LOGIN': 'your_login',
        'FEDOT_PASSWORD': 'your_password',
        'AUTH_SERVER': 'your_url',
        'CONTR_SERVER': 'your_url',
        'PROJECT_ID': 'your_project_id',
        'DATA_ID': 'your_data_id'
    }))

if fitter.use_remote:
    fitted_pipelines = fitter.compute_pipelines([pipeline_1, pipeline_2])
    print(fitted_pipelines[0].is_fitted)
    print(fitted_pipelines[1].is_fitted)
