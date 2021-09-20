from fedot.core.pipelines.node import PrimaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.remote.remote_fit import ComputationalSetup

pipeline_1 = Pipeline(PrimaryNode('linear'))
pipeline_2 = Pipeline(PrimaryNode('ridge'))

ComputationalSetup.remote_eval_params = {
    'mode': 'remote',
    'dataset_name': 'cholesterol',
    'task_type': 'Task(TaskTypesEnum.regression)',
    'train_data_idx': [],
    'is_multi_modal': True,
    'var_names': [],
    'max_parallel': 2,
    'access_params': {
        'FEDOT_LOGIN': 'test1234',
        'FEDOT_PASSWORD': 'test1234',
        'AUTH_SERVER': 'http://10.32.0.51:30880/b',
        'CONTR_SERVER': 'http://10.32.0.51:30880/models-controller',
        'PROJECT_ID': '82',
        'DATA_ID': '61'
    }
}

fitter = ComputationalSetup()
if fitter.is_remote:
    fitted_pipelines = fitter.fit([pipeline_1, pipeline_2])
    print(fitted_pipelines[0].is_fitted)
    print(fitted_pipelines[1].is_fitted)
