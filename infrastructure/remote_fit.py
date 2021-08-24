import os
import time

from infrastructure.datamall.models_controller.computations import Client


def remote_pipeline_fit(pipeline: 'Pipeline', remote_eval_params: dict):
    client = Client(
        authorization_server=os.environ['AUTH_SERVER'],
        controller_server=os.environ['CONTR_SERVER']
    )

    client.login(login=os.environ['FEDOT_LOGIN'],
                 password=os.environ['FEDOT_PASSWORD'])

    pid = int(os.environ['PROJECT_ID'])
    client.create_execution_group(project_id=pid)
    response = client.get_execution_groups(project_id=pid)
    client.set_group_token(project_id=pid, group_id=response[-1]['id'])
    pipeline_json = pipeline.save('tmp.json').replace('\n', '')

    dataset_name = remote_eval_params['dataset_name']
    task_type = remote_eval_params['task_type']

    config = f"""[DEFAULT]
    pipeline_description = {pipeline_json}
    train_data = input_data_dir/data/55/{dataset_name}/{dataset_name}.csv
    task = {task_type}
    output_path = output_data_dir/fitted_pipeline
    [OPTIONAL]
    """.encode('utf-8')

    client.create_execution(
        container_input_path="/home/FEDOT/input_data_dir",
        container_output_path="/home/FEDOT/output_data_dir",
        container_config_path="/home/FEDOT/.config",
        container_image="fedot:dm-2",
        timeout=60,
        config=config
    )

    status = ''

    while status != 'Succeeded' and status != 'Failed':
        executions = client.get_executions()
        execution = executions[-1]
        status = execution['status']
        time.sleep(0.5)

    ex_id = execution['id']
    client.download_result(
        execution_id=ex_id,
        path=f'./remote_fit_results',
        unpack=True
    )

    results_path_out = f'./remote_fit_results/execution-{ex_id}/out'
    results_folder = os.listdir(results_path_out)[0]

    pipeline.load(os.path.join(results_path_out, results_folder, 'fitted_pipeline.json'))

    return pipeline
