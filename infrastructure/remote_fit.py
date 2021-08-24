import os
import time
from typing import List

from infrastructure.datamall.models_controller.computations import Client


class RemoteFitter:
    remote_eval_params = {
        'use': False,
        'dataset_name': '',  # name of the dataset for composer evaluation
        'task_type': ''  # name of the modelling task for dataset
    }

    @property
    def is_use(self):
        return RemoteFitter.remote_eval_params['use']

    def fit(self, pipelines: List['Pipeline']):
        remote_eval_params = RemoteFitter.remote_eval_params
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

        dataset_name = remote_eval_params['dataset_name']
        task_type = remote_eval_params['task_type']

        for pipeline in pipelines:
            pipeline_json = pipeline.save('tmp.json').replace('\n', '')

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

        statuses = ['']
        all_executions = client.get_executions()
        print(all_executions)
        while any(s != 'Succeeded' and s != 'Failed' for s in statuses):
            executions = client.get_executions()
            statuses = [execution['status'] for execution in executions]
            print([f"{execution['id']}={execution['status']};" for execution in executions])
            time.sleep(5)

        print('Success')

        for p_id, pipeline in enumerate(pipelines):
            ex_id = all_executions[p_id]['id']
            client.download_result(
                execution_id=ex_id,
                path=f'./remote_fit_results',
                unpack=True
            )

            results_path_out = f'./remote_fit_results/execution-{ex_id}/out'
            results_folder = os.listdir(results_path_out)[0]

            pipeline.load(os.path.join(results_path_out, results_folder, 'fitted_pipeline.json'))

        return pipelines
