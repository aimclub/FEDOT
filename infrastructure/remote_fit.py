import os
import time
from typing import List

import numpy as np

from fedot.core.pipelines.validation import validate
from infrastructure.datamall.models_controller.computations import Client


class RemoteFitter:
    remote_eval_params = {
        'use': False,
        'dataset_name': '',  # name of the dataset for composer evaluation
        'task_type': '',  # name of the modelling task for dataset
        'max_parallel': 10
    }

    @property
    def is_use(self):
        return RemoteFitter.remote_eval_params['use']

    def fit(self, pipelines: List['Pipeline']):
        num_parts = np.floor(len(pipelines) / RemoteFitter.remote_eval_params['max_parallel'])
        pipelines_parts = [x.tolist() for x in np.array_split(pipelines, num_parts)]
        data_id = int(os.environ['DATA_ID'])
        pid = int(os.environ['PROJECT_ID'])

        final_pipelines = []
        for pipelines_part in pipelines_parts:

            remote_eval_params = RemoteFitter.remote_eval_params
            client = Client(
                authorization_server=os.environ['AUTH_SERVER'],
                controller_server=os.environ['CONTR_SERVER']
            )

            client.login(login=os.environ['FEDOT_LOGIN'],
                         password=os.environ['FEDOT_PASSWORD'])

            client.create_execution_group(project_id=pid)
            response = client.get_execution_groups(project_id=pid)
            client.set_group_token(project_id=pid, group_id=response[-1]['id'])

            dataset_name = remote_eval_params['dataset_name']
            task_type = remote_eval_params['task_type']

            for pipeline in pipelines_part:
                try:
                    validate(pipeline)
                except ValueError:
                    pipeline.execution_id = None
                    continue

                pipeline_json = pipeline.save('tmp.json').replace('\n', '')

                config = f"""[DEFAULT]
                pipeline_description = {pipeline_json}
                train_data = input_data_dir/data/{data_id}/{dataset_name}/{dataset_name}_comp.csv
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

                execution_id = client.get_executions()[-1]['id']
                pipeline.execution_id = execution_id

            statuses = ['']
            all_executions = client.get_executions()
            print(all_executions)
            while any(s not in ['Succeeded', 'Failed', 'Timeout', 'Interrupted'] for s in statuses):
                executions = client.get_executions()
                statuses = [execution['status'] for execution in executions]
                print([f"{execution['id']}={execution['status']};" for execution in executions])
                time.sleep(5)

            print('Success')

            for p_id, pipeline in enumerate(pipelines_part):
                if pipeline.execution_id:
                    client.download_result(
                        execution_id=pipeline.execution_id,
                        path=f'./remote_fit_results',
                        unpack=True
                    )

                    try:
                        results_path_out = f'./remote_fit_results/execution-{pipeline.execution_id}/out'
                        results_folder = os.listdir(results_path_out)[0]
                        pipeline.load(os.path.join(results_path_out, results_folder, 'fitted_pipeline.json'))
                    except Exception as ex:
                        print(p_id, ex)
            final_pipelines.extend(final_pipelines)

        return final_pipelines
