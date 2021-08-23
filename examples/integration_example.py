import os
import time

from infrastructure.datamall.models_controller.computations import Client

# pipeline = Pipeline(PrimaryNode('linear'))
# pipeline.save('exported_pipeline.json')


client = Client(
    authorization_server="http://10.32.0.51:30880/b",
    controller_server="http://10.32.0.51:30880/models-controller"
)

client.login(login=os.environ['FEDOT_LOGIN'],
             password=os.environ['FEDOT_PASSWORD'])

pid = 83
client.create_execution_group(project_id=pid)
response = client.get_execution_groups(project_id=pid)
print(response)
client.set_group_token(project_id=83, group_id=response[-1]['id'])
result = client.get_executions()

CONFIG_EXAMPLE = b"""[DEFAULT]
pipeline_file_path = input_data_dir/data/55/exported_pipeline.json
train_data = input_data_dir/data/55/cholesterol/cholesterol.csv
task = Task(TaskTypesEnum.regression)
output_path = output_data_dir/fitted_pipeline
[OPTIONAL]
"""

client.create_execution(
    container_input_path="/home/FEDOT/input_data_dir",
    container_output_path="/home/FEDOT/output_data_dir",
    container_config_path="/home/FEDOT/.config",
    container_image="fedot:dm",
    timeout=60,
    config=CONFIG_EXAMPLE
)

status = ''

while status != 'Succeeded':
    executions = client.get_executions()
    print(executions)
    execution = executions[-1]
    status = execution['status']
    print(status)
    time.sleep(0.5)

client.download_result(
    execution_id=execution['id'],
    path="downloads",
    unpack=True
)

print(result)
