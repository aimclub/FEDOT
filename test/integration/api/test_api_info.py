from fedot.api.help import operations_for_task, print_data_operations_info, print_models_info


def test_api_help_correct():
    regression_operations = operations_for_task(task_name='regression')
    regression_models = regression_operations.get('model')

    classification_operations = operations_for_task(task_name='classification')
    classification_models = classification_operations.get('model')

    assert 'ridge' in regression_models
    assert 'logit' in classification_models


def test_api_print_info_correct():
    task_types = ['regression', 'classification', 'clustering', 'ts_forecasting']
    for task_name in task_types:
        print_models_info(task_name=task_name)
        print_data_operations_info(task_name=task_name)
