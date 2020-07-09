import os

import joblib

from benchmark.benchmark_utils import get_models_hyperparameters
from core.models.data import InputData
from core.models.evaluation.automl_eval import fit_tpot, predict_tpot_class, predict_tpot_reg
from core.repository.tasks import Task, TaskTypesEnum


def run_tpot(params: 'ExecutionParams'):
    train_file_path = params.train_file
    test_file_path = params.test_file
    case_label = params.case_label
    task = params.task

    models_hyperparameters = get_models_hyperparameters()['TPOT']
    generations = models_hyperparameters['GENERATIONS']
    population_size = models_hyperparameters['POPULATION_SIZE']

    result_model_filename = f'{case_label}_g{generations}' \
                            f'_p{population_size}_{task.name}.pkl'
    current_file_path = str(os.path.dirname(__file__))
    result_file_path = os.path.join(current_file_path, result_model_filename)

    train_data = InputData.from_csv(train_file_path, task=Task(task))

    if result_model_filename not in os.listdir(current_file_path):
        # TODO change hyperparameters to actual from variable
        model = fit_tpot(train_data, models_hyperparameters['MAX_RUNTIME_MINS'])

        model.export(output_file_name=f'{result_model_filename[:-4]}_pipeline.py')

        # sklearn pipeline object
        fitted_model_config = model.fitted_pipeline_
        joblib.dump(fitted_model_config, result_file_path, compress=1)

    imported_model = joblib.load(result_file_path)

    predict_data = InputData.from_csv(test_file_path, task=Task(task))
    true_target = predict_data.target
    if task == TaskTypesEnum.regression:
        predicted = predict_tpot_reg(imported_model, predict_data)
    elif task == TaskTypesEnum.classification:
        predicted = predict_tpot_class(imported_model, predict_data)
    else:
        print('Incorrect type of ml task')
        raise NotImplementedError()

    print(f'BEST_model: {imported_model}')

    return true_target, predicted
