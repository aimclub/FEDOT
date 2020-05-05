import os

import joblib

from benchmark.benchmark_utils import get_scoring_case_data_paths, get_models_hyperparameters
from core.models.data import InputData
from core.models.evaluation.automl_eval import fit_tpot, predict_tpot
from core.repository.task_types import MachineLearningTasksEnum


def run_tpot(train_file_path: str, test_file_path: str, task: MachineLearningTasksEnum,
             case_name='tpot_default'):
    models_hyperparameters = get_models_hyperparameters()['TPOT']
    generations = models_hyperparameters['GENERATIONS']
    population_size = models_hyperparameters['POPULATION_SIZE']

    result_model_filename = f'{case_name}_g{generations}' \
                            f'_p{population_size}_{task.name}.pkl'
    current_file_path = str(os.path.dirname(__file__))
    result_file_path = os.path.join(current_file_path, result_model_filename)

    train_data = InputData.from_csv(train_file_path)

    if result_model_filename not in os.listdir(current_file_path):
        model = fit_tpot(train_data)

        model.export(output_file_name=f'{result_model_filename[:-4]}_pipeline.py')

        # sklearn pipeline object
        fitted_model_config = model.fitted_pipeline_
        joblib.dump(fitted_model_config, result_file_path, compress=1)

    imported_model = joblib.load(result_file_path)

    predict_data = InputData.from_csv(test_file_path)
    true_target = predict_data.target
    predicted = predict_tpot(imported_model, predict_data)

    print(f'BEST_model: {imported_model}')

    return true_target, predicted


if __name__ == '__main__':
    train_data_path, test_data_path = get_scoring_case_data_paths()

    run_tpot(train_data_path, test_data_path, task=MachineLearningTasksEnum.classification)
