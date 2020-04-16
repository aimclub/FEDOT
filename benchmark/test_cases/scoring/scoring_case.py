from benchmark.benchmark_utils import get_scoring_case_data_paths, save_metrics_result_file, get_models_hyperparameters
from benchmark.executor import CaseExecutor


if __name__ == '__main__':
    train_file, test_file = get_scoring_case_data_paths()
    config_models_data = get_models_hyperparameters()

    result_metrics = CaseExecutor(train_file=train_file,
                                  test_file=test_file,
                                  is_classification=True,
                                  label='default',
                                  case='scoring', hyperparameters=config_models_data).execute()

    result_metrics['hyperparameters'] = config_models_data

    save_metrics_result_file(result_metrics, file_name='scoring_metrics')
