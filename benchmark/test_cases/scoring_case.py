from benchmark.benchmark_utils import get_scoring_case_data_paths, save_metrics_result_file
from benchmark.executor import CaseExecutor

if __name__ == '__main__':
    train_file, test_file = get_scoring_case_data_paths()
    metrics = CaseExecutor(train_file=train_file,
                           test_file=test_file,
                           is_classification=True,
                           case='scoring').execute()
    save_metrics_result_file(metrics, file_name='scoring_metrics')
