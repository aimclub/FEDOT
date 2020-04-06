from benchmark.benchmark_utils import get_scoring_case_data_paths
from benchmark.executor import CaseExecutor

if __name__ == '__main__':
    train_file, test_file = get_scoring_case_data_paths()
    CaseExecutor(train_file=train_file, test_file=test_file, case='scoring').execute()
