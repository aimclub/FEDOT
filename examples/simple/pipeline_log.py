import os

import pathlib

from examples.simple.classification.classification_pipelines import classification_complex_pipeline
from examples.simple.pipeline_tune import get_case_train_test_data, get_scoring_case_data_paths
from fedot.core.log import Log


def run_log_example(log_file):
    train_file_path, test_file_path = get_scoring_case_data_paths()

    current_path = os.path.dirname(__name__)
    train_data, test_data = get_case_train_test_data()

    # Use default_log if you do not have json config file for log
    log = Log(logger_name='logger', log_file=log_file).get_adapter(prefix=pathlib.Path(__file__).stem)

    log.info('start creating pipeline')
    pipeline = classification_complex_pipeline()

    log.info('start fitting pipeline')
    pipeline.fit(train_data)


if __name__ == '__main__':
    run_log_example(log_file='example_log.log')
