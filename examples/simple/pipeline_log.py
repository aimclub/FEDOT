import os

from examples.simple.classification.classification_pipelines import classification_complex_pipeline
from examples.simple.pipeline_tune import get_case_train_test_data, get_scoring_case_data_paths
from fedot.core.log import default_log


def run_log_example(log_file_name):
    train_file_path, test_file_path = get_scoring_case_data_paths()

    current_path = os.path.dirname(__name__)
    train_data, test_data = get_case_train_test_data()

    # Use default_log if you do not have json config file for log
    log = default_log('pipeline_log',
                      log_file=os.path.join(current_path, log_file_name))

    log.info('start creating pipeline')
    pipeline = classification_complex_pipeline(log=log)

    log.info('start fitting pipeline')
    pipeline.fit(train_data, use_fitted=False)


if __name__ == '__main__':
    run_log_example(log_file_name='example_log.log')
