import logging
import pathlib

from examples.simple.classification.classification_pipelines import classification_complex_pipeline
from examples.simple.pipeline_tune import get_case_train_test_data
from fedot.core.log import Log


def run_log_example(log_file):
    train_data, _ = get_case_train_test_data()

    # Use default_log if you do not have json config file for log
    log = Log(log_file=log_file, output_logging_level=logging.DEBUG).get_adapter(prefix=pathlib.Path(__file__).stem)

    log.info('start creating pipeline')
    pipeline = classification_complex_pipeline()

    log.info('start fitting pipeline')
    pipeline.fit(train_data)


if __name__ == '__main__':
    run_log_example(log_file='example_log.log')
