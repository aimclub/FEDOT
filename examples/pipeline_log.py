import os

from examples.pipeline_tune import get_case_train_test_data, get_scoring_case_data_paths
from fedot.core.log import default_log
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline


def get_simple_pipeline(log):
    first = PrimaryNode(operation_type='xgboost', log=log)
    second = PrimaryNode(operation_type='knn', log=log)
    final = SecondaryNode(operation_type='logit',
                          nodes_from=[first, second],
                          log=log)

    # if you do not pass the log object, Pipeline will create default log.log file placed in core
    pipeline = Pipeline(final, log=log)

    return pipeline


def run_log_example(log_file_name):
    train_file_path, test_file_path = get_scoring_case_data_paths()

    current_path = os.path.dirname(__name__)
    train_data, test_data = get_case_train_test_data()

    # Use default_log if you do not have json config file for log
    log = default_log('pipeline_log',
                      log_file=os.path.join(current_path, log_file_name))

    log.info('start creating pipeline')
    pipeline = get_simple_pipeline(log=log)

    log.info('start fitting pipeline')
    pipeline.fit(train_data, use_fitted=False)


if __name__ == '__main__':
    run_log_example(log_file_name='example_log.log')
