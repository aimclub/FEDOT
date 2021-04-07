import os

from examples.chain_tune import get_case_train_test_data, get_scoring_case_data_paths
from fedot.core.chains.chain import Chain
from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.log import default_log


def get_simple_chain(log):
    first = PrimaryNode(operation_type='xgboost', log=log)
    second = PrimaryNode(operation_type='knn', log=log)
    final = SecondaryNode(operation_type='logit',
                          nodes_from=[first, second],
                          log=log)

    # if you do not pass the log object, Chain will create default log.log file placed in core
    chain = Chain(final, log=log)

    return chain


def run_log_example(log_file_name):
    train_file_path, test_file_path = get_scoring_case_data_paths()

    current_path = os.path.dirname(__name__)
    train_data, test_data = get_case_train_test_data()

    # Use default_log if you do not have json config file for log
    log = default_log('chain_log',
                      log_file=os.path.join(current_path, log_file_name))

    log.info('start creating chain')
    chain = get_simple_chain(log=log)

    log.info('start fitting chain')
    chain.fit(train_data, use_cache=False)


if __name__ == '__main__':
    run_log_example(log_file_name='example_log.log')
