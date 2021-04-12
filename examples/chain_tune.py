import numpy as np
from sklearn.metrics import roc_auc_score as roc_auc

from cases.data.data_utils import get_scoring_case_data_paths
from fedot.core.chains.chain import Chain
from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.chains.tuning.unified import ChainTuner
from fedot.core.data.data import InputData


def get_case_train_test_data():
    """ Function for getting data for train and validation """
    train_file_path, test_file_path = get_scoring_case_data_paths()

    train_data = InputData.from_csv(train_file_path)
    test_data = InputData.from_csv(test_file_path)
    return train_data, test_data


def get_simple_chain():
    """ Function return simple chain with the following structure:
    xgboost \
             -> logit
      knn   |
    """
    first = PrimaryNode(operation_type='xgboost')
    second = PrimaryNode(operation_type='knn')
    final = SecondaryNode(operation_type='logit',
                          nodes_from=[first, second])

    chain = Chain(final)

    return chain


def chain_tuning(chain: Chain, train_data: InputData,
                 test_data: InputData, local_iter: int,
                 tuner_iter_num: int = 30) -> (float, list):
    """ Function for tuning chain with ChainTuner

    :param chain: chain to tune
    :param train_data: InputData for train
    :param test_data: InputData for validation
    :param local_iter: amount of tuner launches
    :param tuner_iter_num: amount of iterations, which tuner will perform

    :return mean_metric: mean value of ROC AUC metric
    :return several_iter_scores_test: list with metrics
    """
    several_iter_scores_test = []
    for iteration in range(local_iter):
        print(f'current local iteration {iteration}')

        # Chain tuning
        chain_tuner = ChainTuner(chain=chain,
                                 task=train_data.task,
                                 iterations=tuner_iter_num)
        tuned_chain = chain_tuner.tune_chain(input_data=train_data,
                                             loss_function=roc_auc)

        # After tuning prediction
        tuned_chain.fit(train_data)
        after_tuning_predicted = tuned_chain.predict(test_data)

        # Metrics
        aft_tun_roc_auc = roc_auc(y_true=test_data.target,
                                  y_score=after_tuning_predicted.predict)
        several_iter_scores_test.append(aft_tun_roc_auc)

    mean_metric = float(np.mean(several_iter_scores_test))
    return mean_metric, several_iter_scores_test


if __name__ == '__main__':
    train_data, test_data = get_case_train_test_data()

    # Chain composition
    chain = get_simple_chain()

    # Before tuning prediction
    chain.fit(train_data, use_cache=False)
    before_tuning_predicted = chain.predict(test_data)
    bfr_tun_roc_auc = roc_auc(y_true=test_data.target,
                              y_score=before_tuning_predicted.predict)

    local_iter = 5
    # Chain tuning
    after_tune_roc_auc, several_iter_scores_test = chain_tuning(chain=chain,
                                                                train_data=train_data,
                                                                test_data=test_data,
                                                                local_iter=local_iter)

    print(f'Several test scores {several_iter_scores_test}')
    print(f'Mean test score over {local_iter} iterations: {after_tune_roc_auc}')
    print(round(bfr_tun_roc_auc, 3))
    print(round(after_tune_roc_auc, 3))
