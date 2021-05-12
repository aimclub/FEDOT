import datetime
import os
import random

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score as roc_auc

from fedot.core.chains.chain import Chain
from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.data.data import InputData
from fedot.core.repository.tasks import Task, TaskTypesEnum
from cases.credit_scoring.credit_scoring_problem import get_scoring_data, calculate_validation_metric

random.seed(1)
np.random.seed(1)


def get_refinement_chain():
    """ Create a chain like this
                    (Main branch)
             /->     knn        -->       \
            /         |                    \
    scaling           |                     logit
            \         V                   /
             \-> class_decompose -> ridge
                (Side - regression branch)
       1        2                  3       4
    """

    # 1
    node_scaling = PrimaryNode('scaling')

    # 2
    node_knn = SecondaryNode('knn', nodes_from=[node_scaling])
    node_decompose = SecondaryNode('class_decompose', nodes_from=[node_knn, node_scaling])

    # 3
    node_ridge = SecondaryNode('ridge', nodes_from=[node_decompose])

    # 4
    node_logit = SecondaryNode('logit', nodes_from=[node_ridge, node_knn])
    chain = Chain(node_logit)
    return chain


def get_non_refinement_chain():
    """ Create a chain like this
             /->         knn           ->\
            /                             \
    scaling                                logit
            \                             /
             \->        logit          ->/
       1                  2                  3
    """
    # 1
    node_scaling = PrimaryNode('scaling')

    # 2
    node_knn = SecondaryNode('knn', nodes_from=[node_scaling])
    node_logit_1 = SecondaryNode('logit', nodes_from=[node_scaling])

    # 3
    node_logit_2 = SecondaryNode('logit', nodes_from=[node_logit_1, node_knn])
    chain = Chain(node_logit_2)
    return chain


def run_refinement_scoring_example(train_path, test_path, with_tuning=False):
    """ Function launch example with error modeling for classification task

    :param train_path: path to the csv file with training sample
    :param test_path: path to the csv file with test sample
    :param with_tuning: is it need to tune chains or not
    """

    task = Task(TaskTypesEnum.classification)
    train_dataset = InputData.from_csv(train_path, task=task)
    test_dataset = InputData.from_csv(test_path, task=task)

    # Get and fit chains
    # non_refinement_chain = get_non_refinement_chain()
    refinement_chain = get_refinement_chain()

    # non_refinement_chain.fit(train_dataset)
    refinement_chain.fit(train_dataset)

    # Check metrics
    # roc_auc_metric = calculate_validation_metric(non_refinement_chain, test_dataset)
    # print(f'Non decomposition chain ROC AUC: {roc_auc_metric:.2f}')

    roc_auc_metric, f1_metric = calculate_validation_metric(refinement_chain, test_dataset)
    print(f'With decomposition chain ROC AUC: {roc_auc_metric:2f}')

    if with_tuning:
        # non_refinement_chain.fine_tune_all_nodes(loss_function=roc_auc,
        #                                          loss_params=None,
        #                                          input_data=train_dataset,
        #                                          iterations=30)
        # roc_auc_metric = calculate_validation_metric(non_refinement_chain, test_dataset)
        # print(f'Non decomposition chain ROC AUC after tuning: {roc_auc_metric:.2f}')

        refinement_chain.fine_tune_all_nodes(loss_function=roc_auc,
                                             loss_params=None,
                                             input_data=train_dataset,
                                             iterations=30)
        roc_auc_metric = calculate_validation_metric(refinement_chain, test_dataset)
        print(f'With decomposition chain ROC AUC after tuning: {roc_auc_metric:.2f}')


if __name__ == '__main__':
    full_path_train, full_path_test = get_scoring_data()
    run_refinement_scoring_example(full_path_train, full_path_test, with_tuning=True)
