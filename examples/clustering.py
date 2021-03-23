import random
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, make_blobs

from fedot.api.main import Fedot
from fedot.core.chains.chain import Chain
from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum

random.seed(1)
np.random.seed(1)


def create_simple_clustering_example(data_size=200):
    num_features = 2
    num_clusters = 3

    predictors, response = make_blobs(data_size, num_features, centers=num_clusters, random_state=1)

    return InputData(features=predictors, target=response, idx=np.arange(0, len(predictors)),
                     task=Task(TaskTypesEnum.clustering),
                     data_type=DataTypesEnum.table)


def create_iris_clustering_example(data_size=-1):
    predictors, response = load_iris(return_X_y=True)
    if data_size < 0:
        data_size = predictors.shape[0]
    return InputData(features=predictors[:data_size, :], target=response[:data_size],
                     idx=np.arange(0, data_size),
                     task=Task(TaskTypesEnum.clustering),
                     data_type=DataTypesEnum.table)


def run_clustering_example(is_fast=False):
    opt_time_sec = 30
    iris_size = -1

    if is_fast:
        opt_time_sec = 1
        iris_size = 10

    # ensemble clustering example
    data = pd.read_csv('./data/heart.csv', sep=',')
    data_train = deepcopy(data)
    del data_train['target']

    baseline_clustering_fedot = Fedot(problem='clustering')
    baseline_clustering_fedot.fit(data_train, predefined_model='aglo_clust')
    baseline_clustering_fedot.predict(data_train)
    prediction_basic = baseline_clustering_fedot.get_metrics(target=data['target'])

    auto_clustering_fedot = Fedot(problem='clustering', verbose_level=4,
                                  learning_time=opt_time_sec, seed=42)

    chain = Chain(SecondaryNode('consensus_ensembler',
                                nodes_from=[PrimaryNode('meanshift_clust'),
                                            PrimaryNode('kmeans'),
                                            PrimaryNode('aglo_clust')]))
    composite_model = auto_clustering_fedot.fit(data_train, predefined_model=chain)
    auto_clustering_fedot.predict(data_train)
    prediction_composite = auto_clustering_fedot.get_metrics(target=data['target'])

    if not is_fast:
        composite_model.show()

    print(f'score for basic model {prediction_basic}')
    print(f'score for composite model {prediction_composite}')

    return prediction_basic, prediction_composite


if __name__ == '__main__':
    run_clustering_example()
