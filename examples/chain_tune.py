import numpy as np
from sklearn.metrics import roc_auc_score as roc_auc

from benchmark.benchmark_utils import get_scoring_case_data_paths
from core.composer.chain import Chain
from core.composer.node import NodeGenerator
from core.models.data import InputData
from core.repository.model_types_repository import ModelTypesIdsEnum


def scoring_chain_root_node_tune_correct(train_data: InputData, test_data: InputData, chain: Chain):
    print('root_node_tuning')
    # Before tuning prediction
    chain.fit(train_data, use_cache=False)
    before_tuning_predicted = chain.predict(test_data)

    # root node tuning preprocessing
    primary_pred = []
    for node in chain.nodes[:2]:
        pred = node.predict(train_data)
        primary_pred.append(pred.predict)

    new_features = np.array(primary_pred[::-1]).T

    root_node_input_data = InputData(features=new_features,
                                     target=train_data.target,
                                     idx=train_data.idx,
                                     task_type=train_data.task_type)

    # root node tuning
    chain.fine_tune_root_node(root_node_input_data, iterations=20)

    # chain.fit(train_data)
    after_tun_root_node_predicted = chain.predict(test_data)

    bfr_tun_roc_auc = roc_auc(y_true=test_data.target, y_score=before_tuning_predicted.predict)
    aft_tun_roc_auc = roc_auc(y_true=test_data.target, y_score=after_tun_root_node_predicted.predict)

    print(bfr_tun_roc_auc)
    print(aft_tun_roc_auc)


def scoring_chain_primary_nodes_tune_correct(train_data: InputData, test_data: InputData, chain: Chain):
    print('primary_node_tuning')
    # Before tuning prediction
    chain.fit(train_data, use_cache=False)
    before_tuning_predicted = chain.predict(test_data)

    # Chain tuning
    chain.fine_tune_primary_nodes(train_data, iterations=50)

    # After tuning prediction
    chain.fit(train_data)
    after_tuning_predicted = chain.predict(test_data)

    # Metrics
    bfr_tun_roc_auc = roc_auc(y_true=test_data.target, y_score=before_tuning_predicted.predict)
    aft_tun_roc_auc = roc_auc(y_true=test_data.target, y_score=after_tuning_predicted.predict)

    print(bfr_tun_roc_auc)
    print(aft_tun_roc_auc)


if __name__ == '__main__':
    train_file_path, test_file_path = get_scoring_case_data_paths()

    train_data = InputData.from_csv(train_file_path)
    test_data = InputData.from_csv(test_file_path)

    # Chain composition
    first = NodeGenerator.primary_node(model_type=ModelTypesIdsEnum.xgboost)
    second = NodeGenerator.primary_node(model_type=ModelTypesIdsEnum.knn)
    final = NodeGenerator.secondary_node(model_type=ModelTypesIdsEnum.logit,
                                         nodes_from=[first, second])

    chain = Chain()
    for node in [first, second, final]:
        chain.add_node(node)

    scoring_chain_primary_nodes_tune_correct(train_data, test_data, chain)
    scoring_chain_root_node_tune_correct(train_data, test_data, chain)
