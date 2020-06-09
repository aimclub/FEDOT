import numpy as np
from sklearn.metrics import roc_auc_score as roc_auc

from benchmark.benchmark_utils import get_scoring_case_data_paths
from core.composer.chain import Chain
from core.composer.node import NodeGenerator
from core.models.data import InputData
from core.repository.model_types_repository import ModelTypesIdsEnum


def chain_tuning(nodes_to_tune: str, chain: Chain, train_data: InputData, test_data: InputData, local_iter: int) -> (
        int, list):
    several_iter_scores_test = []

    if nodes_to_tune == 'primary':
        print('primary_node_tuning')

        for iteration in range(local_iter):
            print(f'current local iteration {iteration}')
            # Chain tuning
            chain.fine_tune_primary_nodes(train_data, iterations=50)

            # After tuning prediction
            chain.fit(train_data)
            after_tuning_predicted = chain.predict(test_data)

            # Metrics
            aft_tun_roc_auc = roc_auc(y_true=test_data.target, y_score=after_tuning_predicted.predict)
            several_iter_scores_test.append(aft_tun_roc_auc)

    elif nodes_to_tune == 'root':
        print('root_node_tuning')

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

        for iteration in range(local_iter):
            print(f'current local iteration {iteration}')

            # root node tuning
            chain.fine_tune_root_node(root_node_input_data, iterations=50)

            # After tuning prediction
            after_tun_root_node_predicted = chain.predict(test_data)

            # Metrics
            aft_tun_roc_auc = roc_auc(y_true=test_data.target, y_score=after_tun_root_node_predicted.predict)
            several_iter_scores_test.append(aft_tun_roc_auc)
    else:
        raise ValueError(f'Invalid type of nodes. Nodes must be primary or root')

    return np.mean(several_iter_scores_test), several_iter_scores_test


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

    # Before tuning prediction
    chain.fit(train_data, use_cache=False)
    before_tuning_predicted = chain.predict(test_data)
    bfr_tun_roc_auc = roc_auc(y_true=test_data.target, y_score=before_tuning_predicted.predict)

    local_iter = 5
    # Chain tuning
    aft_tun_roc_auc, several_iter_scores_test = chain_tuning(nodes_to_tune='primary', chain=chain,
                                                             train_data=train_data,
                                                             test_data=test_data, local_iter=local_iter)

    print(f'Several test scores {several_iter_scores_test}')
    print(f'Mean test score over {local_iter} iterations: {aft_tun_roc_auc}')
    print(round(bfr_tun_roc_auc, 3))
    print(round(float(aft_tun_roc_auc), 3))
