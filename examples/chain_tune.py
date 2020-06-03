import numpy as np
from sklearn.metrics import roc_auc_score as roc_auc

from benchmark.benchmark_utils import get_scoring_case_data_paths
from core.composer.chain import Chain
from core.composer.node import NodeGenerator
from core.models.data import InputData
from core.repository.model_types_repository import ModelTypesIdsEnum
from core.repository.tuner_types_repository import SklearnTunerTypeEnum


def scoring_chain_root_node_tune_correct(train_data: InputData, test_data: InputData, chain: Chain,
                                         iterations: int = 1):
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

    bfr_tun_roc_auc: float = 0.
    aft_tun_roc_auc: float = 0.
    several_iter_scores_test = []
    local_iter = iterations

    for iteration in range(local_iter):
        print(f'current local iteration {iteration}')

        # root node tuning
        chain.fine_tune_root_node(root_node_input_data, iterations=50, tuner_type=SklearnTunerTypeEnum.rand)

        after_tun_root_node_predicted = chain.predict(test_data)

        bfr_tun_roc_auc = roc_auc(y_true=test_data.target, y_score=before_tuning_predicted.predict)
        aft_tun_roc_auc = roc_auc(y_true=test_data.target, y_score=after_tun_root_node_predicted.predict)
        several_iter_scores_test.append(aft_tun_roc_auc)

    print(f'Several test scores {several_iter_scores_test}')
    print(f'Mean test score over {local_iter} iterations: {np.mean(several_iter_scores_test)}')
    print(f'before tuning test {bfr_tun_roc_auc}')
    print(f'after tuning test {aft_tun_roc_auc}', '\n')


def scoring_chain_primary_nodes_tune_correct(train_data: InputData, test_data: InputData, chain: Chain,
                                             iterations: int = 1):
    print('primary_node_tuning')
    # Before tuning prediction
    chain.fit(train_data, use_cache=False)
    before_tuning_predicted = chain.predict(test_data)

    bfr_tun_roc_auc: float = 0.
    aft_tun_roc_auc: float = 0.

    several_iter_scores_test = []
    local_iter = iterations
    for iteration in range(local_iter):
        print(f'current local iteration {iteration}')
        # Chain tuning
        chain.fine_tune_primary_nodes(train_data, iterations=50, tuner_type=SklearnTunerTypeEnum.rand)

        # After tuning prediction
        chain.fit(train_data)
        after_tuning_predicted = chain.predict(test_data)

        # Metrics
        bfr_tun_roc_auc = roc_auc(y_true=test_data.target, y_score=before_tuning_predicted.predict)
        aft_tun_roc_auc = roc_auc(y_true=test_data.target, y_score=after_tuning_predicted.predict)
        several_iter_scores_test.append(aft_tun_roc_auc)

    print(f'Several test scores {several_iter_scores_test}')
    print(f'Mean test score over {local_iter} iterations: {np.mean(several_iter_scores_test)}')
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
    scoring_chain_root_node_tune_correct(train_data, test_data, chain, iterations=5)
