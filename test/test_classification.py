from sklearn.datasets import load_iris
from sklearn.metrics import roc_auc_score as roc_auc

from core.composer.chain import Chain
from core.composer.node import NodeGenerator
from core.models.data import InputData, train_test_data_setup
from core.models.model import *
from core.repository.task_types import MachineLearningTasksEnum


def compose_chain() -> Chain:
    chain = Chain()
    node_first = NodeGenerator.primary_node(ModelTypesIdsEnum.xgboost)
    node_second = NodeGenerator.primary_node(ModelTypesIdsEnum.lda)
    node_third = NodeGenerator.secondary_node(ModelTypesIdsEnum.rf)

    node_third.nodes_from.append(node_first)
    node_third.nodes_from.append(node_second)

    chain.add_node(node_first)
    chain.add_node(node_second)
    chain.add_node(node_third)

    return chain


def get_iris_data() -> InputData:
    synthetic_data = load_iris()
    input_data = InputData(idx=np.arange(0, len(synthetic_data.target)),
                           features=synthetic_data.data,
                           target=synthetic_data.target,
                           task_type=MachineLearningTasksEnum.classification)
    return input_data


def test_multiclassification_chain_fit_correct():
    data = get_iris_data()
    chain = compose_chain()
    train_data, test_data = train_test_data_setup(data, shuffle_flag=True)

    chain.fit(input_data=train_data)
    results = chain.predict(input_data=test_data)

    roc_auc_on_test = roc_auc(y_true=test_data.target,
                              y_score=results.predict,
                              multi_class='ovo',
                              average='macro')

    assert roc_auc_on_test > 0.95
