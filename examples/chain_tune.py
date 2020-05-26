from sklearn.metrics import roc_auc_score as roc_auc

from core.composer.chain import Chain
from core.composer.node import NodeGenerator
from core.models.model import *
from benchmark.benchmark_utils import get_scoring_case_data_paths
import copy

train_file_path, test_file_path = get_scoring_case_data_paths()

train_data = InputData.from_csv(train_file_path)
test_data = InputData.from_csv(test_file_path)

training_features = train_data.features
testing_features = test_data.features
training_target = train_data.target
testing_target = test_data.target

chain = Chain()
node0 = NodeGenerator.primary_node(ModelTypesIdsEnum.xgboost)
node1 = NodeGenerator.primary_node(ModelTypesIdsEnum.knn)
node2 = NodeGenerator.secondary_node(ModelTypesIdsEnum.logit)

node2.nodes_from.append(node0)
node2.nodes_from.append(node1)

chain.add_node(node0)
chain.add_node(node1)
chain.add_node(node2)
chain_tuned = copy.deepcopy(chain)

chain.fit(train_data)
chain_tuned.fit(train_data)

chain_tuned.fine_tune_primary_nodes(test_data)
chain_tuned.fit(train_data)

results = chain.predict(test_data)
results_tuned = chain_tuned.predict(test_data)

roc_auc_value = roc_auc(y_true=testing_target,
                        y_score=results.predict)
roc_auc_value_tuned = roc_auc(y_true=testing_target,
                              y_score=results_tuned.predict)

print(roc_auc_value)
print(roc_auc_value_tuned)
