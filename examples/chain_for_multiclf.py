from sklearn.metrics import roc_auc_score as roc_auc
from core.composer.chain import Chain
from core.composer.node import NodeGenerator
from core.models.model import *
from benchmark.benchmark_utils import get_multi_clf_data_paths

train_file_path, test_file_path = get_multi_clf_data_paths()

train_data = InputData.from_csv(train_file_path)
test_data = InputData.from_csv(test_file_path)
testing_target = test_data.target

chain = Chain()
node_first = NodeGenerator.primary_node(ModelTypesIdsEnum.xgboost)
node_second = NodeGenerator.primary_node(ModelTypesIdsEnum.lda)
node_third = NodeGenerator.secondary_node(ModelTypesIdsEnum.rf)

node_third.nodes_from.append(node_first)
node_third.nodes_from.append(node_second)

chain.add_node(node_first)
chain.add_node(node_second)
chain.add_node(node_third)

chain.fit(train_data)
results = chain.predict(test_data)

roc_auc_value = roc_auc(y_true=testing_target,
                        y_score=results.predict,
                        multi_class='ovo',
                        average='macro')
print(roc_auc_value)
