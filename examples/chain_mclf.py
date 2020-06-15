from sklearn.metrics import roc_auc_score as roc_auc
# from sklearn.metrics import precision_score
from core.composer.chain import Chain
from core.composer.node import NodeGenerator
from core.models.model import *
from benchmark.benchmark_utils import get_multi_clf_data_paths

train_file_path, test_file_path = get_multi_clf_data_paths()

train_data = InputData.from_csv(train_file_path)
test_data = InputData.from_csv(test_file_path)

training_features = train_data.features
testing_features = test_data.features
training_target = train_data.target
testing_target = test_data.target

chain = Chain()
node0 = NodeGenerator.primary_node(ModelTypesIdsEnum.xgboost)
node1 = NodeGenerator.primary_node(ModelTypesIdsEnum.lda)
node2 = NodeGenerator.secondary_node(ModelTypesIdsEnum.rf)

node2.nodes_from.append(node0)
node2.nodes_from.append(node1)

chain.add_node(node0)
chain.add_node(node1)
chain.add_node(node2)

chain.fit(train_data)
results1 = chain.predict(train_data)
results = chain.predict(test_data)

# precision_score = precision_score(testing_target,
# results.predict,
# average='macro')

roc_auc_value = roc_auc(y_true=testing_target,
                        y_score=results.predict,
                        multi_class='ovo',
                        average='macro')
print(roc_auc_value)

# print(precision_score)
