from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score as roc_auc
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import make_pipeline
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive

from core.composer.chain import Chain
from core.composer.node import NodeGenerator
from core.models.model import *

train_file_path = "../cases/data/scoring/scoring_train.csv"
test_file_path = "../cases/data/scoring/scoring_test.csv"

train_data = InputData.from_csv(train_file_path)
test_data = InputData.from_csv(test_file_path)

training_features = train_data.features
testing_features = test_data.features
training_target = train_data.target
testing_target = test_data.target

# Average CV score on the training set was: 0.9375499999999999
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=BernoulliNB()),
    RandomForestClassifier()
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 1)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict_proba(testing_features)[:, 1]

roc_auc_value = roc_auc(y_true=testing_target,
                        y_score=results)

print(roc_auc_value)

chain = Chain()
node0 = NodeGenerator.primary_node(ModelTypesIdsEnum.datamodel)
node1 = NodeGenerator.primary_node(ModelTypesIdsEnum.bernb)
node2 = NodeGenerator.secondary_node(ModelTypesIdsEnum.rf)

node2.nodes_from.append(node0)
node2.nodes_from.append(node1)

chain.add_node(node0)
chain.add_node(node1)
chain.add_node(node2)

chain.fit(train_data)
results = chain.predict(test_data)

roc_auc_value = roc_auc(y_true=testing_target,
                        y_score=results.predict)
print(roc_auc_value)
