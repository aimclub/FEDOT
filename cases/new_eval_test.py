import numpy as np
from sklearn.metrics import roc_auc_score as roc_auc

from core.composer.node import NodeGenerator
from core.models.data import InputData
from core.models.evaluation import SkLearnEvaluationStrategy
from core.models.model import train_test_data_setup, Model, preprocess
from core.repository.dataset_types import NumericalDataTypesEnum
from core.repository.model_types_repository import ModelTypesIdsEnum


def classification_dataset():
    samples = 1000
    x = 10.0 * np.random.rand(samples, ) - 5.0
    x = np.expand_dims(x, axis=1)
    y = 1.0 / (1.0 + np.exp(np.power(x, -1.0)))
    threshold = 0.5
    classes = np.array([0.0 if val <= threshold else 1.0 for val in y])
    classes = np.expand_dims(classes, axis=1)
    data = InputData(features=x, target=classes, idx=np.arange(0, len(x)))

    return data


data = classification_dataset()
train, test = train_test_data_setup(data=data)
eval_strategy = SkLearnEvaluationStrategy()

fitted_xgb = eval_strategy.fit(model_type=ModelTypesIdsEnum.xgboost, train_data=train)

predicted = eval_strategy.predict(trained_model=fitted_xgb, predict_data=test)

new_predicted = eval_strategy.predict(trained_model=fitted_xgb, predict_data=test)

assert np.array_equal(predicted, new_predicted)

model = Model(model_type=ModelTypesIdsEnum.xgboost,
              input_type=NumericalDataTypesEnum.table,
              output_type=NumericalDataTypesEnum.vector,
              eval_strategy=eval_strategy)

predicted_by_model = model.fit(data=data)

print(np.array_equal(new_predicted, predicted_by_model))

# (y1, y2) -> y
last_node = NodeGenerator.secondary_node(model_type=ModelTypesIdsEnum.knn)
last_node.nodes_from = []

node_first = NodeGenerator.primary_node(model_type=ModelTypesIdsEnum.logit)
node_second = NodeGenerator.primary_node(model_type=ModelTypesIdsEnum.xgboost)

last_node.nodes_from = [node_first, node_second]

data.features = preprocess(data.features)
train_data, test_data = train_test_data_setup(data=data)

predict_train = last_node.fit(input_data=train_data)
predict_test = last_node.predict(input_data=test_data)

roc_on_train = roc_auc(y_true=train_data.target,
                       y_score=predict_train.predict)
roc_on_test = roc_auc(y_true=test_data.target,
                      y_score=predict_test.predict)

print(roc_on_train)
print(roc_on_test)
