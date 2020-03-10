import numpy as np

from core.models.data import InputData
from core.models.evaluation import SkLearnEvaluationStrategy
from core.models.model import train_test_data_setup
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

model = eval_strategy.fit(model=ModelTypesIdsEnum.xgboost, train_data=train)

predicted = eval_strategy.predict(trained_model=model, predict_data=test)

print(test)
print(predicted)
