import numpy as np
from sklearn.metrics import roc_auc_score as roc_auc

from core.chain_validation import validate
from core.models.data import InputData, train_test_data_setup
from core.models.model import sklearn_model_by_type
from core.repository.model_types_repository import ModelTypesIdsEnum
from experiments.chain_template import (
    chain_template_balanced_tree, show_chain_template,
    real_chain, fit_template
)
from experiments.generate_data import synthetic_dataset

if __name__ == '__main__':
    model_types = [ModelTypesIdsEnum.logit, ModelTypesIdsEnum.xgboost, ModelTypesIdsEnum.knn]
    samples, features_amount, classes = 1000, 10, 2

    # chain = chain_template_random(model_types=model_types, depth=3, models_per_level=2,
    #                               samples=samples, features=features_amount)
    chain = chain_template_balanced_tree(model_types=model_types, depth=4, models_per_level=[8, 4, 2, 1],
                                         samples=samples, features=features_amount)
    show_chain_template(chain)
    fit_template(chain_template=chain, classes=classes)
    real = real_chain(chain)
    validate(real)
    features, target = synthetic_dataset(samples_amount=samples,
                                         features_amount=features_amount,
                                         classes_amount=classes)
    target = np.expand_dims(target, axis=1)
    data_test = InputData(idx=np.arange(0, samples),
                          features=features, target=target)
    predictions = real.predict(input_data=data_test)

    logit = sklearn_model_by_type(model_type=ModelTypesIdsEnum.logit)
    train, test = train_test_data_setup(data_test)
    fitted_model, predict_train = logit.fit(data=train)
    roc_score = roc_auc(y_true=train.target,
                        y_score=predict_train)
    print(f'Roc train: {roc_score}')
    predict_test = logit.predict(fitted_model=fitted_model, data=test)
    roc_score = roc_auc(y_true=test.target,
                        y_score=predict_test)
    print(f'Roc test: {roc_score}')
