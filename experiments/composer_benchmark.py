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


def to_labels(predictions):
    thr = 0.5
    labels = [0 if val <= thr else 1 for val in predictions]
    labels = np.expand_dims(np.array(labels), axis=1)
    return labels


def robust_test():
    np.random.seed(42)
    model_types = [ModelTypesIdsEnum.logit, ModelTypesIdsEnum.xgboost, ModelTypesIdsEnum.knn]
    samples, features_amount, classes = 50000, 10, 2
    chain = chain_template_balanced_tree(model_types=model_types, depth=4, models_per_level=[8, 4, 2, 1],
                                         samples=samples, features=features_amount)
    show_chain_template(chain)
    runs = 30

    roc_train, roc_test = [], []
    for run in range(runs):
        fit_template(chain_template=chain, classes=classes)
        real = real_chain(chain)
        features, target = synthetic_dataset(samples_amount=samples,
                                             features_amount=features_amount,
                                             classes_amount=classes)
        target = np.expand_dims(target, axis=1)
        data_test = InputData(idx=np.arange(0, samples),
                              features=features, target=target)
        synth_target = real.predict(input_data=data_test).predict
        synth_labels = to_labels(synth_target)
        data = InputData(idx=np.arange(0, samples),
                         features=features, target=synth_labels)
        logit = sklearn_model_by_type(model_type=ModelTypesIdsEnum.logit)
        train, test = train_test_data_setup(data)
        fitted_model, predict_train = logit.fit(data=train)
        roc_score = roc_auc(y_true=train.target,
                            y_score=predict_train)
        print(f'Roc train: {roc_score}')
        roc_train.append(roc_score)

        predict_test = logit.predict(fitted_model=fitted_model, data=test)
        roc_score = roc_auc(y_true=test.target,
                            y_score=predict_test)
        print(f'Roc test: {roc_score}')
        roc_test.append(roc_score)

    print(f'ROC on train: {np.mean(roc_train)}+/ {np.std(roc_train)}')
    print(f'ROC on test: {np.mean(roc_test)}+/ {np.std(roc_test)}')


def default_run():
    model_types = [ModelTypesIdsEnum.logit, ModelTypesIdsEnum.xgboost, ModelTypesIdsEnum.knn]
    samples, features_amount, classes = 50000, 10, 2

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
    synth_target = real.predict(input_data=data_test).predict
    synth_labels = to_labels(synth_target)
    data = InputData(idx=np.arange(0, samples),
                     features=features, target=synth_labels)
    logit = sklearn_model_by_type(model_type=ModelTypesIdsEnum.logit)
    train, test = train_test_data_setup(data)
    fitted_model, predict_train = logit.fit(data=train)
    roc_score = roc_auc(y_true=train.target,
                        y_score=predict_train)
    print(f'Roc train: {roc_score}')
    predict_test = logit.predict(fitted_model=fitted_model, data=test)
    roc_score = roc_auc(y_true=test.target,
                        y_score=predict_test)
    print(f'Roc test: {roc_score}')


if __name__ == '__main__':
    robust_test()
