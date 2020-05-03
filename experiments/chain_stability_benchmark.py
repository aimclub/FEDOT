import csv
import itertools
from copy import deepcopy
from random import seed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score as roc_auc

from core.models.data import InputData
from core.models.data import train_test_data_setup
from core.repository.model_types_repository import ModelTypesIdsEnum
from core.repository.task_types import MachineLearningTasksEnum
from experiments.chain_template import (chain_template_balanced_tree, fit_template,
                                        show_chain_template, real_chain, with_calculated_shapes)
from experiments.composer_benchmark import to_labels, predict_with_xgboost
from experiments.generate_data import synthetic_dataset

np.random.seed(42)
seed(42)


def models_to_use():
    models = [ModelTypesIdsEnum.logit, ModelTypesIdsEnum.xgboost, ModelTypesIdsEnum.knn,
              ModelTypesIdsEnum.dt]
    return models


def source_chain(model_types, samples, features, classes):
    template = chain_template_balanced_tree(model_types=model_types, depth=4, models_per_level=[8, 4, 2, 1],
                                            samples=samples, features=features)
    show_chain_template(template)
    fit_template(template, classes=classes, skip_fit=False, with_gaussian=True)
    initialized_chain = real_chain(template)

    return initialized_chain, template


def data_generated_by(chain, samples, features_amount, classes):
    task_type = MachineLearningTasksEnum.classification
    features, target = synthetic_dataset(samples_amount=samples,
                                         features_amount=features_amount,
                                         classes_amount=classes)
    target = np.expand_dims(target, axis=1)
    train = InputData(idx=np.arange(0, samples),
                      features=features, target=target, task_type=task_type)
    synth_target = chain.predict(input_data=train).predict
    synth_labels = to_labels(synth_target)
    data_synth_train = InputData(idx=np.arange(0, samples),
                                 features=features, target=synth_labels, task_type=task_type)

    # data_synth_train.features = preprocessing.StandardScaler().fit_transform(data_synth_train.features)

    chain.fit_from_scratch(input_data=data_synth_train)

    features, target = synthetic_dataset(samples_amount=samples,
                                         features_amount=features_amount,
                                         classes_amount=classes)
    target = np.expand_dims(target, axis=1)
    test = InputData(idx=np.arange(0, samples),
                     features=features, target=target, task_type=task_type)
    synth_target = chain.predict(input_data=test).predict
    synth_labels = to_labels(synth_target)
    data_synth_test = InputData(idx=np.arange(0, samples),
                                features=features, target=synth_labels, task_type=task_type)
    return data_synth_test


def roc_score(chain, data_to_compose, data_to_validate):
    predicted_train = chain.predict(data_to_compose)

    predicted_test = chain.predict(data_to_validate)
    # the quality assessment for the simulation results

    for i in range(len(data_to_compose.target)):
        if data_to_compose.target[i] != predicted_train.predict[i]:
            print(f'PROBLEM, {i}')
            # raise ValueError(i)

    roc_train = roc_auc(y_true=data_to_compose.target,
                        y_score=predicted_train.predict)

    roc_test = roc_auc(y_true=data_to_validate.target,
                       y_score=predicted_test.predict)
    print(f'Train ROC: {roc_train}')
    print(f'Test ROC: {roc_test}')

    return roc_train, roc_test


def remove_first_node(template, samples, features):
    if len(template[0]) == 0:
        del template[0]
    node_to_remove = template[0][0]
    all_nodes = list(itertools.chain.from_iterable(template))
    childs = [node for node in all_nodes if node_to_remove in node.parents]
    for child in childs:
        child.parents.remove(node_to_remove)
    template[0].remove(node_to_remove)
    features_shape = [samples, features]
    target_shape = [samples, 1]
    fixed_template = with_calculated_shapes(template,
                                            source_features=features_shape,
                                            source_target=target_shape)

    fixed_chain = real_chain(fixed_template)
    return fixed_template, fixed_chain


def write_header_to_csv(f):
    with open(f, 'w', newline='') as file:
        writer = csv.writer(file, quoting=csv.QUOTE_NONE)
        writer.writerow(['iter', 'test_auc'])


def add_result_to_csv(f, iter, roc_auc_test):
    with open(f, 'a', newline='') as file:
        writer = csv.writer(file, quoting=csv.QUOTE_NONE)
        writer.writerow([iter, roc_auc_test])


if __name__ == '__main__':
    exp_file_name = 'chain_stab_res.csv'
    write_header_to_csv(exp_file_name)

    for exp in range(10):
        samples, features_amount, classes = 10000, 10, 2
        chain, template = source_chain(model_types=models_to_use(),
                                       samples=samples, features=features_amount,
                                       classes=classes)

        data_synth_test = data_generated_by(chain, samples, features_amount, classes)
        train, test = train_test_data_setup(data_synth_test)

        if False:
            pred_res = chain.predict(data_synth_test)

            for i in range(len(data_synth_test.target)):
                if data_synth_test.target[i] != to_labels(pred_res.predict)[i]:
                    print(i)
                    raise ValueError("!1")

            pred_res = chain.predict(data_synth_test)

            for i in range(len(data_synth_test.target)):
                if data_synth_test.target[i] != to_labels(pred_res.predict)[i]:
                    print(i)
                    raise ValueError("!2")

            # train=data_synth_test
            print("!!!")
            train = InputData(features=data_synth_test.features[0:1000], target=data_synth_test.target[0:1000],
                              idx=data_synth_test.idx[0:1000], task_type=data_synth_test.task_type)

            pred_res = chain.predict(train)

            for i in range(len(train.target)):
                if train.target[i] != to_labels(pred_res.predict)[i]:
                    print(i)
                    raise ValueError("!3")

            # exit(0)
            # exit(0)

        roc_train, roc_test = roc_score(chain, train, test)
        add_result_to_csv(exp_file_name, 0, roc_test)

        prev_chain, prev_template = deepcopy(chain), deepcopy(template)
        iterations = chain.length // 2

        _, roc_test_ = predict_with_xgboost(data_synth_test)
        add_result_to_csv(exp_file_name, "BL", roc_test_)

        for iter in range(iterations):
            print(f'Iteration  #{iter}')
            new_template, new_chain = remove_first_node(prev_template, samples=samples, features=features_amount)
            new_chain.fit_from_scratch(train)
            roc_train, roc_test = roc_score(new_chain, train, test)
            prev_chain, prev_template = new_chain, new_template
            add_result_to_csv(exp_file_name, iter + 1, roc_test)

        chain_stab_data_base = pd.read_csv(exp_file_name, delimiter=',')
        chain_stab_data = chain_stab_data_base[chain_stab_data_base.iter != 'BL']
        p = sns.lineplot(chain_stab_data.iter, chain_stab_data.test_auc)
        p.set(xlabel='Str. diff', ylabel='ROC AUC')
        baselines = chain_stab_data_base[chain_stab_data_base.iter == 'BL'].test_auc

        crit_high = plt.hlines(np.mean(baselines) + np.std(baselines),
                               xmin=0, xmax=6, colors='red', linestyles='dashed')
        crit_low = plt.hlines(np.mean(baselines) - np.std(baselines),
                              xmin=0, xmax=6, colors='red', linestyles='dashed')
        crit = plt.hlines(np.mean(baselines),
                          xmin=0, xmax=6, colors='red', linestyles='solid')

        plt.ylim(0.6, 1.01)
        plt.show()
