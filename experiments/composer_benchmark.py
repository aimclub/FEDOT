from collections import Counter

import numpy as np
from sklearn.metrics import roc_auc_score as roc_auc

from core.chain_validation import validate
from core.composer.gp_composer.gp_composer import GPComposer, GPComposerRequirements
from core.composer.visualisation import ComposerVisualiser
from core.models.data import InputData, train_test_data_setup
from core.models.model import Model
from core.repository.dataset_types import NumericalDataTypesEnum, CategoricalDataTypesEnum
from core.repository.model_types_repository import (
    ModelTypesIdsEnum, ModelTypesRepository, ModelMetaInfoTemplate)
from core.repository.quality_metrics_repository import (
    MetricsRepository, ClassificationMetricsEnum)
from core.repository.task_types import MachineLearningTasksEnum
from experiments.chain_template import (
    chain_template_balanced_tree, show_chain_template,
    real_chain, fit_template
)
from experiments.generate_data import synthetic_dataset, gauss_quantiles
from experiments.viz import show_fitness_by_generations


def to_labels(predictions):
    thr = 0.5
    labels = [0 if val <= thr else 1 for val in predictions]
    labels = np.expand_dims(np.array(labels), axis=1)
    return labels


def data_robust_test():
    np.random.seed(42)
    model_types = [ModelTypesIdsEnum.logit, ModelTypesIdsEnum.xgboost, ModelTypesIdsEnum.knn]
    samples, features_amount, classes = 10000, 10, 2
    chain = chain_template_balanced_tree(model_types=model_types, depth=4, models_per_level=[8, 4, 2, 1],
                                         samples=samples, features=features_amount)
    show_chain_template(chain)
    runs = 30

    with_gaussian = True

    roc_train, roc_test = [], []
    for run in range(runs):
        fit_template(chain_template=chain, classes=classes, with_gaussian=with_gaussian)
        real = real_chain(chain)
        if with_gaussian:
            features, target = gauss_quantiles(samples_amount=samples,
                                               features_amount=features_amount,
                                               classes_amount=classes)
        else:
            features, target = synthetic_dataset(samples_amount=samples,
                                                 features_amount=features_amount,
                                                 classes_amount=classes)
        target = np.expand_dims(target, axis=1)
        task_type = MachineLearningTasksEnum.classification
        data_test = InputData(idx=np.arange(0, samples),
                              features=features, target=target, task_type=task_type)
        synth_target = real.predict(input_data=data_test).predict
        synth_labels = to_labels(synth_target)
        data = InputData(idx=np.arange(0, samples),
                         features=features, target=synth_labels, task_type=task_type)
        roc_train_, roc_test_ = predict_with_xgboost(data)
        roc_train.append(roc_train_)
        roc_test.append(roc_test_)
    print(f'ROC on train: {np.mean(roc_train)}+/ {np.std(roc_train)}')
    print(f'ROC on test: {np.mean(roc_test)}+/ {np.std(roc_test)}')
    roc_diff = [train_ - test_ for train_, test_ in zip(roc_train, roc_test)]
    print(f'ROC diff: {roc_diff}')
    print(f'Max diff: {np.max(roc_diff)}')


def default_run():
    model_types = [ModelTypesIdsEnum.logit, ModelTypesIdsEnum.xgboost, ModelTypesIdsEnum.knn]
    samples, features_amount, classes = 10000, 10, 2

    # chain = chain_template_random(model_types=model_types, depth=3, models_per_level=2,
    #                               samples=samples, features=features_amount)
    chain = chain_template_balanced_tree(model_types=model_types, depth=4, models_per_level=[8, 4, 2, 1],
                                         samples=samples, features=features_amount)
    show_chain_template(chain)
    fit_template(chain_template=chain, classes=classes, with_gaussian=True)
    real = real_chain(chain)
    validate(real)
    features, target = synthetic_dataset(samples_amount=samples,
                                         features_amount=features_amount,
                                         classes_amount=classes)
    target = np.expand_dims(target, axis=1)
    task_type = MachineLearningTasksEnum.classification
    data_test = InputData(idx=np.arange(0, samples),
                          features=features, target=target, task_type=task_type)
    synth_target = real.predict(input_data=data_test).predict
    synth_labels = to_labels(synth_target)
    data = InputData(idx=np.arange(0, samples),
                     features=features, target=synth_labels, task_type=task_type)
    print(predict_with_log_reg(data))


def predict_with_log_reg(data):
    logit = Model(model_type=ModelTypesIdsEnum.logit)
    train, test = train_test_data_setup(data)
    fitted_model, predict_train = logit.fit(data=train)
    roc_train = roc_auc(y_true=train.target,
                        y_score=predict_train)
    print(f'Roc train: {roc_train}')
    predict_test = logit.predict(fitted_model=fitted_model, data=test)
    roc_test = roc_auc(y_true=test.target,
                       y_score=predict_test)
    print(f'Roc test: {roc_test}')

    return roc_train, roc_test


def predict_with_xgboost(data):
    xgboost = Model(model_type=ModelTypesIdsEnum.xgboost)
    train, test = train_test_data_setup(data)
    fitted_model, predict_train = xgboost.fit(data=train)
    roc_train = roc_auc(y_true=train.target,
                        y_score=predict_train)
    print(f'Roc train: {roc_train}')
    predict_test = xgboost.predict(fitted_model=fitted_model, data=test)
    roc_test = roc_auc(y_true=test.target,
                       y_score=predict_test)
    print(f'Roc test: {roc_test}')

    return roc_train, roc_test


def data_distribution():
    samples, features_amount, classes = 10000, 10, 2
    features, target = synthetic_dataset(samples_amount=samples,
                                         features_amount=features_amount,
                                         classes_amount=classes)
    labels = list(to_labels(target).flatten())
    half = len(labels) // 2
    full_counter = Counter(labels)
    first_part = Counter(labels[:half])
    second_part = Counter(labels[half:])
    print(f'Full: {full_counter}')
    print(f'First part: {first_part}')
    print(f'Second part: {second_part}')


def models_to_use():
    models = [ModelTypesIdsEnum.logit, ModelTypesIdsEnum.xgboost, ModelTypesIdsEnum.knn,
              ModelTypesIdsEnum.qda, ModelTypesIdsEnum.dt]
    return models


def composer_robust_test():
    dataset_to_compose, data_to_validate = train_test_data_setup(data_by_synthetic_chain(with_gaussian=True))

    models_repo = ModelTypesRepository()
    available_model_types, _ = models_repo.search_models(
        desired_metainfo=ModelMetaInfoTemplate(input_type=NumericalDataTypesEnum.table,
                                               output_type=CategoricalDataTypesEnum.vector,
                                               task_type=MachineLearningTasksEnum.classification,
                                               can_be_initial=True,
                                               can_be_secondary=True))
    available_model_types = models_to_use()
    metric_function = MetricsRepository().metric_by_id(ClassificationMetricsEnum.ROCAUC)

    composer_requirements = GPComposerRequirements(
        primary=available_model_types,
        secondary=available_model_types, max_arity=2,
        max_depth=4, pop_size=2, num_of_generations=5,
        crossover_prob=0.8, mutation_prob=0.8)
    composer = GPComposer()
    print('Starting to compose:')
    chain_evo_composed = composer.compose_chain(data=dataset_to_compose,
                                                initial_chain=None,
                                                composer_requirements=composer_requirements,
                                                metrics=metric_function, is_visualise=False)
    history = composer.history
    show_fitness_by_generations([history])
    chain_evo_composed.fit(input_data=dataset_to_compose, verbose=True)
    ComposerVisualiser.visualise(chain_evo_composed)

    predicted_train = chain_evo_composed.predict(dataset_to_compose)
    predicted_test = chain_evo_composed.predict(data_to_validate)
    # the quality assessment for the simulation results
    roc_train = roc_auc(y_true=dataset_to_compose.target,
                        y_score=predicted_train.predict)

    roc_test = roc_auc(y_true=data_to_validate.target,
                       y_score=predicted_test.predict)
    print(f'Train ROC: {roc_train}')
    print(f'Test ROC: {roc_test}')


def composer_multiple_run(runs=2):
    history_by_runs = []
    generations, pop_size = 5, 3
    for run in range(runs):
        dataset_to_compose, data_to_validate = train_test_data_setup(data_by_synthetic_chain(with_gaussian=True))

        models_repo = ModelTypesRepository()
        available_model_types, _ = models_repo.search_models(
            desired_metainfo=ModelMetaInfoTemplate(input_type=NumericalDataTypesEnum.table,
                                                   output_type=CategoricalDataTypesEnum.vector,
                                                   task_type=MachineLearningTasksEnum.classification,
                                                   can_be_initial=True,
                                                   can_be_secondary=True))
        available_model_types = models_to_use()
        metric_function = MetricsRepository().metric_by_id(ClassificationMetricsEnum.ROCAUC)

        composer_requirements = GPComposerRequirements(
            primary=available_model_types,
            secondary=available_model_types, max_arity=2,
            max_depth=4, pop_size=pop_size, num_of_generations=generations,
            crossover_prob=0.8, mutation_prob=0.8)
        composer = GPComposer()
        print('Starting to compose:')
        chain_evo_composed = composer.compose_chain(data=dataset_to_compose,
                                                    initial_chain=None,
                                                    composer_requirements=composer_requirements,
                                                    metrics=metric_function, is_visualise=False)
        history_by_runs.append(composer.history)
        chain_evo_composed.fit(input_data=dataset_to_compose, verbose=True)
        ComposerVisualiser.visualise(chain_evo_composed)

        predicted_train = chain_evo_composed.predict(dataset_to_compose)
        predicted_test = chain_evo_composed.predict(data_to_validate)
        # the quality assessment for the simulation results
        roc_train = roc_auc(y_true=dataset_to_compose.target,
                            y_score=predicted_train.predict)

        roc_test = roc_auc(y_true=data_to_validate.target,
                           y_score=predicted_test.predict)
        print(f'Train ROC: {roc_train}')
        print(f'Test ROC: {roc_test}')

    reduced = [_reduced_history_best(history, generations, pop_size) for history in history_by_runs]
    show_fitness_by_generations(reduced, generations)


def _reduced_history_best(history, generations, pop_size):
    reduced = []
    for gen in range(generations):
        fitness_values = [individ[1] for individ in history[gen: gen + pop_size]]
        best = min(fitness_values)
        print(f'Min in generation #{gen}: {best}')
        reduced.append(best)

    return reduced


def data_by_synthetic_chain(with_gaussian=False):
    model_types = [ModelTypesIdsEnum.logit, ModelTypesIdsEnum.xgboost, ModelTypesIdsEnum.knn]
    samples, features_amount, classes = 10000, 10, 2

    chain = chain_template_balanced_tree(model_types=model_types, depth=4, models_per_level=[8, 4, 2, 1],
                                         samples=samples, features=features_amount)
    show_chain_template(chain)
    fit_template(chain_template=chain, classes=classes, with_gaussian=with_gaussian)
    real = real_chain(chain)
    validate(real)
    task_type = MachineLearningTasksEnum.classification
    features, target = synthetic_dataset(samples_amount=samples,
                                         features_amount=features_amount,
                                         classes_amount=classes)
    target = np.expand_dims(target, axis=1)
    data_test = InputData(idx=np.arange(0, samples),
                          features=features, target=target, task_type=task_type)
    synth_target = real.predict(input_data=data_test).predict
    synth_labels = to_labels(synth_target)
    data = InputData(idx=np.arange(0, samples),
                     features=features, target=synth_labels, task_type=task_type)

    return data


def composer_test_with_gaussian():
    dataset_to_compose, data_to_validate = train_test_data_setup(data_by_synthetic_chain())

    models_repo = ModelTypesRepository()
    available_model_types, _ = models_repo.search_models(
        desired_metainfo=ModelMetaInfoTemplate(input_type=NumericalDataTypesEnum.table,
                                               output_type=CategoricalDataTypesEnum.vector,
                                               task_type=MachineLearningTasksEnum.classification,
                                               can_be_initial=True,
                                               can_be_secondary=True))

    metric_function = MetricsRepository().metric_by_id(ClassificationMetricsEnum.ROCAUC)

    composer_requirements = GPComposerRequirements(
        primary=available_model_types,
        secondary=available_model_types, max_arity=2,
        max_depth=4, pop_size=5, num_of_generations=10,
        crossover_prob=0.8, mutation_prob=0.8)
    composer = GPComposer()
    print('Starting to compose:')
    chain_evo_composed = composer.compose_chain(data=dataset_to_compose,
                                                initial_chain=None,
                                                composer_requirements=composer_requirements,
                                                metrics=metric_function, is_visualise=True)
    chain_evo_composed.fit(input_data=dataset_to_compose, verbose=True)
    ComposerVisualiser.visualise(chain_evo_composed)

    predicted_train = chain_evo_composed.predict(dataset_to_compose)
    predicted_test = chain_evo_composed.predict(data_to_validate)
    # the quality assessment for the simulation results
    roc_train = roc_auc(y_true=dataset_to_compose.target,
                        y_score=predicted_train.predict)

    roc_test = roc_auc(y_true=data_to_validate.target,
                       y_score=predicted_test.predict)
    print(f'Train ROC: {roc_train}')
    print(f'Test ROC: {roc_test}')


if __name__ == '__main__':
    composer_multiple_run()
    # default_run()
    # data_robust_test()
