import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score as roc_auc

from core.composer.chain import Chain
from core.composer.node import NodeGenerator
from core.models.data import train_test_data_setup
from core.repository.model_types_repository import ModelTypesIdsEnum
from utilities.synthetic.chain import separately_fit_chain
from utilities.synthetic.data import (
    classification_dataset, gauss_quantiles_dataset, synthetic_benchmark_dataset
)


def data_generator_example():
    samples_total, features_amount = 100, 10
    classes = 2
    options = {
        'informative': 8,
        'redundant': 1,
        'repeated': 1,
        'clusters_per_class': 1
    }
    features, target = classification_dataset(samples_total, features_amount, classes,
                                              features_options=options,
                                              noise_fraction=0.1, full_shuffle=False)

    plt.subplot(121)
    plt.title("Two informative features, one cluster per class")
    plt.scatter(features[:, 0], features[:, 1], marker='o', c=target,
                s=25, edgecolor='k')

    features, target = gauss_quantiles_dataset(samples_total, features_amount=2, classes_amount=classes)
    plt.subplot(122)
    plt.title("Gaussian divided into three quantiles")
    plt.scatter(features[:, 0], features[:, 1], marker='o', c=target,
                s=25, edgecolor='k')

    plt.show()


def synthetic_benchmark_composing_example():
    fitted_chain = separately_fit_chain(samples=5000, features_amount=10,
                                        classes=2)
    data = synthetic_benchmark_dataset(samples_amount=5000, features_amount=10,
                                       fitted_chain=fitted_chain)

    train, test = train_test_data_setup(data)
    simple_chain = two_level_chain()
    simple_chain.fit(input_data=train, use_cache=False)
    train_predicted = simple_chain.predict(train)
    test_predicted = simple_chain.predict(test)


def two_level_chain():
    first = NodeGenerator.primary_node(model_type=ModelTypesIdsEnum.logit)
    second = NodeGenerator.primary_node(model_type=ModelTypesIdsEnum.knn)
    third = NodeGenerator.secondary_node(model_type=ModelTypesIdsEnum.xgboost,
                                         nodes_from=[first, second])

    chain = Chain()
    for node in [first, second, third]:
        chain.add_node(node)

    return chain


def roc_value(chain: Chain, dataset_to_validate) -> float:
    predicted = chain.predict(dataset_to_validate)
    roc_auc_value = roc_auc(y_true=dataset_to_validate.target,
                            y_score=predicted.predict)
    return roc_auc_value


if __name__ == '__main__':
    data_generator_example()
    synthetic_benchmark_composing_example()
