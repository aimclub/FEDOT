import numpy as np

from fedot.core.composer.chain import Chain
from fedot.core.composer.composer import ComposerRequirements, DummyChainTypeEnum, \
    DummyComposer
from fedot.core.models.data import InputData, train_test_data_setup
from fedot.core.repository.quality_metrics_repository import ClassificationMetricsEnum, \
    MetricsRepository
from test.models.test_split_train_test import get_roc_auc_value, get_synthetic_input_data


def compose_chain(data: InputData) -> Chain:
    dummy_composer = DummyComposer(DummyChainTypeEnum.hierarchical)
    composer_requirements = ComposerRequirements(primary=['kmeans', 'kmeans'],
                                                 secondary=['logit'])

    metric_function = MetricsRepository().metric_by_id(ClassificationMetricsEnum.ROCAUC)

    chain = dummy_composer.compose_chain(data=data,
                                         initial_chain=None,
                                         composer_requirements=composer_requirements,
                                         metrics=metric_function, is_visualise=False)
    return chain


def test_chain_with_clusters_fit_correct():
    mean_roc_on_test = 0

    # mean ROC AUC is analysed because of stochastic clustering
    for _ in range(5):

        data = get_synthetic_input_data(n_samples=10000)

        chain = compose_chain(data=data)
        train_data, test_data = train_test_data_setup(data)

        chain.fit(input_data=train_data)
        _, roc_on_test = get_roc_auc_value(chain, train_data, test_data)
        mean_roc_on_test = np.mean([mean_roc_on_test, roc_on_test])

    roc_threshold = 0.5
    assert mean_roc_on_test > roc_threshold
