import numpy as np

from core.composer.chain import Chain
from core.composer.node import preprocessing_for_tasks
from core.models.data import InputData
from core.models.preprocessing import Normalization
from core.repository.task_types import MachineLearningTasksEnum
from utilities.synthetic.chain import separately_fit_chain
from utilities.synthetic.data import classification_dataset


def synthetic_benchmark_dataset(samples_amount: int, features_amount: int,
                                fitted_chain: Chain = None) -> InputData:
    """
    Generates a binary classification benchmark dataset that was obtained using
    the (TODO: add. reference) proposed fitting schema.
    :param samples_amount: Total amount of samples in the resulted dataset.
    :param features_amount: Total amount of features per sample.
    :param fitted_chain: Chain with separately fitted models.
    If None then 3-level balanced tree were fitted and taken as a default.
    :return: Benchmark dataset that is ready to be used by Chain.
    """
    if fitted_chain is None:
        fitted_chain = _default_chain()
    options = {
        'informative': 10,
        'redundant': 0,
        'repeated': 0,
        'clusters_per_class': 1
    }

    features, target = classification_dataset(samples_amount=samples_amount,
                                              features_amount=features_amount,
                                              classes_amount=2,
                                              features_options=options)
    target = np.expand_dims(target, axis=1)

    task_type = MachineLearningTasksEnum.classification
    train = InputData(idx=np.arange(0, samples_amount),
                      features=features, target=target, task_type=task_type)

    synth_target = fitted_chain.predict(input_data=train).predict
    synth_labels = _to_labels(synth_target)
    data_synth_train = InputData(idx=np.arange(0, samples_amount),
                                 features=features, target=synth_labels, task_type=task_type)

    # TODO: fix preproc issues
    preprocessing_for_tasks[MachineLearningTasksEnum.classification] = Normalization

    fitted_chain.fit_from_scratch(input_data=data_synth_train)

    features, target = classification_dataset(samples_amount=samples_amount,
                                              features_amount=features_amount,
                                              classes_amount=2,
                                              features_options=options)
    target = np.expand_dims(target, axis=1)
    test = InputData(idx=np.arange(0, samples_amount),
                     features=features, target=target, task_type=task_type)
    synth_target = fitted_chain.predict(input_data=test).predict
    synth_labels = _to_labels(synth_target)
    data_synth_final = InputData(idx=np.arange(0, samples_amount),
                                 features=features, target=synth_labels, task_type=task_type)

    return data_synth_final


def _default_chain():
    chain = separately_fit_chain(samples=5000, features_amount=10,
                                 classes=2)
    return chain


def _to_labels(predictions):
    thr = 0.5
    labels = [0 if val <= thr else 1 for val in predictions]
    labels = np.expand_dims(np.array(labels), axis=1)
    return labels
